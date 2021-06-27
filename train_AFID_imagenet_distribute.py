import sys
import os
import argparse
import random
import time
import warnings
import network
import utils

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from thop import profile
from thop import clever_format
from torch.utils.data import DataLoader
from config import Config, set_gpu
set_gpu(Config.GPU)

from tools import DataPrefetcher, get_logger, AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--network',
                        type=str,
                        default=Config.network,
                        help='name of network')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=Config.input_image_size,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')
    parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')

    return parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return utils.sigmoid_rampup(epoch, args.consistency_rampup)

def train(train_loader, model, module, criterion, criterion_kl, optimizer, optimizer_FM, scheduler, scheduler_FM, epoch, logger,
          args):
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()
    top1_f = AverageMeter()
    top5_f = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    module.train()

    iters = len(train_loader.dataset) // args.batch_size
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1
    consistency_weight = get_current_consistency_weight(epoch)
    while inputs is not None:   # one epoch
        inputs, labels = inputs.cuda(), labels.cuda()

        #####################################################################
        outputs1, outputs2, fmap = model(inputs)
        ensemble_logit = (outputs1 + outputs2) / 2 
        fused_logit = module(fmap[0], fmap[1])  # fused_logit

        loss_ce = criterion(outputs1, labels) + criterion(outputs2, labels) + criterion(fused_logit, labels)
        loss_kl = consistency_weight * (
            criterion_kl(outputs1, fused_logit) + criterion_kl(outputs2, fused_logit) + criterion_kl(fused_logit, ensemble_logit)
        )
        loss = loss_ce + loss_kl
        ####################################################################

        loss = loss / args.accumulation_steps

        loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()    # Updata para
            optimizer.zero_grad()
            optimizer_FM.step()
            optimizer_FM.zero_grad()

        # measure accuracy and record loss
        acc1_1, acc5_1 = accuracy(outputs1, labels, topk=(1, 5))
        acc1_2, acc5_2 = accuracy(outputs2, labels, topk=(1, 5))
        acc1_fused, acc5_fused = accuracy(fused_logit, labels, topk=(1, 5))
        top1_1.update(acc1_1.item(), inputs.size(0))
        top5_1.update(acc5_1.item(), inputs.size(0))
        top1_2.update(acc1_2.item(), inputs.size(0))
        top5_2.update(acc5_2.item(), inputs.size(0))
        top1_f.update(acc1_fused.item(), inputs.size(0))
        top5_f.update(acc5_fused.item(), inputs.size(0))        
        losses.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, n1_top1 acc: {acc1_1.item():.2f}%, n1_top5 acc: {acc5_1.item():.2f}%, n2_top1 acc: {acc1_2.item():.2f}%, n2_top5 acc: {acc5_2.item():.2f}%, f_top1 acc: {acc1_fused.item():.2f}%, f_top5 acc: {acc5_fused.item():.2f}%, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step()
    scheduler_FM.step()

    return (top1_1.avg, top5_1.avg), (top1_2.avg, top5_2.avg), (top1_f.avg, top5_f.avg), losses.avg


def validate(val_loader, model, module, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()
    top1_f = AverageMeter()
    top5_f = AverageMeter()

    # switch to evaluate mode
    model.eval()
    module.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs1, outputs2, fmap = model(inputs)
            fused_logit = module(fmap[0], fmap[1])

            acc1_1, acc5_1 = accuracy(outputs1, labels, topk=(1, 5))
            acc1_2, acc5_2 = accuracy(outputs2, labels, topk=(1, 5))
            acc1_f, acc5_f = accuracy(fused_logit, labels, topk=(1, 5))
            top1_1.update(acc1_1.item(), inputs.size(0))
            top5_1.update(acc5_1.item(), inputs.size(0))
            top1_2.update(acc1_2.item(), inputs.size(0))
            top5_2.update(acc5_2.item(), inputs.size(0))
            top1_f.update(acc1_f.item(), inputs.size(0))
            top5_f.update(acc5_f.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return (top1_1.avg, top5_1.avg), (top1_2.avg, top5_2.avg), (top1_f.avg, top5_f.avg), throughput


def main(logger, args):
    if args.network not in ['resnet18', 'resnet34']:
        raise Exception("network specified is not exit!")

    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(Config.val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    logger.info('finish loading data')

    logger.info(f"creating model '{args.network}'")
    logger.info(f"creating Fusion_module")

    if args.network == 'resnet18':
        model = network.ffl_resnet18(num_classes=Config.num_classes)
    if args.network == 'resnet34':
        model = network.ffl_resnet34(num_classes=Config.num_classes)
    module = network.Fusion_module(512, Config.num_classes, 7)

    flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size) 
    flops, params = profile(model, inputs=(flops_input, ))  # use thop to compute the number of params
    flops, params = clever_format([flops, params], "%.3f")
    logger.info(f"model: '{args.network}', flops: {flops}, params: {params}")

    for name, param in model.named_parameters():            # show layers
        logger.info(f"{name},{param.requires_grad}")

    model = model.cuda()    # load model to GPU
    module = module.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_kl = utils.KLLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(       # adjust learning rate
        optimizer, milestones=args.milestones, gamma=0.1)


    optimizer_FM = torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)

    scheduler_FM = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_FM, milestones=args.milestones, gamma=0.1)

    model = nn.DataParallel(model)  # use parallel strategy
    module = nn.DataParallel(module)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        module.load_state_dict(checkpoint['module_state_dict'])
        (acc1_1, acc5_1), (acc1_2, acc5_2), (acc1_f, acc5_f), throughput = validate(val_loader, model, module, args)
        logger.info(
            f"epoch {checkpoint['epoch']:0>3d}, n1_top1 acc: {acc1_1:.2f}%, n1_top5 acc: {acc5_1:.2f}%, n2_top1 acc: {acc1_2:.2f}%, n2_top5 acc: {acc5_2:.2f}%, f_top1 acc: {acc1_f:.2f}%, f_top5 acc: {acc5_f:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return

    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        module.load_state_dict(checkpoint['module_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer_FM.load_state_dict(checkpoint['optimizer_FM_state_dict'])
        scheduler_FM.load_state_dict(checkpoint['scheduler_FM_state_dict'])

        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"n1_top1_acc: {checkpoint['acc1_1']}%, n2_top1_acc: {checkpoint['acc1_2']}%, f_top1_acc: {checkpoint['acc1_f']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    if (start_epoch == 100):
        start_epoch =1
    for epoch in range(start_epoch, args.epochs + 1):
        (acc1_1, acc5_1), (acc1_2, acc5_2), (acc1_f, acc5_f), losses = train(train_loader, model, module, criterion, criterion_kl, optimizer, optimizer_FM,
                                   scheduler, scheduler_FM, epoch, logger, args)
        logger.info(
            f"train: epoch {epoch:0>3d}, n1_top1 acc: {acc1_1:.2f}%, n1_top5 acc: {acc5_1:.2f}%, n2_top1 acc: {acc1_2:.2f}%, n2_top5 acc: {acc5_2:.2f}%, f_top1 acc: {acc1_f:.2f}%, f_top5 acc: {acc5_f:.2f}%, losses: {losses:.2f}"
        )

        (acc1_1, acc5_1),(acc1_2, acc5_2),(acc1_f, acc5_f), throughput = validate(val_loader, model, module, args)
        logger.info(
            f"val: epoch {epoch:0>3d}, n1_top1 acc: {acc1_1:.2f}%, n1_top5 acc: {acc5_1:.2f}%, n2_top1 acc: {acc1_2:.2f}%, n2_top5 acc: {acc5_2:.2f}%, f_top1 acc: {acc1_f:.2f}%, f_top5 acc: {acc5_f:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'acc1_1': acc1_1,
                'acc1_2': acc1_2,
                'acc1_f': acc1_f,
                'loss': losses,
                'lr': scheduler.get_lr()[0],
                'model_state_dict': model.state_dict(),
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_FM_state_dict': optimizer_FM.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scheduler_FM_state_dict': scheduler_FM.state_dict(),
            }, os.path.join(args.checkpoints, 'latest.pth'))
        '''
        if epoch == args.epochs:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    args.checkpoints,
                    "{}-epoch{}-acc{}.pth".format(args.network, epoch, acc1_f)))
        '''
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)
