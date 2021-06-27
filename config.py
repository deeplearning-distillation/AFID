import os
import sys
import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

def set_gpu(gpu_id: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


class Config(object):
    config_txt = open('config.json', 'r', encoding='utf8')
    config_json = json.load(config_txt)

    GPU = config_json['GPU']
    log = config_json['train']['log']
    checkpoint_path = config_json['train']['checkpoint_path']
    resume = config_json['train']['resume']
    evaluate = None
    train_dataset_path = config_json['train']['dataset_train_path']
    val_dataset_path = config_json['val']['dataset_val_path']
    network = config_json['network']
    pretrained = False
    num_classes = config_json['num_classes']
    seed = config_json['seed']
    input_image_size = config_json['image_size']
    scale = 256 / input_image_size

    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomResizedCrop(input_image_size), 
            transforms.RandomHorizontalFlip(),              
            transforms.ToTensor(),                          
            transforms.Normalize(mean=[0.485, 0.456, 0.406],# x = (x - mean(x))/std(x)
                                 std=[0.229, 0.224, 0.225]),
        ]))
    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    milestones = config_json['train']['milestones']
    epochs = config_json['train']['epochs']
    batch_size = config_json['train']['batch_size']
    accumulation_steps = config_json['train']['accumulation_steps']
    lr = config_json['train']['learning_rate']
    weight_decay = config_json['train']['weight_decay']
    momentum = config_json['train']['momentum']
    num_workers = config_json['num_workers']
    print_interval = config_json['train']['print_interval']
    apex = config_json['train']['apex']

    config_txt.close()
