# Attention-based Feature Interaction for Efficient Online Knowledge Distillation 

![Framework](https://github.com/deeplearning-distillation/AFID/blob/main/images/AFID.jpg)

We propose a simple but effective online knowledge distillation algorithm, called Attentive Feature Interaction Distillation (AFID). It applies interactive teaching in whcih the
teacher and the student can send, receive, and give feedback on an equal footing, ultimately promoting the generality of both. Specifically, we set up a Feature Interaction Module for two sub-networks to conduct low-level and mid-level feature learning. They can alternately transfer attentive features maps to exchange interested regions and fuse the other party’s map with the features of self extraction for information enhancement.


# Getting Started
The code has been tested using Pytorch1.5.1 and CUDA10.2 on Ubuntu 18.04.
  
> pip install -r requirements.txt

# AFID

You can run the following code to train models on CIFAR-100:

> CUDA_VISIBLE_DEVICES=0 python train_AFID_wrn.py --lr 0.2 --depth 16 --widen_factor 2

You can run the following code to train models on ImageNet 2012 and modify parameters with the file named "config.json"(ONLY imagenet):

> python train_AFID_imagenet_distribute.py
