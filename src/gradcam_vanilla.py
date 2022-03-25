#Grad-CAM tutorial
#https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchsummary import summary

#custom methods
import custom_utilities as cu

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset``
# ImageNet과 마찬가지로 경로 : ../../ ... /data/ 로 끝나야 하고,
# ImageNet과 마찬가지로 class 이름의 directory 하나 파서 이미지 1개 넣기.
# example : /src/data/Elephant/0001.png 에 이미지 넣고,
# root = '...../src/data' 로 설정.
dataset = datasets.ImageFolder(
    root='/home/jwb/clvision-v1/tutorial_data/data', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


class VGG_pretrained(nn.Module):
    def __init__(self):
        super(VGG_pretrained, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)

        # disect the network to access its last convolutional layer
        # 마지막 convolutional layer 에서 network 절단,
        # VGG 에서는 maxpooling 직전 layer.
        # 하단의 print(vgg.vgg) 통해서 확인 가능.
        self.features_conv = self.vgg.features[:36]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    # 간단한 hook. Class & context manager 화 할 예정.
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# initialize the VGG model
vgg = VGG_pretrained()

# 원래 vgg19 layer 모양 확인.
# (35): ReLU(inplace=True) 까지 포함되도록 절단.
print("=======original VGG shape============")
print(vgg.vgg)

# __init__() 안에 있는 vgg.feature_conv 통해서
# 절단된 vgg19 network 확인 가능.
print("=======[:36] 절단된 VGG============")
print(vgg.features_conv)

# set the evaluation mode
vgg.eval()

# get the image from the dataloader
img, _ = next(iter(dataloader))

# 1000개 class에 대한 예측치 전부.
pred = vgg(img)
print('Pred shape: ',pred.shape)
# get the most likely prediction of the model
predicted = vgg(img).argmax(dim=1)
print('Predicted shape: ',predicted)

#print(pred.shape)

# get the gradient of the output with respect to the parameters of the model
# argmax를 통해 가장 가능성 높다고 판별한 logit's index 획득,
# 그 logit에 대해서 backward() 진행, gradient 획득.
pred[:, predicted].backward()

# pull the gradients out of the model
gradients = vgg.get_activations_gradient()
print('Gradients shape: ',gradients.shape)

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
print('Pooled gradient shape: ', pooled_gradients.shape)

# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()
print('last layer activation shape: ', activations.shape)

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# 그냥 쓰게되면, 음수 값이 큰 값들이 같이 표시되어서 보기 안좋음.
# -relu(x) 하게 된다면 inverse heat map? cold map? 확인 가능.
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

print('heatmap shape: ', heatmap.shape)

# draw the heatmap
cu.generate_heatmap_png(img, heatmap, "elephant")


