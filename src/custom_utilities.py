import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchsummary import summary


def generate_heatmap_png(img, heatmap, class_name=None):
    '''
    heapmap shape : torch.Size([n, n]) 이어야 함!
    class_name : 선택.
    
    returns nothing!
    
    output:
    1. n by n original heatmap
    2. normalized image
    3. overlay 해서 합쳐진 최종 결과
    
    '''

    heat_map = plt.matshow(heatmap.squeeze())
    if(class_name == None):
        plt.savefig("heatmap_n_by_n.png")
    else:
        plt.savefig("heatmap_n_by_n_{}.png".format(class_name))

    original_image = plt.imshow(img.squeeze().permute(1, 2, 0))
    if(class_name == None):
        plt.savefig("normalized_image.png")
    else:
        plt.savefig("normalized_image_{}.png".format(class_name))

    #fig = plt.figure()
    add_original = plt.imshow(img.squeeze().permute(1, 2, 0))
    add_heatmap = plt.imshow(heatmap.squeeze(),
                             alpha=0.6,
                             extent=(0, 224, 224, 0),
                             interpolation='bilinear',
                             cmap='magma')
    if(class_name == None):
        plt.savefig("final_heatmap_overlay.png")
    else:
        plt.savefig("final_heatmap_overlay_{}.png".format(class_name))

def img_rotation(img, degree):
    '''
    img : PIL Image or Tensor
    degree : float
    
    return rotated image
    '''
    rotated_img = torchvision.transforms.functional.rotate(img, degree)
    return rotated_img

def heatmap_rotation(hmap, num, clockwise=False):
    '''
    hmap : torch.size([n,n])
    num : number of 90deg rotations
    clockwise : default as false.
    '''
    rotate_val = 0
    if(clockwise):
        rotate_val = -1
    else:
        rotate_val = 1
    
    rotate_val *= num
    
    heatmap_rot90 = torch.rot90(hmap,rotate_val,[0,1])
    
    return heatmap_rot90