from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks.utils import Compose

def transformation():
    # This is the normalization used in torchvision models
    # https://pytorch.org/vision/stable/models.html
    torchvision_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]) #todo 이 값도 정확한건가? train data 사용해서 확인해볼 필요도 있을 듯

    # Add additional transformations here
    train_transform = Compose(
        [RandomCrop(224, padding=10, pad_if_needed=True),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor(), torchvision_normalization]
    )
    # ---------

    return train_transform, eval_transform