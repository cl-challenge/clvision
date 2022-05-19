import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Resize
from torch.utils.data.dataset import Dataset
import numpy as np
import random

from avalanche.benchmarks.utils import Compose
from utils.cutmix_utils import onehot, rand_bbox


def transformation():
    # This is the normalization used in torchvision models
    # https://pytorch.org/vision/stable/models.html
    torchvision_normalization = transforms.Normalize(
        mean=[0.4097, 0.3659, 0.3349],
        std=[0.2449, 0.2412, 0.2417])

    # Add additional transformations here
    train_transform = Compose(
        [Resize(256),
        RandomCrop(224, padding=10, pad_if_needed=True),
         RandomHorizontalFlip(0.5),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Compose(
        [ToTensor(), torchvision_normalization]
    )
    # ---------

    return train_transform, eval_transform



class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Resize, Compose

    sample_classification_dataset = torchvision.datasets.ImageFolder(root='/home/miil/Dataset/clvision',
                                                                     transform=Compose([Resize([256, 256]),ToTensor()]))
    dataloader = DataLoader(sample_classification_dataset, batch_size=256, shuffle=True, num_workers=16)

    def get_mean_and_std(dataloader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for i, (x, y) in enumerate(dataloader):
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(x, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(x ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std

    m, std = get_mean_and_std(dataloader)
    print(m, std)
    # result : tensor([0.4097, 0.3659, 0.3349]) tensor([0.2449, 0.2412, 0.2417])
