import torch
import numpy as np
import random
import torch.nn as nn
from avalanche.core import SupervisedPlugin, Template

class EvalMode(SupervisedPlugin):
    def __init__(self, start_exp=1):
        super().__init__()
        self.start_exp = start_exp

    def before_training_exp(self, strategy: Template, *args, **kwargs):
        super().before_training_exp(strategy, *args, **kwargs)
        # exp 1부터는 eval mode
        curr_exp = strategy.experience.current_experience

        if curr_exp < self.start_exp:
            print("=====Train Mode=====")
            strategy.model.train()
        else:
            print("=====Eval Mode=====")
            # strategy.model.eval()
            for m in strategy.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class CutMixPlugin(SupervisedPlugin):
    def __init__(self, num_class, beta, prob, num_mix, device='cuda'):
        super().__init__()
        self.num_class = num_class
        self.beta = beta
        self.prob = prob
        self.num_mix = num_mix
        self.device = device

    def before_training_iteration(self, strategy: Template, *args, **kwargs):
        length = strategy.mbatch[0].shape[0]
        data = torch.zeros((length,3,224,224)).to(self.device)
        label = torch.zeros((length, 1110)).to(self.device)

        for index in range(length):
            img, lb = strategy.mb_x[index], strategy.mb_y[index]
            lb_onehot = self.onehot(self.num_class, lb).to(self.device)

            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue

                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_index = random.choice(range(length))

                img2, lb2 = strategy.mb_x[rand_index], strategy.mb_y[rand_index]
                lb2_onehot = onehot(self.num_class, lb2).to(self.device)

                bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
                img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)
                data[index], label[index] = img, lb_onehot

        strategy.mbatch[0], strategy.mbatch[1] = data, label
        # strategy.mb_x, strategy.mb_y = data, label

    def onehot(self, size, target):
        vec = torch.zeros(size, dtype=torch.float32)
        vec[target] = 1.
        return vec

    def rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

__all__ = [
    'EvalMode',
    'CutMixPlugin'
]
