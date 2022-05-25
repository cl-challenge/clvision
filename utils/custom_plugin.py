import torch.nn as nn
from avalanche.core import SupervisedPlugin, Template
from data import CutMix

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
    def __init__(self, num_class, beta, prob, num_mix):
        super().__init__()
        self.num_class = num_class
        self.beta = beta
        self.prob = prob
        self.num_mix = num_mix

    def train_dataset_adaptation(self, **kwargs):
        self.adapted_dataset = CutMix(self.experience.dataset, num_class=self.num_class, beta=self.beta, prob=self.prob,
                                      num_mix=self.num_mix)
        self.adapted_dataset.current_transform_group = 'train'

    # def after_train_dataset_adaptation(self, s):

__all__ = [
    'EvalMode'
]
