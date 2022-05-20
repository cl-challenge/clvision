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
            strategy.model.eval()

__all__ = [
    'EvalMode'
]
