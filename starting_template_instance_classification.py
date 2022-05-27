################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-02-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
Starting template for the "object classification - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified.
- Don't remove the mandatory plugin (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the accuracy metric. You
    are free to add more metrics or change the logger.
- The use of Avalanche training and logging code is not mandatory. However,
    you are required to use the given benchmark generation procedure. If not
    using Avalanche, make sure you are following the same train/eval loop and
    please make sure you are able to export the output in the expected format.
"""

import datetime
import time
from pathlib import Path
from typing import List
import sys

import timm
import torch
import torchvision.models
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger, TextLogger
from avalanche.training.plugins import *
from avalanche.training.supervised import *
from devkit_tools.benchmarks import challenge_classification_benchmark
from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter
from devkit_tools.plugins.model_checkpoint import *

from utils.utils import *
from utils.data import transformation
from utils.cutmix_utils import *
from utils.custom_plugin import EvalMode, CutMixPlugin
# from avalanche.models.dynamic_optimizers import


def get_optimizer(args, model):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD([{'params': model.layer1.parameters(), 'lr': args.lr_base},
                                     {'params': model.layer2.parameters(), 'lr': args.lr_base},
                                     {'params': model.layer3.parameters(), 'lr': args.lr_base},
                                     {'params': model.layer4.parameters(), 'lr': args.lr_base},
                                     {'params': model.fc.parameters(), 'lr': args.lr_cf},
                                     ],
                                    momentum=0.9, nesterov=True, weight_decay=args.decay)


    else:
        raise NotImplementedError

    if args.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    elif args.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                         gamma=args.gamma)
    else:
        raise NotImplementedError

    return optimizer, scheduler

def main(args):
    # --- TRANSFORMATIONS
    train_transform, eval_transform = transformation()

    # --- BENCHMARK CREATION
    benchmark = challenge_classification_benchmark(
        dataset_path=args.data_path,
        train_transform=train_transform,
        eval_transform=eval_transform,
        n_validation_videos=args.use_val,
        validation_video_selection_seed=args.seed
    )

    # --- EXP NAME CREATION
    name = exp_name(args)
    result_path = './results/instance_classification_results/{}'.format(name)
    ensure_path(result_path)

    # --- MODEL CREATION
    model = timm.create_model(args.model, pretrained=args.use_pretrain)
    model.fc = torch.nn.Linear(model.fc.in_features, out_features=benchmark.n_classes)
    optimizer, scheduler = get_optimizer(args, model)


    # --- PLUGINS CREATION  https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    mandatory_plugins = [ClassificationOutputExporter(benchmark, save_folder=result_path)]
    utils_plugin = [LRSchedulerPlugin(scheduler, step_granularity='epoch'),
                    ]
    # algo_plugin = [getattr(sys.modules[__name__], p)(**args.hp_plugins[idx]) for idx, p in enumerate(args.plugins)]
    algo_plugin = [
        # EWCPlugin(ewc_lambda=0.4),
        # ReplayPlugin(mem_size=2000)
        GEMPlugin(patterns_per_experience=256, memory_strength=0.5),
        # SynapticIntelligencePlugin()
    ]

    plugins = algo_plugin + utils_plugin + mandatory_plugins
    if not args.use_bn:
        plugins += [EvalMode()]

    # --- METRICS AND LOGGING
    evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True, experience=True),
        loss_metrics(minibatch=False, epoch_running=True, experience=True),
        # confusion_matrix_metrics(stream=True),
        timing_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(),
                 # TensorboardLogger(tb_log_dir='./results/tblog/track_inst_cls/exp_' + datetime.datetime.now().isoformat()),
                 # WandBLogger(project_name=args.project_name, run_name=name, save_code=False, sync_tfboard=True, dir='./results/'),
                 # TextLogger(open('./results/txtlog/'+ name+ datetime.datetime.now().isoformat() + '.txt', 'a'))
                 ],)

    """
    # --- CREATE THE STRATEGY INSTANCE
    # In Avalanche, you can customize the training loop in 3 ways:
    #   1. Adapt the make_train_dataloader, make_optimizer, forward,
    #   criterion, backward, optimizer_step (and other) functions. This is the
    #   clean way to do things!
    #   2. Change the loop itself by reimplementing training_epoch or even
    #   _train_exp (not recommended).
    #   3. Create a Plugin that, by implementing the proper callbacks,
    #   can modify the behavior of the strategy.
    #  -------------
    #  Consider that popular strategies (EWC, LwF, Replay) are implemented
    #  as plugins. However, writing a plugin from scratch may be a tad
    #  tedious. For the challenge, we recommend going with the 1st option. 
    #  In particular, you can create a subclass of the SupervisedTemplate
    #  (Naive is mostly an alias for the SupervisedTemplate) and override only
    #  the methods required to implement your solution.
    """
    if args.use_cutmix:
        criterion = CutMixCrossEntropyLoss(True)
        plugins += [
            CutMixPlugin(benchmark.n_classes, num_mix=2, beta=1.0, prob=0.5)
        ]
    else:
        criterion = CrossEntropyLoss()


    # MAKE STRATEGY
    if args.strategy == 'Naive':
        strategy_args = {'model': model, 'optimizer': optimizer, 'criterion': criterion,
                         'train_mb_size':args.train_batch, 'train_epochs':args.epoch, 'eval_mb_size':args.test_batch,
                         'device': args.device, 'plugins': plugins, 'evaluator': evaluator, 'eval_every':args.eval_every}
        if args.hp_strategy:
            cl_strategy = getattr(sys.modules[__name__], args.strategy)(**strategy_args, **args.hp_strategy)
        else:
            cl_strategy = getattr(sys.modules[__name__], args.strategy)(**strategy_args)
    elif args.strategy == 'ICaRL': # Not used
        # model.features
        # model.classifier 되는지 확인
        encoder = [{'params': model.layer1.parameters()}, {'params': model.layer2.parameters()},
                   {'params': model.layer3.parameters()}, {'params': model.layer4.parameters()}]
        fc = model.fc.parameters()
        cl_strategy = ICaRL(feature_extractor=encoder, classifier=fc, criterion=criterion, optimizer=optimizer,
                            memory_size=100, buffer_transform=train_transform, fixed_memory=False)
    elif args.strategy == 'EWC':
        cl_strategy = EWC(
            model, optimizer, criterion, ewc_lambda=0.4, mode="online", decay_factor=0.1, train_mb_size=args.train_batch,
            eval_mb_size=args.test_batch, train_epochs=args.epoch, device=args.device, plugins = plugins, evaluator=evaluator,
            eval_every=args.eval_every
        )
    elif args.strategy == 'Cumulative':
        cl_strategy = Cumulative(
            model, optimizer, criterion, train_mb_size=args.train_batch, eval_mb_size=args.test_batch, train_epochs=args.epoch,
            device=args.device, plugins = plugins, evaluator=evaluator, eval_every=args.eval_every
        )
    elif args.strategy == 'AGEM':
        pass
    else:
        raise NotImplementedError

    # TRAINING LOOP
    for experience in benchmark.train_stream:
        start_time = time.time()
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)
        print("Current Classes: ", experience.classes_in_this_experience)

        data_loader_arguments = dict(num_workers=10, persistent_workers=True)

        if 'valid' in benchmark.streams:
            cl_strategy.train(
                experience,
                eval_streams=[benchmark.valid_stream[current_experience_id]],
                **data_loader_arguments)
        else:
            cl_strategy.train(
                experience,
                **data_loader_arguments)
        # cl_strategy.save()

        print("Training completed")

        print("Computing accuracy on the complete test set")
        cl_strategy.eval(benchmark.test_stream, num_workers=10,
                         persistent_workers=True)

        print('This task takes %d seconds' % (time.time() - start_time),
              '\nstill need around %.2f mins to finish this session' % (
                      (time.time() - start_time) * (14 - current_experience_id) / 60))

