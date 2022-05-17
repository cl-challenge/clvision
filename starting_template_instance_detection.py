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
Starting template for the "object detection - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified.
- Don't remove the mandatory metric (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the detection metrics. You
    are free to add more metrics or change the logger.
- The use of Avalanche training and logging code is not mandatory. However,
    you are required to use the given benchmark generation procedure. If not
    using Avalanche, make sure you are following the same train/eval loop and
    please make sure you are able to export the output in the expected format.
"""

import argparse
import datetime
import logging
from pathlib import Path
from typing import List

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from avalanche.benchmarks.utils import Compose
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import timing_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, EWCPlugin, ReplayPlugin
from avalanche.training.supervised.naive_object_detection import \
    ObjectDetectionTemplate
from devkit_tools.benchmarks import challenge_instance_detection_benchmark
from devkit_tools.metrics.detection_output_exporter import \
    make_ego_objects_metrics
from devkit_tools.metrics.dictionary_loss import dict_loss_metrics

from examples.tvdetection.transforms import RandomHorizontalFlip, ToTensor
from utils.utils import *

import sys

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler


# This sets the root logger to write to stdout (your console).
# Customize the logging level as you wish.
logging.basicConfig(level=logging.NOTSET)


def get_optimizer(args, model):
    if args.optim == 'SGD':
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr,
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


    # Define the scheduler
    # train_mb_size = 4

    # When using LinearLR, the LR will start from optimizer.lr / start_factor
    # (here named warmup_factor) and will then increase after each call to
    # scheduler.step(). After start_factor steps (here called warmup_iters),
    # the LR will be set optimizer.lr and never changed again.
    # warmup_factor = 1.0 / 1000
    # warmup_iters = min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)

    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    # )

    return optimizer, scheduler


def main(args):
    # --- TRANSFORMATIONS
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    eval_transform = Compose([ToTensor()])


    # --- BENCHMARK CREATION
    benchmark = challenge_instance_detection_benchmark(
        dataset_path=args.data_path,
        train_transform=train_transform,
        eval_transform=eval_transform,
        n_validation_videos=args.use_val,
        validation_video_selection_seed=args.seed)


    # --- EXP NAME CREATION
    name = exp_name(args)
    result_path = './results/instance_detection_results/{}'.format(name)
    ensure_path(result_path)


    # --- MODEL CREATION
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=args.use_pretrain)
    num_classes, in_features = benchmark.n_classes + 1, model.roi_heads.box_predictor.cls_score.in_features # N classes + background
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(args.device)
    print('Num classes (including background)', num_classes)

    # --- OPTIMIZER AND SCHEDULER CREATION
    optimizer, scheduler = get_optimizer(args, model)


    # --- PLUGINS CREATION
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins

    # --- PLUGINS CREATION     https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    mandatory_plugins = []
    lr_plugin = [LRSchedulerPlugin(scheduler, step_granularity='iteration', first_exp_only=False, first_epoch_only=True)]
    algo_plugin = [getattr(sys.modules[__name__], p)(**args.hp_plugins[idx]) for idx, p in enumerate(args.plugins)]

    plugins = algo_plugin + lr_plugin + mandatory_plugins

    # --- METRICS AND LOGGING
    mandatory_metrics = [make_ego_objects_metrics(save_folder=result_path, filename_prefix='track3_output')]

    evaluator = EvaluationPlugin(
        mandatory_metrics,
        timing_metrics(experience=True, stream=True),
        loss_metrics(minibatch=True, epoch_running=True),
        dict_loss_metrics(minibatch=True, epoch_running=True, epoch=True, dictionary_name='detection_loss_dict'),
        loggers=[InteractiveLogger(),
                 TensorboardLogger(tb_log_dir='./results/tblog/track_inst_det/exp_' + datetime.datetime.now().isoformat()),
                 WandBLogger(project_name='track3_{}'.format(args.project_name), run_name=name, save_code=False, sync_tfboard=True,
                             dir='./results/'),
                 TextLogger(open('./results/txtlog/' + name + datetime.datetime.now().isoformat() + '.txt', 'a'))],
        benchmark=benchmark
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    cl_strategy = ObjectDetectionTemplate(
        model=model,
        optimizer=optimizer,
        train_mb_size=args.train_batch,
        train_epochs=args.epoch,
        eval_mb_size=args.test_batch,
        device=args.device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=0 if 'valid' in benchmark.streams else -1
    )

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)

        data_loader_arguments = dict(num_workers=10, persistent_workers=True)
        if 'valid' in benchmark.streams:
            cl_strategy.train(experience, eval_streams=[benchmark.valid_stream[current_experience_id]], **data_loader_arguments)
        else:
            cl_strategy.train(experience, **data_loader_arguments)
        print("Training completed")

        print("Computing accuracy on the full test set")
        cl_strategy.eval(benchmark.test_stream, cur_exp=experience.current_experience, num_workers=10)
