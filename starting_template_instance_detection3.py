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
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised.naive_object_detection import \
	ObjectDetectionTemplate
from devkit_tools.benchmarks import challenge_instance_detection_benchmark
from devkit_tools.metrics.detection_output_exporter import \
	make_ego_objects_metrics
from devkit_tools.metrics.dictionary_loss import dict_loss_metrics

from examples.tvdetection.transforms import RandomHorizontalFlip, ToTensor

# 임의추가
import torchvision.models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
from utils.utils import *
import time
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger, TextLogger
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
# CHG: added DetentionMetrics
from avalanche.evaluation.metrics.detection import DetectionEvaluator, SupportedDatasetApiDef, DetectionMetrics

# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = Path.home() / '3rd_clvision_challenge' / 'challenge'

# This sets the root logger to write to stdout (your console).
# Customize the logging level as you wish.
logging.basicConfig(level=logging.NOTSET)


def main(args):
	print(f"args.exp_name:{args.exp_name}")
	# --- TRANSFORMATIONS
	# Add additional transformations here
	# You can take some detection transformations here:
	# https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
	# Beware that:
	# - transforms found in torchvision.transforms.transforms will only act on
	#    the image and they will not adjust bounding boxes accordingly: don't
	#    use them (apart from ToTensor)!
	# - make sure you are using the "Compose" from avalanche.benchmarks.utils,
	#    not the one from torchvision or from the aforementioned link.
	train_transform = Compose(
		[ToTensor(), RandomHorizontalFlip(0.5)]
	)
	
	# Don't add augmentation transforms to the eval transformations!
	eval_transform = Compose(
		[ToTensor()]
	)
	# ---------
	
	# --- BENCHMARK CREATION
	benchmark = challenge_instance_detection_benchmark(
		dataset_path=args.data_path,
		train_transform=train_transform,
		eval_transform=eval_transform,
		n_validation_videos=1 if args.type == "tune" else 0,
		validation_video_selection_seed=1337
	)
	
	# ---------
	
	# --- exp name
	name = exp_name_track3(args)
	# ---------
	
	# --- MODEL CREATION
	# Load a model pre-trained on COCO
	if args.model == "fasterrcnn_resnet50":
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
		pretrained=True)
	
	num_classes = benchmark.n_classes + 1  # N classes + background
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	
	# model = model.to(device)
	print('Num classes (including background)', num_classes)
	# --- OPTIMIZER AND SCHEDULER CREATION
	
	# Create the optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
										 momentum=0.9, weight_decay=1e-5)
	if args.schedule == 'Step':
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
	elif args.schedule == 'Milestone':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
																		 gamma=args.gamma)
	
	
	
	# Define the scheduler
	train_mb_size = args.train_batch
	
	# When using LinearLR, the LR will start from optimizer.lr / start_factor
	# (here named warmup_factor) and will then increase after each call to
	# scheduler.step(). After start_factor steps (here called warmup_iters),
	# the LR will be set optimizer.lr and never changed again.
	warmup_factor = 1.0 / 1000
	warmup_iters = \
		min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)
	
	lr_scheduler = torch.optim.lr_scheduler.LinearLR(
		optimizer, start_factor=warmup_factor, total_iters=warmup_iters
	)
	# ---------
	
	# TODO: ObjectDetectionTemplate == Naive == plain fine tuning without
	#  replay, regularization, etc.
	# For the challenge, you'll have to implement your own strategy (or a
	# strategy plugin that changes the behaviour of the ObjectDetectionTemplate)
	
	# --- PLUGINS CREATION
	# Avalanche already has a lot of plugins you can use!
	# Many mainstream continual learning approaches are available as plugins:
	# https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
	
	# Note on LRSchedulerPlugin
	# Consider that scheduler.step() may be called after each epoch or
	# iteration, depending on the needed granularity. In the Torchvision
	# object detection tutorial, in the train_one_epoch function, step() is
	# called after each iteration. In addition, the scheduler is only used in
	# the very first epoch. The same setup is here replicated.
	ensure_path('./results/instance_detection_results/{}'.format(name))
	mandatory_plugins = []
	
	plugins: List[SupervisedPlugin] = [
													 LRSchedulerPlugin(
														 lr_scheduler, step_granularity='iteration',
														 first_exp_only=True, first_epoch_only=True),
													 # ...
												 ] + mandatory_plugins
	# ---------
	
	# --- METRICS AND LOGGING
	mandatory_metrics = [
		make_ego_objects_metrics(
			save_folder=f'{args.exp_name}/instance_detection_results',
			filename_prefix='track3_output')]
	
	evaluator = EvaluationPlugin(
		mandatory_metrics,
		timing_metrics(
			experience=True,
			stream=True
		),
		loss_metrics(
			minibatch=True,
			epoch_running=True,
		),
		dict_loss_metrics(
			minibatch=True,
			epoch_running=True,
			epoch=True,
			dictionary_name='detection_loss_dict'
		),
		loggers=[InteractiveLogger(),
					TensorboardLogger(
						tb_log_dir='./log/track_inst_det/exp_' +
									  datetime.datetime.now().isoformat()),
					WandBLogger(project_name='clvision-track3', run_name=name, save_code=False, sync_tfboard=True,
									dir='./results/'),
					],
		benchmark=benchmark
	)
	# ---------
	
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
	#  In particular, you can create a subclass of this ObjectDetectionTemplate
	#  and override only the methods required to implement your solution.
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
	# ---------
	
	data_loader_arguments = dict(
		num_workers=args.num_workers,
		persistent_workers=True
	)
	if args.type == "tune":
		# HP Tuning: OPTUNA
		study = optuna.create_study(direction='maximize', sampler=TPESampler())
		study_trial = 3  # 몇 번 탐색할지
		
		def objective(trial: Trial):
			# 탐색할 파라미터 구간
			config = {
				'lr': trial.suggest_discrete_uniform('lr', 0.01, 0.11, 0.05),
				'momentum': trial.suggest_discrete_uniform('momentum', 0.89, 0.99, 0.05),
				'optimizer': trial.suggest_categorical('optimizer', [torch.optim.SGD, torch.optim.RMSprop]),
				'n_epochs': trial.suggest_int('n_epochs', 1, 3, 1),
				
			}
			config_exp = {
				'lr': 0.005,
				'momentum': 0.9,
				'optimizer': torch.optim.SGD,
				'n_epochs': trial.suggest_int('n_epochs', 1, 3, 1),
				
			}
			model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
				pretrained=True)
			num_classes = benchmark.n_classes + 1  # N classes + background
			in_features = model.roi_heads.box_predictor.cls_score.in_features
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
			params = [p for p in model.parameters() if p.requires_grad]
			optimizer = config_exp['optimizer'](params, weight_decay=1e-5,
															lr=config_exp["lr"], momentum=config_exp["momentum"])
			
			cl_strategy = ObjectDetectionTemplate(
				model=model,
				optimizer=optimizer,
				train_mb_size=8,
				train_epochs=config_exp["n_epochs"],
				eval_mb_size=8,
				device=torch.device("cuda"),
				plugins=plugins,
				evaluator=evaluator,
				eval_every=0 if 'valid' in benchmark.streams else -1)
			
			print("Starting experiment...")
			valid_ap = []
			for experience in benchmark.train_stream:
				start_time = time.time()
				current_experience_id = experience.current_experience
				print("Start of experience: ", current_experience_id)
				
				res = cl_strategy.train(
					benchmark.valid_stream[current_experience_id],  # 원래대로 라면 여기 train_stream 들어가야 되는데 빠르게 확인위해서 제거
					eval_streams=[benchmark.valid_stream[current_experience_id]],
					**data_loader_arguments)
				m = res[f'LvisMetrics/eval_phase/valid_stream/Exp00{current_experience_id}/bbox/AP']
				valid_ap.append(m)
			
			score = np.mean(valid_ap)
			
			return score
		
		study.optimize(lambda trial: objective(trial), n_trials=study_trial)
		
		best_param = study.best_trial.params
		print(study.trials_dataframe())
		print(f"BEST_Param: {best_param}")
		lr = best_param["lr"]
		momentum = best_param["momentum"]
	
	lr = 0.005
	momentum = 0.9
	weight_decay = 1e-5
	
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
		pretrained=True)
	
	num_classes = benchmark.n_classes + 1  # N classes + background
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	# Create the optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params,
										 lr=lr, momentum=momentum, weight_decay=1e-5)
	train_mb_size = args.train_batch
	
	warmup_factor = 1.0 / 1000
	warmup_iters = \
		min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)
	lr_scheduler = torch.optim.lr_scheduler.LinearLR(
		optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
	
	cl_strategy = ObjectDetectionTemplate(
		model=model,
		optimizer=optimizer,
		train_mb_size=args.train_batch,
		train_epochs=args.epoch,
		eval_mb_size=args.test_batch,
		device=torch.device("cuda"),
		plugins=plugins,
		evaluator=evaluator,
		eval_every=0 if 'valid' in benchmark.streams else -1)
	
	# TRAINING LOOP
	print("Starting experiment...")
	for experience in benchmark.train_stream:
		start_time = time.time()
		current_experience_id = experience.current_experience
		print("Start of experience: ", current_experience_id)
		# print("Current Classes: ", experience.classes_in_this_experience)
		
		cl_strategy.train(
			experience,
			**data_loader_arguments)
		print("Training completed")
		print('This task takes %d seconds' % (time.time() - start_time))
		
		print("Computing accuracy on the full test set")
		cl_strategy.eval(benchmark.test_stream, num_workers=args.num_workers,
							  persistent_workers=True)
