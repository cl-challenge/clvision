import importlib
from utils.utils import *


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # benchmark parameter
    # ------------------------------------------------
    parser.add_argument('--data_path', type=str, default='/home/miil/Dataset/clvision')
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2, 3], help='which track run')
    parser.add_argument('--model', type=str, default='resnet34', help='model name for calling timm') # https://rwightman.github.io/pytorch-image-models/models/
    parser.add_argument('--use_pretrain', type=str2bool, default=True)
    parser.add_argument('--use_val', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=-1)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--schedule', type=str, default='Milestone', choices=['Step', 'Milestone'])
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 160])
    parser.add_argument('--step', type=int, default=40)
    parser.add_argument('--decay', type=float, default=0.0002)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--train_batch', type=int, default=256)
    parser.add_argument('--test_batch', type=int, default=256)


    # instance classification parameter
    # ------------------------------------------------
    parser.add_argument('--lr_base', type=float, default=0.001)
    parser.add_argument('--lr_cf', type=float, default=0.01)
    parser.add_argument('--use_cutmix', type=str2bool, default=False)
    parser.add_argument('--use_bn', type=str2bool, default=True)

    # instance detection parameter
    # ------------------------------------------------
    parser.add_argument('--lr', type=float, default=0.005)

    # use avalanche parameter
    # ------------------------------------------------
    parser.add_argument('--plugins', nargs='+', default=['ReplayPlugin'], help='use exact name for calling plugin in avalanche') # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    parser.add_argument('--hp_plugins', nargs='+', default=[{'mem_size':2000}], help='hyper-parameter for plugin. it\'s len is must same with num of plugin')
    parser.add_argument('--strategy', type=str, default='Naive')
    parser.add_argument('--hp_strategy', default=None)

    # etc option
    # ------------------------------------------------
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--memo', default=None, help='memo for experiment explanation')
    parser.add_argument('--project_name', default='clvision-debug', help='wandb project name')

    return parser

if __name__ == '__main__':
    args = get_command_line_parser().parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.gpu = set_gpu(args)
    args.device = torch.device("cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    if args.exp == 1:
        importlib.import_module('starting_template_instance_classification').main(args)
    elif args.exp == 2:
        importlib.import_module('starting_template_category_detection').main(args)
    elif args.exp == 3:
        importlib.import_module('starting_template_instance_detection').main(args)
    else:
        raise ValueError



