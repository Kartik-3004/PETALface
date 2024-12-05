import argparse
import sys
import os
from time import gmtime, strftime

def get_args():

    parser = argparse.ArgumentParser(add_help=False)
    ## Partial FC ##
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--interclass_filtering_threshold', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    ## Logging ##
    parser.add_argument('--verbose', type=int, default=2000)
    parser.add_argument('--frequent', type=int, default=10)
    parser.add_argument('--seed', default=2048, type=int)
    ## Dali ##
    parser.add_argument('--dali', type=bool, default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    ## Wandb ##
    parser.add_argument('--wandb_key', type=str)
    parser.add_argument('--suffix_run_name', type=str, default="None")
    parser.add_argument('--using_wandb', default=False, type=bool)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_log_all', type=bool, default=False)
    parser.add_argument('--save_artifacts', type=bool, default=False)
    parser.add_argument('--wandb_resume', type=bool, default=False)
    parser.add_argument('--notes', type=str, default="Lorem Ipsum")
    parser.add_argument('--dir', type=str)
    ## Data ## 
    parser.add_argument('--val_targets', type=str, default='lfw,cfp_fp,agedb_30')
    parser.add_argument('--train_data', type=str, default='webface4m', help='ms1mv3 or webface4m')
    parser.add_argument('--image_size', type=int, default=112, help='112 or 120')
    ## Training ##
    parser.add_argument('--network', type=str, default='r50')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', default=0, type=int) 
    parser.add_argument('--dali_aug', action="store_true")
    parser.add_argument('--head', type=str, default='partial_fc')
    parser.add_argument('--use_lora', action="store_true")
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_scale', type=int, default=1)
    parser.add_argument('--load_pretrained', type=str)
    ## Data webface4m ##
    parser.add_argument('--rec', type=str, default='/data/knaraya4/data/WebFace4M')
    parser.add_argument('--num_classes', type=int, default=205990)
    parser.add_argument('--num_image', type=int, default=4235242)
    ## Loss ##
    parser.add_argument('--margin_list', type=str, default='1.0,0.5,0.0')
    ## Logging ##
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save_all_states', type=bool, default=True)
    parser.add_argument('--output', type=str)
    ## Optimizer ##
    parser.add_argument('--optimizer', type=str, default='sgd', help="sgd or adamW")
    parser.add_argument('--lr', type=float, default=0.1, help='0.1 or 0.001')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='5e-4 or 0.1')
    parser.add_argument('--gradient_acc', default=1, type=int)
    ## IQA ##
    parser.add_argument('--iqa', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    args.margin_list = [float(x) for x in args.margin_list.split(',')]
    args.val_targets = [str(x) for x in args.val_targets.split(',')]
    return args