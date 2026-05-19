import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(project_root)

import argparse

from sven.trainer import PrefixTrainer, TextPromptTrainer
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--output_dir', type=str, default='../trained/')
    parser.add_argument('--model_type', type=str, default='prefix')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--n_prefix_token', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--kl_loss_ratio', type=int, default=None) # will be divided by 1000
    parser.add_argument('--learning_rate', type=float, default=None)

    parser.add_argument('--contrastive_loss_ratio', type=int, default=400) # will be divided by 100
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, choices=['prog', 'line', 'char', 'mix'], default='mix')
    parser.add_argument('--lm_loss_ratio', type=int, default=1)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)

    # upsampling arguments
    """
    --sampling_size:
        the size of sampling, <=0 means no sampling
        dataset.Upsampler._upsample_all_prob: the percentange of the sampled sec dataset compared to the func dataset
        dataset.Upsampler._upsample_minority: sample classes with <sampling_size samples to sampling_size
    --sampling_weight:
        select the mode of how the sampling weight of each cwe at lang is calcualted when doing -all sempling modes:
            uniform: each example is treated equally and uniform sampling is employed across the whole dataset
            inverse-prop: cwes with less example are more likely to get sampled, chosing this balances the cwes
    --cwes:
        select a list of the cwes you want to upsample, or select all
    --langs:
        select a list of the langs you want to include in upsampling, or select all
    """
    parser.add_argument('--sampling_size', type=int, default=-1)
    parser.add_argument('--sampling_method', type=str, choices=['uniform', 'inverse-prop', 'minority'], default='minority')
    parser.add_argument('--cwes', type=str, nargs='*', default=['all'])
    parser.add_argument('--langs', type=str, nargs='*', default=['all'])

    args = parser.parse_args()

    if args.pretrain_dir is None:
        args.pretrain_dir = '2b'
    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    if args.n_prefix_token is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':
            args.n_prefix_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-3B' or args.pretrain_dir == 'bigcode/starcoder2-3b':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-7B' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-6.7b-base' or args.pretrain_dir == 'codellama/CodeLlama-7b-hf'\
             or args.pretrain_dir == 'ByteDance-Seed/Seed-Coder-8B-Base':
            args.n_prefix_token = 12
        else:
            assert False

    if args.num_train_epochs is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':
            args.num_train_epochs = 8
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-3B' or args.pretrain_dir == 'bigcode/starcoder2-3b':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-7B' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-6.7b-base' or args.pretrain_dir == 'codellama/CodeLlama-7b-hf'\
            or args.pretrain_dir == 'ByteDance-Seed/Seed-Coder-8B-Base':
            args.num_train_epochs = 5
        else:
            assert False

    if args.kl_loss_ratio is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-1.3b-base':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-3B' or args.pretrain_dir == 'bigcode/starcoder2-3b':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi' or args.pretrain_dir == 'Qwen/Qwen2.5-Coder-7B' or args.pretrain_dir == 'deepseek-ai/deepseek-coder-6.7b-base' or args.pretrain_dir == 'codellama/CodeLlama-7b-hf'\
            or args.pretrain_dir == 'ByteDance-Seed/Seed-Coder-8B-Base':
            args.kl_loss_ratio = 2000
        else:
            assert False

    if args.model_type == 'prefix':
        if args.learning_rate is None:
            args.learning_rate = 1e-2

        if args.contrastive_loss_ratio == 0:
            args.learning_rate = 5e-2
            args.grad_acc_steps = args.grad_acc_steps * 2

        if args.model_type == 'prefix' and args.diff_level in ('prog', 'line'):
            args.learning_rate = 1e-3
    elif args.model_type == 'text':
        args.learning_rate = 5e-5

    args.output_dir = os.path.join(args.output_dir, args.output_name)
    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args)

    if args.model_type == 'prefix':
        trainer = PrefixTrainer(args)
    elif args.model_type == 'text':
        trainer = TextPromptTrainer(args)
    else:
        raise NotImplementedError()

    trainer.run()

if __name__ == '__main__':
    main()