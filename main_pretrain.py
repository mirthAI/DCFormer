import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from Dataset.Datasets import DATASET
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertModel
from models.ct_clip import CTCLIP
from losses.loss import ClipLoss
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.models_config import Config

from engine_pretrain import train_one_epoch, evaluate
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

def readable_flops(flops):
    units = ["FLOPs", "kFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    for unit in units:
        if flops < 1000:
            return f"{flops:.2f} {unit}"
        flops /= 1000
    return f"{flops:.2f} PFLOPs"
def readable_params(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def get_args_parser():
    parser = argparse.ArgumentParser('DCFormer CLIP Training', add_help=False)


    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--mae', action='store_true')

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    parser.add_argument('--epochs', default=15, type=int)

    parser.add_argument('--num_steps', default=100001, type=int)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='decomp_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--reduction', default='depth')

    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--scheduler', action='store_true')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/blue/weishao/share/CLIPData/CT-Rate/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='RESULTS/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--file', default='ctvit_finetune/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True    
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(DEVICE)

    # if DEVICE == 'cpu':
    #     path = '/Users/gorkemcanates/code/PythonProjects/CT-CLIP-main/data/'
    #     # path = '/Users/gorkemcanates/code/PythonProjects/CT-CLIP-main/data/'
    # else:

    input_shape = (args.input_size, args.input_size, args.input_size//2)
    dataset = DATASET(      args,
                            debug=args.debug,
                            subs=True if args.debug else False,
                            mae=args.mae,
                            train_metadata=args.data_path + 'metadata/train_metadata.csv',
                            val_metadata=args.data_path + "metadata/validation_metadata.csv",
                            non_path_train=args.data_path + 'nones.csv',
                            non_path_val=args.data_path + 'non_nones_val.csv',
                            reports_file_train= args.data_path + "radiology_text_reports/train_reports.csv",
                            reports_file_valid= args.data_path + "radiology_text_reports/validation_reports.csv",
                            train_data_folder= args.data_path + "data_volumes/dataset/train",
                            val_data_folder = args.data_path + "data_volumes/dataset/valid",
                            labels = args.data_path + "valid_labels.csv",
                            target_shape=input_shape
                            )


    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_train = %s" % str(sampler_train))
        sampler_test = torch.utils.data.DistributedSampler(
            dataset.test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        # print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset.train_dataset)
        sampler_test = torch.utils.data.RandomSampler(dataset.test_dataset)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset.train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False if args.debug else True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset.test_dataset, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    criterion = ClipLoss()

    # define the model
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")        
    image_encoder = Config(model=args.model, input_size=args.input_size, reduction=args.reduction)


    model = CTCLIP(
        image_encoder = image_encoder.image_encoder,
        text_encoder = text_encoder,
        dim_image = image_encoder.DIM_IMAGE,
        dim_text = 768,
        dim_latent = 512,
        reduction=args.reduction,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
        )

    # with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #     f.write(json.dumps(log_stats) + "\n")
    flops = FlopCountAnalysis(image_encoder.image_encoder.encoder, torch.rand(1, 1, args.input_size // 2, args.input_size, args.input_size))
    flop_table = flop_count_table(flops)
    params_all = readable_params(sum(p.numel() for p in model.parameters() if p.requires_grad))
    params = readable_params(sum(p.numel() for p in image_encoder.image_encoder.encoder.parameters() if p.requires_grad))
    flop_total = readable_flops(flops.total())

    print('PARAMS CLIP: ', params_all)
    print('PARAMS Image Encoder: ', params)
    print('FLOPS: ', flop_total)  
    
    model_stats = {'FLOP TABLE': flop_table}
    params_stats = {'PARAMS CLIP': params_all, 'PARAMS Encoder': params, 'FLOPS': flop_total}
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(params_stats) + "\n")
    f.close()

    with open(os.path.join(args.output_dir, "summary.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(model_stats) + "\n")


    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    print(len(data_loader_train))
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)


        train_stats = train_one_epoch(
            model, tokenizer, data_loader_train,
            optimizer, device, epoch, criterion, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if args.distributed:
            data_loader_val.sampler.set_epoch(epoch)            
        test_stats = evaluate(epoch, data_loader_val, model, tokenizer, device, args.output_dir)
        
        print(f"Accuracy of the network on the {len(dataset.test_dataset)} test images: {test_stats['acc']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        # log_stats = {'Epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()},
        #                 }
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.device == 'cpu':
    #     args.data_path = '/Volumes/weishao/share/CLIPData/CT-Rate/'
    args.output_dir += args.file + '/'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 
