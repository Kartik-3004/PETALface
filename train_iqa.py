import argparse
import logging
import os
from datetime import datetime, timedelta
from skimage import img_as_float32
from brisque import BRISQUE

import numpy as np
import torch
import random
import config
import torchvision
from backbones import get_model
from heads import get_head
from dataset.dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from torch import distributed
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import pyiqa

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
distributed.init_process_group("nccl")


def generate_alpha(img, iqa, thresh):
    device = img.device
    BS, C, H, W = img.shape
    alpha = torch.zeros((BS, 1), dtype=torch.float32, device=device)

    score = iqa(img)
    threshold = thresh
    for i in range(BS):
        if score[i] == threshold:
            alpha[i] = 0.5
        elif score[i] < threshold:
            alpha[i] = 0.5 - (threshold - score[i])
        else:
            alpha[i] = 0.5 + (score[i] - threshold)
    return alpha

def main(args):
    setup_seed(seed=args.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(args.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if args.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=args.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if args.suffix_run_name is None else run_name + f"_{args.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = args.wandb_entity, 
                project = args.wandb_project, 
                sync_tensorboard = True,
                resume=args.wandb_resume,
                name = run_name, 
                notes = args.notes) if rank == 0 or args.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(args)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        args.rec,
        local_rank,
        args.batch_size,
        args.image_size,
        args.dali,
        args.dali_aug,
        args.seed,
        args.num_workers
    )

    backbone = get_model(args.network, dropout=0.0, fp16=args.fp16, num_features=args.embedding_size, r=args.lora_rank, scale=args.lora_scale, use_lora=args.use_lora).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        args.margin_list[0],
        args.margin_list[1],
        args.margin_list[2],
        args.interclass_filtering_threshold
    )
    head = get_head(args.head,
            margin_loss=margin_loss, embedding_size=args.embedding_size, num_classes=args.num_classes,
            sample_rate=args.sample_rate, fp16=False)

    if args.use_lora:
        weights_path = os.path.join(args.load_pretrained, f"checkpoint_gpu_{rank}.pt")
        if os.path.isfile(weights_path):
            dict_checkpoint = torch.load(weights_path)
            backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"], strict=False)
        else:
            dict_checkpoint = torch.load(os.path.join(args.load_pretrained, f"model.pt"))
            backbone.module.load_state_dict(dict_checkpoint, strict=False) 
        for p in head.parameters():
            p.requires_grad = True
        for p in backbone.parameters():
            p.requires_grad = False
        for name, p in backbone.named_parameters():
            if 'trainable_lora' in name:
                p.requires_grad = True



    if args.optimizer == "sgd":
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad)
        logging.info("Total Parameters: %d", total_params)
        logging.info('Number of trainable parameters: %d', trainable_params)
        head.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": filter(lambda p: p.requires_grad, backbone.parameters()) }, {"params": head.parameters()}],
            lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad)
        logging.info("Total Parameters: %d", total_params)
        logging.info('Number of trainable parameters: %d', trainable_params)
        head.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": filter(lambda p: p.requires_grad, backbone.parameters()) }, {"params": head.parameters()}],
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise


    args.total_batch_size = args.batch_size * world_size
    args.warmup_step = args.num_image // args.total_batch_size * args.warmup_epoch
    args.total_step = args.num_image // args.total_batch_size * args.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=args.warmup_step,
        total_iters=args.total_step)

    start_epoch = 0
    global_step = 0
    if args.resume:
        dict_checkpoint = torch.load(os.path.join(args.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        head.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint


    for arg in vars(args):
        num_space = 25 - len(arg)
        logging.info(": " + arg + " " * num_space + str(getattr(args, arg)))

    callback_verification = CallBackVerification(
        val_targets=args.val_targets, rec_prefix=args.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=args.frequent,
        total_step=args.total_step,
        batch_size=args.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    if args.iqa == "brisque":
        iqa = pyiqa.create_metric('brisque').cuda()
        threshold = args.threshold
    elif args.iqa == "cnniqa":
        iqa = pyiqa.create_metric('cnniqa').cuda()
        threshold = args.threshold

    logging.info("Total Parameters: %d", sum(p.numel() for p in iqa.parameters()))
    logging.info("IQA: %d", iqa.lower_better)

    for epoch in range(start_epoch, args.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            alpha = generate_alpha(img.clone(), iqa, threshold)
            local_embeddings = backbone(img, alpha)
            loss: torch.Tensor = head(local_embeddings, local_labels)

            assert loss.requires_grad

            if args.fp16:
                amp.scale(loss).backward()
                if global_step % args.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % args.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, args.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % args.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if args.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": head.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(args.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and args.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if args.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(args.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and args.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

    torch.distributed.barrier()
    destroy_process_group()                                
    return 


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = config.get_args()
    main(args)