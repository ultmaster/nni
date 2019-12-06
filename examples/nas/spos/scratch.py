import argparse
import logging

import torch
import torch.nn as nn
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import get_imagenet, CrossEntropyLabelSmooth, accuracy

logger = logging.getLogger("nni")


def train(epoch, model, criterion, optimizer, loader, writer, args):
    model.train()
    meters = AverageMeterGroup()
    cur_lr = optimizer.param_groups[0]["lr"]

    for step, (x, y) in enumerate(loader):
        cur_step = len(loader) * epoch + step
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        metrics = accuracy(logits, y)
        metrics["loss"] = loss.item()
        meters.update(metrics)

        writer.add_scalar("lr", cur_lr, global_step=cur_step)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", metrics["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", metrics["acc5"], global_step=cur_step)

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                        args.epochs, step + 1, len(loader), meters)


def validate(epoch, model, criterion, loader, writer, args):
    model.eval()
    meters = AverageMeterGroup()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            metrics = accuracy(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)

            if step % args.log_frequency == 0 or step + 1 == len(loader):
                logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                            args.epochs, step + 1, len(loader), meters)

    writer.add_scalar("loss/test", meters.loss.avg, global_step=epoch)
    writer.add_scalar("acc1/test", meters.acc1.avg, global_step=epoch)
    writer.add_scalar("acc5/test", meters.acc5.avg, global_step=epoch)

    logger.info("Epoch %d validation: %s", epoch + 1, meters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--tb-dir", type=str, default="runs")
    parser.add_argument("--architecture", type=str, default="architecture_final.json")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)

    args = parser.parse_args()
    dataset_train, dataset_valid = get_imagenet(args.imagenet_dir, False)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = ShuffleNetV2OneShot()
    model.cuda()
    apply_fixed_architecture(model, args.architecture)
    model = nn.DataParallel(model)
    criterion = CrossEntropyLabelSmooth(1000, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.epochs else 0,
                                                  last_epoch=-1)
    writer = SummaryWriter(log_dir=args.tb_dir)

    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, writer, args)
        validate(epoch, model, criterion, valid_loader, writer, args)
        scheduler.step()
    writer.close()
