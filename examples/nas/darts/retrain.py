import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
import utils
from model import CNN
from nni.nas.pytorch.fixed import FixedArchitecture
from nni.nas.pytorch.utils import AverageMeter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(config, train_loader, model, archit, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(x)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(config, valid_loader, model, archit, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(accuracy["acc1"], N)
            top5.update(accuracy["acc5"], N)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)

    args = parser.parse_args()
    assert torch.cuda.is_available()
    dataset_train, dataset_valid = datasets.get_dataset("cifar10", cutout_length=16)

    model = CNN(32, 3, 36, 10, args.layers, auxiliary=True)
    archit = FixedArchitecture(model, "./checkpoints/epoch_0.json")
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    criterion.cuda()
    # TODO: move architecture to cuda

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    best_top1 = 0.
    for epoch in range(args.epochs):
        drop_prob = args.drop_path_prob * epoch / args.epochs
        model.drop_path_prob(drop_prob)

        # training
        train(args, train_loader, model, archit, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(args, valid_loader, model, archit, criterion, epoch, cur_step)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))