import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torchvision
from config_hyperparam import cfg
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode
from utils import train_one_epoch, evaluate, test
import shutil
from scheduler import WarmupCosineSchedule
from dataset import MyDataset
from network import Model
from autoaugment import RandomRotationCustom


def main():
    if not os.path.exists(cfg.check_path):
        os.makedirs(cfg.check_path)

    shutil.copy2("config_hyperparam.py", cfg.check_path)

    train_transform = torchvision.transforms.Compose([
        RandomRotationCustom([0, 90, 180, 270]),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ]),
        transforms.ToTensor()
    ])
    data_transform = torchvision.transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_dataset = MyDataset(cfg.cover_path, cfg.stego_path, "train", transform=train_transform)
    test_dataset = MyDataset(cfg.cover_path, cfg.stego_path, "test", transform=data_transform)

    nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, 8])  # number of workers
    #print("Using {} num_workers for dataloader".format(nw))


    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=nw)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True,
                             num_workers=nw)

    model = Model()
    #print(model)

    num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    #print("number of parameters: {:.2f} M".format(num_parameters / 1000000))

    if cfg.pth != "":
        assert os.path.exists(cfg.pth), "weights file: '{}' not exist.".format(cfg.pth)
        weights_dict = torch.load(cfg.pth)['model']
        weights_dict = {k: v for k, v in weights_dict.items() if 'head' not in k}
        print(model.load_state_dict(weights_dict, strict=False))

    model = model.to(cfg.device)
    # model = nn.DataParallel(model, device_ids=cfg.gpu_idx)

    # 冻结除全连接外的所有层
    if cfg.freeze:
        for name, value in model.named_parameters():
            if "fc" in name:
                value.requires_grad = True
            else:
                value.requires_grad = False

    pg = [p for p in model.parameters() if p.requires_grad] 
    optimizer = optim.SGD(pg, lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup, t_total=cfg.epochs)


    start_epoch = 0
    if cfg.resume: 
        assert os.path.isfile(cfg.best_model_path)
        checkpoint = torch.load(cfg.best_model_path)
        best_acc = checkpoint["best_acc"] 
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"]) 
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load checkpoints at epoch {}.".format(start_epoch))
        print("Best accuracy so far: {}.".format(best_acc))

    best_acc = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    lr_list = []

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                  device=cfg.device, epoch=epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model, data_loader=test_loader, device=cfg.device, epoch=epoch)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        lr_list.append(optimizer.param_groups[0]["lr"])
        draw_result(lr_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        checkpoint = {
            "best_acc": best_acc,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if is_best:
            torch.save(checkpoint, cfg.best_model_path)

    test(model=model, data_loader=test_loader, device=cfg.device)


def draw_result(lr_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list):
    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(cfg.result_loss)
    plt.close()
    plt.figure()
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(cfg.result_acc)
    plt.close()
    plt.figure()
    plt.plot(lr_list, label='LR')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig(cfg.result_lr)
    plt.close()


if __name__ == '__main__':
    main()
