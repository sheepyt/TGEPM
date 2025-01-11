import os
import torch
from torch import nn
from tqdm import tqdm
import sys
from config_hyperparam import cfg
import torch.nn.functional as F



def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device) 
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images = torch.cat((data["cover"], data["stego"]), 0)
        labels = torch.cat((data["label"][0], data["label"][1]), 0)
        images, labels, = images.to(device), labels.to(device)
        sample_num += images.shape[0]
        pres = model(images)
        pres_ls = F.log_softmax(pres, dim=1)
        loss = F.nll_loss(pres_ls, labels)
        loss.backward()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        pred_classes = torch.max(nn.Softmax(dim=1)(pres), dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss.detach()

        cur_loss = accu_loss.item() / (step + 1)
        cur_acc = accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.4f}, lr: {:.6f}".format(
            epoch,
            cur_loss,
            cur_acc,
            optimizer.param_groups[0]["lr"],
            width=os.get_terminal_size().columns)

        optimizer.step()
        optimizer.zero_grad()

    return cur_loss, cur_acc


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images = torch.cat((data["cover"], data["stego"]), 0)
        labels = torch.cat((data["label"][0], data["label"][1]), 0)
        images, labels, = images.to(device), labels.to(device)
        sample_num += images.shape[0]
        pres = model(images)
        pres_ls = F.log_softmax(pres, dim=1)
        loss = F.nll_loss(pres_ls, labels)
        pred_classes = torch.max(nn.Softmax(dim=1)(pres), dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss.detach()
        cur_loss = accu_loss.item() / (step + 1)
        cur_acc = accu_num.item() / sample_num
        data_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.4f}".format(epoch, cur_loss, cur_acc, width=os.get_terminal_size().columns)

    return cur_loss, cur_acc


@torch.no_grad()
def test(model, data_loader, device):

    assert os.path.isfile(cfg.best_model_path)
    checkpoint = torch.load(cfg.best_model_path)
    #print(model.load_state_dict(checkpoint["model"]))
    model.eval()
    accu_num = torch.zeros(1).to(device) 
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images = torch.cat((data["cover"], data["stego"]), 0)
        labels = torch.cat((data["label"][0], data["label"][1]), 0)
        images, labels, = images.to(device), labels.to(device)
        sample_num += images.shape[0]
        feats, pres = model(images)
        pres_ls = F.log_softmax(pres, dim=1)
        loss = F.nll_loss(pres_ls, labels)
        pred_classes = torch.max(nn.Softmax(dim=1)(pres), dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        accu_loss += loss
        cur_loss = accu_loss.item() / (step + 1)
        cur_acc = accu_num.item() / sample_num
        data_loader.desc = "[test] loss: {:.3f}, acc: {:.4f}".format(cur_loss, cur_acc)