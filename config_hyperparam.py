from easydict import EasyDict
import torch

cfg = EasyDict()

cfg.check_path = "checkpoint-test"
cfg.num_class = 2


cfg.batch_size = 32
cfg.lr = 0.003
cfg.warmup = 3
cfg.resume = 0
cfg.epochs = 500
cfg.freeze = False
cfg.pth = ""
cfg.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
cfg.gpu_idx = [0, 1]

cfg.cover_path = '/data/public/steganography_dataset/Bossbase1.01-size256/'
# cfg.stego_path = '/data/user/yangyaotian/dataset/steganalysis/Bossbase-0.4bpp-S-UNIWARD-size256/'
# cfg.stego_path = '/data/user/yangyaotian/dataset/steganalysis/Bossbase-0.4bpp-WOW-size256/'
cfg.stego_path = '/data/user/yangyaotian/dataset/steganalysis/Bossbase-0.4bpp-HILL-size256/'
cfg.best_model_path = cfg.check_path + "/best_model_checkpoint.pth.tar"
cfg.result_loss = cfg.check_path + "/result_loss.png"
cfg.result_acc = cfg.check_path + "/result_acc.png"
cfg.result_lr = cfg.check_path + "/result_lr.png"
