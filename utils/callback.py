import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.acc        = []
        self.losses     = []
        self.val_acc    = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        self.writer.add_graph(model, dummy_input)

    #-----------------------------------------------------------------------#
    # 每一个epoch，都将epoch,accuracy,loss,validation loss等信息记录到txt当中
    #-----------------------------------------------------------------------#
    def append_loss(self, epoch, acc, loss, val_acc, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f1:
            f1.write("Epoch" + str(epoch) + ":" + str(acc))
            f1.write("\n")
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f2:
            f2.write("Epoch" + str(epoch) + ":" + str(loss))
            f2.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f1:
            f1.write("Epoch" + str(epoch) + ":" + str(val_acc))
            f1.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f3:
            f3.write("Epoch" + str(epoch) + ":" + str(val_loss))
            f3.write("\n")

        self.writer.add_scalar('acc', acc, epoch)
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_acc', val_acc, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    #-----------------------------------------------------------------------#
    # 每一个epoch，都将从起始到当前epoch所有的Loss和accuracy的折线图
    #-----------------------------------------------------------------------#
    def loss_plot(self):
        # 绘制Loss的折线图
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
        
        # 绘制accuracy的折线图
        plt.figure()
        plt.plot(iters, self.acc, 'red', linewidth = 2, label='acc')
        plt.plot(iters, self.val_acc, 'coral', linewidth = 2, label='val acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.acc, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val acc')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
        plt.cla()
        plt.close("all")
