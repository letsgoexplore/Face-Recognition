import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr

#-----------------------------------------------------------------#
# 1 model_train = torch.nn.DataParallel(model)
#   model_train = model_train.cuda()
# 2 model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
# 4 loss            = triplet_loss()
# 7 epoch_step      = num_train // batch_size
# 8 epoch_step_val  = num_val // batch_size
# 9 gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
#                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
# 10 gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
#                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)
# 11 Epoch 模型总共训练的epoch
# 12 cuda  是否使用Cuda显卡
# 16 scaler:
#     if fp16:
#     from torch.cuda.amp import GradScaler as GradScaler
#     scaler = GradScaler()
# 17 save_period     多少个epoch保存一次权值，默认每个世代都保存
# 19 local_rank      distributed中的排序
# 20 last_val_acc    上一次的validation accuracy，用来作早停使用
# 21 decrease_epoch  已经连续下降多少个epoch
#-----------------------------------------------------------------#
def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, Batch_size, fp16, scaler, save_period, save_dir, local_rank, last_val_acc, decrease_epoch):
    total_triple_loss   = 0
    total_CE_loss       = 0
    total_accuracy      = 0

    val_total_triple_loss   = 0
    val_total_CE_loss       = 0
    val_total_accuracy      = 0

    #############################################################################################################
    # Step 1: 训练集gen开始训练
    #############################################################################################################
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        # 通过iteration的判断，来控制迭代轮次；而非gen的内部控制函数
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs1, outputs2 = model_train(images, "train")
            # 分别计算triplet loss和center loss，避免过拟合
            _triplet_loss   = loss(outputs1, Batch_size)
            _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _loss           = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()      
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2 = model_train(images, "train")

                _triplet_loss   = loss(outputs1, Batch_size)
                _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
                _loss           = 10*_triplet_loss + _CE_loss
            
            # 反向传播
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()  

        with torch.no_grad():
            accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            
        total_triple_loss   += _triplet_loss.item()
        total_CE_loss       += _CE_loss.item()
        total_accuracy      += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_triple_loss' : total_triple_loss / (iteration + 1), 
                                'total_CE_loss'     : total_CE_loss / (iteration + 1), 
                                'accuracy'          : total_accuracy / (iteration + 1), 
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)

    #############################################################################################################
    # Step 2: 验证集gen_val开始训练
    #############################################################################################################        
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs1, outputs2 = model_train(images, "train")
            
            _triplet_loss   = loss(outputs1, Batch_size)
            _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _loss           = _triplet_loss + _CE_loss
            
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            
            val_total_triple_loss   += _triplet_loss.item()
            val_total_CE_loss       += _CE_loss.item()
            val_total_accuracy      += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_total_triple_loss' : val_total_triple_loss / (iteration + 1), 
                                'val_total_CE_loss'     : val_total_CE_loss / (iteration + 1), 
                                'val_accuracy'          : val_total_accuracy / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)

        if local_rank == 0:
            pbar.close()
            print('Finish Validation')

    #############################################################################################################
    # Step 3: 输出结果
    #############################################################################################################     
        loss_history.append_loss(epoch, total_accuracy / epoch_step, \
            (10*total_triple_loss + total_CE_loss) / epoch_step, val_total_accuracy / epoch_step_val, (10*val_total_triple_loss + val_total_CE_loss) / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % ((10*total_triple_loss + total_CE_loss) / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1),
                                                                    total_accuracy / epoch_step,
                                                                    val_total_accuracy / epoch_step_val)))
        if val_total_accuracy / epoch_step_val >= last_val_acc:
            i = 0
        else:
            i = decrease_epoch + 1
        return val_total_accuracy / epoch_step_val, i
