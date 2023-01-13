import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, dataset_collate
from utils.utils import get_num_classes, show_config
from utils.train import fit_one_epoch

##################################################################################################
# Step 1: 设置各种硬件参数和网络参数
#   Step 1.1: 设置各种硬件参数
#   Step 1.2: 设置网络参数
# Step 2: 载入参数，并且处理成torch网络所需要的参数
#   Step 2.1: 载入模型model
#   Step 2.2: 设置记录函数logging
#   Step 2.3: 设置scaler和训练规则model_train
#   Step 2.4: 划分数据集生成lines，并且构造数据加载器train_dataset/val_dataset
#   Step 2.5: 处理其它所需要的参数：调整后的学习率Max_lr_fit/Min_lr_fit/优化器optimizer/学习率下降公式lr_scheduler_func/epoch_step
# Step 3: 生成torch模型所需要的数据加载器DataLoader
# Step 4: 正式开始训练
##################################################################################################
if __name__ == "__main__":
    #############################################################################################################
    # Step 1: 设置各种硬件参数和网络参数，诸如distributed等参数设置从其它代码中借鉴过来，
    # 虽然本代码中并不使用，但是考虑到可扩展性以及可读性，就保留在代码当中。
    #############################################################################################################
    #############################################################################################################
    # Step 1.1: 设置各种硬件参数
    #############################################################################################################
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers     = 4
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
    
    
    #############################################################################################################
    # Step 1.2: 设置网络参数
    #############################################################################################################
    #--------------------------------------------------------#
    #   指向记录着所有训练图像地址和标签的txt文件
    #--------------------------------------------------------#
    annotation_path = "data_path/training_set.txt"
    #--------------------------------------------------------#
    #   输入图像大小，我们此处参考其它模型选择[160,160,3]
    #--------------------------------------------------------#
    input_shape     = [160, 160, 3]
    #------------------------------------------------------------------------------------------#
    #   主干特征提取网络的选择，对于图像识别常用的有mobilenet和resnet，比如Facenet论文选择了resnet
    #   此处我们尝试选择mobilenet
    #------------------------------------------------------------------------------------------#
    backbone        = "mobilenet"
    #------------------------------------------------------------------#
    #   是否载入先前的网络参数（预训练模型）
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #
    #   Max_lr         模型的最大学习率
    #   Min_lr         模型的最小学习率，默认为最大学习率的0.01
    #   考虑到没有预训练模型时容易陷入到局部极值、故而Max_lr会偏大
    #------------------------------------------------------------------#
    pretrained      = False
    input_path      = "model_data/facenet_mobilenet.pth"
    if pretrained:
        model_path      = input_path
        Max_lr  = 1e-3
        Min_lr   = Max_lr * 0.01
    else:
        model_path      = ""
        Max_lr  = 5e-3
        Min_lr   = Max_lr * 0.01
    #------------------------------------------------------#
    #   Init_Epoch      模型当前开始的训练epoch
    #   batch_size      每次输入的图片数量
    #                   受到数据加载方式与triplet loss的影响
    #                   batch_size需要为3的倍数
    #   Epoch           模型总共训练的epoch
    #------------------------------------------------------#
    batch_size      = 96
    if batch_size % 3 != 0:
        raise ValueError("Batch_size must be the multiple of 3.")
    Init_Epoch      = 0
    Epoch           = 100
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type  = "adam"
    momentum        = 0.9
    weight_decay    = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个epoch都保存
    #------------------------------------------------------------------#
    save_period = 5
    #------------------------------------------------------------------#
    #   early_stop     当验证集accuracy连续多少个epoch下降，就进行早停
    #   考虑到如果不加载预训练模型，最开始的20-30个epoch acc可能波动比较大，故而我们在acc>0.1后再开启早停
    #------------------------------------------------------------------#
    early_stop = 5
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.utcnow(),'%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join('logs', str(time_str) + "_pretrained_" + str(pretrained) + "_Maxlr_" + 
    str(Max_lr) + "_optimizer_" + optimizer_type + "_decaytype_" + lr_decay_type)








    #############################################################################################################
    # Step 2: 载入参数，并且处理成torch网络所需要的参数
    #############################################################################################################
    #############################################################################################################
    # Step 2.1: 载入模型model
    #############################################################################################################
    num_classes = get_num_classes(annotation_path)
    model = Facenet(backbone=backbone, num_classes=num_classes)
    #-------------------------------------------#
    #   如果设置了pretrained, 加载预训练权重    
    #-------------------------------------------#
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #############################################################################################################
    # Step 2.2: 设置记录函数logging
    #############################################################################################################
    loss = triplet_loss()
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    
    #############################################################################################################
    # Step 2.3: 设置scaler和训练规则model_train
    #############################################################################################################    
    #------------------------------------------------------------------#
    #   设置scaler
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #############################################################################################################
    # Step 2.4: 划分数据集生成lines，并且构造数据加载器train_dataset/val_dataset
    #############################################################################################################  
    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #---------------------------------------#
    #   构建数据集加载器。
    #---------------------------------------#
    train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
    val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)

    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Max_lr = Max_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    
    #############################################################################################################
    # Step 2.5: 处理其它所需要的参数：调整后的学习率Max_lr_fit/Min_lr_fit/优化器optimizer/学习率下降公式lr_scheduler_func/epoch_step
    #############################################################################################################  
    #-------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    #-------------------------------------------------------------------#
    nbs             = 64
    lr_limit_max    = 5e-3 if optimizer_type == 'adam' else 1e-1
    lr_limit_min    = 5e-4 if optimizer_type == 'adam' else 5e-4
    Max_lr_fit     = min(max(batch_size / nbs * Max_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #---------------------------------------#
    #   根据optimizer_type选择优化器
    #---------------------------------------#
    optimizer = {
        'adam'  : optim.Adam(model.parameters(), Max_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        'sgd'   : optim.SGD(model.parameters(), Max_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
    }[optimizer_type]

    #---------------------------------------#
    #   获得学习率下降的公式
    #---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Max_lr_fit, Min_lr_fit, Epoch)
    
    #---------------------------------------#
    #   判断每一个epoch的长度
    #---------------------------------------#
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size   
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        


    #############################################################################################################
    # Step 3: 生成torch模型所需要的数据加载器DataLoader
    #############################################################################################################     
    #--------------------------------------------------------------------------#
    # 这里batch_size之所以是1/3，是因为train_dataset是属于FacenetDataset类的，
    # FacenetDataset类的__getitem__规定每取出一个元素就是triplet
    #--------------------------------------------------------------------------#
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                        drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)
    





    #############################################################################################################
    # Step 4: 正式开始训练
    ############################################################################################################# 
    #--------------------------------------------------------------------------#
    # last_val_acc    上一次的validation accuracy，用来作早停使用
    # decrease_epoch  已经连续下降多少个epoch
    #--------------------------------------------------------------------------#
    last_val_acc = 0
    decrease_epoch = 0
    for epoch in range(Init_Epoch, Epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
            
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
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
        # 17 save_period     多少个epoch保存一次权值，默认每个epoch都保存
        # 19 local_rank      distributed中的排序
        # 20 last_val_acc    上一次的validation accuracy，用来作早停使用
        # 21 decrease_epoch  已经连续下降多少个epoch
        #-----------------------------------------------------------------#
        last_val_acc, decrease_epoch = fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda
        , batch_size//3, fp16, scaler, save_period, save_dir, local_rank, last_val_acc, decrease_epoch)
        
        if last_val_acc > 0.1 and decrease_epoch == early_stop:
            break

    if local_rank == 0:
        loss_history.writer.close()