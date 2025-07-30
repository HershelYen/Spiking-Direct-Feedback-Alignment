import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer,encoding

from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import yaml
from torch import amp
import sys
import datetime
from spikingjelly import visualizing
from model.model import MLP_SNN, ConvNet_SNN
from model.vgg import VGG11_SNN
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from spikingjelly.datasets import n_mnist, dvs128_gesture
from module.dataloader import dataset_generator
from module.dataloader_shd import shd_dataloader_from_hdf5

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Spiking Neural Network with SDFA')
    
    # ----------------------basic parameters
    # dataset
    parser.add_argument('-ds', '--dataset', type=str, choices=['nmnist', 'dvs'], default='nmnist', help='datasets used')
    # device
    parser.add_argument('-d','--device', default='cuda', help='device, default=cuda'  )
    # num_workers
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading, default=4')
    # seed
    # data_dir
    parser.add_argument('--data-dir', type=str, help='root dir of dataset')
    # output-dir
    parser.add_argument('--out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint, default=./logs')
    # resume
    parser.add_argument('--resume', type=str, help='resume from the checkpoint path')
    
    # ----------------------neuron parameters

    # time step
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps, default=20')
    # tau
    parser.add_argument('--tau', default=2.0, type=float, help='parameter tau of LIF neuron, default=2.0')
    # v_threshold
    parser.add_argument('-vt', '--v-threshold', type=float, default=1.0, help='parameter v_threshold of LIF neuron, default=1.0')
    
    # ----------------------model parameters
    # fc parameters
    parser.add_argument('--input-dim', type=int, help='input dimension of fc')
    # hidden_dim
    parser.add_argument('--hidden-dim', type=int, default=1000, help='hidden size of fc')
    # output_dim
    parser.add_argument('--classes', type=int, help='output classes of fc')

    # conv parameters
    parser.add_argument('--channels', type=int, default=128, help='number of channels in ConvNet, default=128')
    # network
    parser.add_argument('--net', type=str, choices=['fc', 'conv', 'vgg'], default='fc', help='network used, MLP or ConvNet or VGG11')
    # the number of mlp
    parser.add_argument('--num-layers', type=int, default=2, help='layer number of MLP')
    # neuron type
    parser.add_argument('--neuron-type', type=str, choices=['lif', 'if'], default='lif', help='neuron type of MLP, LIF or IF')
   
    # -----------------------training parameters
    # use cupy
    parser.add_argument('--use-cupy', type=bool, default=False, help='use cupy or not')
    # amp
    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
    # batch size
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size, default=64')
    # epochs
    parser.add_argument('-e', '--epochs', default=20, type=int, 
                        help='number of total epochs to run, default=20')
    # optmizer
    parser.add_argument('-o', '--optim', type=str, choices=['sgd', 'adam', 'adamw'], default='adam', help='use which optimizer. SGD or Adam')
    # momentum for SGD
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum for SGD')
    # weight decay for Adam
    parser.add_argument('--weight-decay', default=0, type=float, help="weight decay for Adam optimizer")
    # lr
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate, default=1e-3')
    # training_method
    parser.add_argument('-tm', '--training-method', choices=['bp', 'sdfa', 'shallow'], default='bp', type=str, help='training method of network, BP,DFA or SHALLOW')
    # loss function
    parser.add_argument('--loss-func', default='cross', choices=['mse', 'cross'], type=str, help="loss function")
    # config
    parser.add_argument('--config', type=str, default=None, help='path to config yaml file')
    
    args, remaining = parser.parse_known_args()
    # if args.config is not None
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    parser.set_defaults(**vars(args))
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # make dir for saving logs and checkpoint
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")
    # make dir which is named as checkpoints+timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.out_dir, f'checkpoints_{args.dataset}_{args.net}_{timestamp}')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Create log file
    log_file_path = os.path.join(checkpoint_dir, 'train.log')
    log_file = open(log_file_path, 'w')
    
    # define log_print function
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()
    print(args)

    # save args to yaml file
    args_file_path = os.path.join(checkpoint_dir, 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        log_print("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    else:
        log_print("Using CUDA to train")
    # Load dataset
    # TODO: Add more datasets like SHD, Brallie letter and N-Caltech101 
    if args.dataset in ['nmnist', 'dvs', 'ncaltech', 'BrailleLetter']:
        train_data_loader, test_data_loader = dataset_generator(args.dataset,
                                                      timesteps=args.T,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      data_dir=args.data_dir)
    elif args.dataset == 'shd':
        # TODO
        train_data_loader, test_data_loader = shd_dataloader_from_hdf5(
            args.data_dir,
            time_steps=args.T,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            max_time=1.4,
            num_workers=args.num_workers,
            shuffle=True
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet.")
    log_print(f"dataset {args.dataset} loaded")

    # Define neuron type
    if args.neuron_type == 'lif':
        spiking_neuron = neuron.LIFNode
    elif args.neuron_type == 'if':
        spiking_neuron = neuron.IFNode

    # neuron dict
    neuron_dict = dict(
        v_threshold=args.v_threshold,
        surrogate_function=surrogate.ATan(),
        detach_reset=False,
    )
    if spiking_neuron == neuron.LIFNode:
        neuron_dict['tau'] = args.tau

    # Create model
    log_print(f"Creating model {args.net}")
    # TODO: ConvNet and VGG11 is not implemented yet
    if args.net == 'fc':
        model = MLP_SNN(
                        data_set = args.dataset,
                        input_dim=args.input_dim,
                        hidden_dim=args.hidden_dim,
                        output_dim=args.classes,
                        training_method=args.training_method,
                        num_layers=args.num_layers,  # Assuming 2 layers for simplicity
                        use_cupy=args.use_cupy,
                        spiking_neuron= spiking_neuron,
                        **neuron_dict
                        )
    elif args.net == 'conv':
        model = ConvNet_SNN(
            data_set = args.dataset,
            channels=args.channels,
            training_method=args.training_method,
            output_classes=args.classes,
            use_cupy=args.use_cupy,
            spiking_neuron=spiking_neuron,
            **neuron_dict
        )
    elif args.net == 'vgg':
        model = VGG11_SNN(
            data_set = args.dataset,
            channels= args.channels,
            training_method= args.training_method,
            output_classes=args.classes,
            use_cupy=args.use_cupy,
            spiking_neuron=spiking_neuron,
            **neuron_dict
        )
    else:
        raise NotImplementedError(f"Network {args.net} is not implemented yet.")
    
    model.to(device)
    log_print(f"Model {args.net} created")
    log_print("---------------------------------------------------")
    log_print(model)
    log_print("---------------------------------------------------")
    # define optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optim} is not implemented yet.")
    
    # add learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # define scaler for amp
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
        print("amp training")
    
    # Begin training
    log_print(f"Begin training {args.net} on {args.dataset} dataset")
    start_epoch = 0
    max_test_acc = -1
    
    #critertion = nn.CrossEntropyLoss()
    if args.loss_func == 'mse':
        loss_func = nn.MSELoss()
    elif args.loss_func == 'cross':
        loss_func = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unimplemented optimizer")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss =0
        train_acc = 0
        train_samples = 0
        model.train()
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame, label = frame.to(args.device), label.to(args.device)
            if args.dataset == 'shd':
                frame = frame.to_dense()
            frame = frame.transpose(0, 1)
            label_onehot = F.one_hot(label, num_classes=args.classes).float()
            if scaler is not None:
                with amp.autocast(args.device):
                    out_fr = model(frame)
                    loss = loss_func(out_fr, label_onehot if args.loss=='mse' else label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = model(frame)
                loss = loss_func(out_fr, label_onehot if args.loss_func=='mse' else label)
                loss.backward()
                optimizer.step()
            functional.reset_net(model)
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(dim=1) == label).sum().item()
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        lr_scheduler.step()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame, label = frame.to(args.device), label.to(args.device)
                if args.dataset == 'shd':
                    frame = frame.to_dense()
                frame = frame.transpose(0, 1)
                label_onehot = F.one_hot(label, num_classes=args.classes).float()
                out_fr = model(frame)
                loss = F.mse_loss(out_fr, label_onehot)
                #loss = critertion(out_fr, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(dim=1) == label).sum().item()
                functional.reset_net(model)
        test_speed = test_samples / (time.time() - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'last_model.pth'))
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'best_model.pth'))
            log_print(f"best model saved at epoch {epoch} with test accuracy {max_test_acc:.4f}")

        log_print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        log_print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        log_print(f'estimated completion = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')



if __name__ == '__main__':
    main()
