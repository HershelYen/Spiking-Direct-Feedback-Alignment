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
    # ----------------------basic parameters
    # dataset
    parser.add_argument('-ds', '--dataset', type=str, choices=['nmnist', 'dvs'], default='nmnist', help='datasets used')
    # device
    parser.add_argument('-d','--device', default='cuda', help='device, default=cuda')
    # num_workers
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading, default=4')
    # data_dir
    parser.add_argument('--data-dir', type=str, help='root dir of dataset')
    # model path
    parser.add_argument('--model-path', type=str, default='./best_model/best_model.pth', help='path to the saved model')
    
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
   
    # -----------------------validation parameters
    # use cupy
    parser.add_argument('--use-cupy', type=bool, default=False, help='use cupy or not')
    # amp
    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
    # batch size
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size, default=64')
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

def load_model(args):
    """load the trained model from checkpoint"""
    device = args.device
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
    print(f"Creating model {args.net}")
    if args.net == 'fc':
        model = MLP_SNN(
                        data_set = args.dataset,
                        input_dim=args.input_dim,
                        hidden_dim=args.hidden_dim,
                        output_dim=args.classes,
                        training_method=args.training_method,
                        num_layers=args.num_layers,
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
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['net'])
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'max_test_acc' in checkpoint:
        print(f"Best training accuracy: {checkpoint['max_test_acc']:.4f}")
    
    return model

def validate(model, test_data_loader, args):
    """Validate the model on test dataset"""
    device = args.device
    if args.loss_func == 'mse':
        loss_func = nn.MSELoss()
    elif args.loss_func == 'cross':
        loss_func = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unimplemented loss function")
    
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0

    print("Starting validation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (frame, label) in enumerate(test_data_loader):
            frame, label = frame.to(device), label.to(device)
            
            if args.dataset == 'shd':
                frame = frame.to_dense()
            frame = frame.transpose(0, 1)
            
            if args.loss_func == 'mse':
                label_onehot = F.one_hot(label, num_classes=args.classes).float()
            
            # Forward pass
            if args.amp:
                with amp.autocast(device):
                    out_fr = model(frame)
                    loss = loss_func(out_fr, label_onehot if args.loss_func=='mse' else label)
            else:
                out_fr = model(frame)
                loss = loss_func(out_fr, label_onehot if args.loss_func=='mse' else label)
            
            # Calculate accuracy
            pred = out_fr.argmax(dim=1)
            correct = (pred == label).sum().item()
            
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += correct
        
            
            # Reset network state
            functional.reset_net(model)
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_data_loader)} batches")
    
    # Calculate final metrics
    validation_time = time.time() - start_time
    test_speed = test_samples / validation_time
    test_loss /= test_samples
    test_acc /= test_samples
    
    # Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Total Samples: {test_samples}")
    print(f"Validation Speed: {test_speed:.2f} samples/s")
    print(f"Validation Time: {validation_time:.2f}s")
    
    return test_acc, test_loss

def main():
    args = parse_args()
    print("Validation Arguments")
    print(args)
    print()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    else:
        print(f"Using device: {device}")

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset in ['nmnist', 'dvs', 'ncaltech', 'BrailleLetter']:
        train_data_loader, test_data_loader = dataset_generator(args.dataset,
                                                      timesteps=args.T,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      data_dir=args.data_dir)
    elif args.dataset == 'shd':
        train_data_loader, test_data_loader = shd_dataloader_from_hdf5(
            args.data_dir,
            time_steps=args.T,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            max_time=1.4,
            num_workers=args.num_workers,
            shuffle=False  # No need to shuffle for validation
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet.")
    
    print(f"Dataset {args.dataset} loaded successfully")
    print(f"Test dataset size: {len(test_data_loader.dataset)}")
    
    # Load model
    model = load_model(args)
    print("Model architecture:")
    print("-" * 50)
    print(model)
    print("-" * 50)
    
    # Validate model
    test_acc, test_loss = validate(model, test_data_loader, args)
    
    print(f"\nFinal Validation Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

if __name__ == '__main__':
    main()