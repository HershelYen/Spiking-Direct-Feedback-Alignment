

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from module.sdfa import DFA, DFALayer
import torch.nn.functional as F

class MLP_SNN(nn.Module):
    def __init__(self, 
                 data_set,
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 training_method='bp',
                 num_layers=2,
                 use_cupy=False,
                 spiking_neuron: callable = None, *args, **kwargs):
        super(MLP_SNN, self).__init__()
        
        # neuron parameters
        self.training_method = training_method
        self.data_set = data_set
        layers = []
        dfa_layers = []
        
        self.flatten = layer.Flatten()

        # Input layer
        layers.append(layer.Linear(input_dim, hidden_dim, bias=False))
        layers.append(spiking_neuron(*args, **kwargs))

        # If using DFA, add DFA layer after input layer
        if self.training_method == 'sdfa':
            dfa_in = DFALayer(batch_dims=(1,), time_dims=(0,),)
            layers.append(dfa_in)
            dfa_layers.append(dfa_in)
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(layer.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(spiking_neuron(*args, **kwargs))
            # If using DFA, add DFA layer after each hidden layer
            if self.training_method == 'sdfa':
                dfa_hidden = DFALayer(batch_dims=(1,), time_dims=(0,),)
                layers.append(dfa_hidden)
                dfa_layers.append(dfa_hidden)
        # Output layer
        layers.append(layer.Linear(hidden_dim, output_dim, bias=False))
        layers.append(spiking_neuron(*args, **kwargs))
        
        # if using DFA, add DFA after output layer
        if self.training_method in ['sdfa', 'shallow']:
            self.dfa = DFA(dfa_layers,
                        no_training= (self.training_method == 'shallow'))

        # model initialization
        self.model = nn.Sequential(*layers)

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x):
        x_seq = self.flatten(x)
        fr = self.model(x_seq).mean(0)
        if self.training_method == 'sdfa':
            fr = self.dfa(fr)
        return fr

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, training_method, spiking_neuron: callable, *args, **kwargs,):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2)
        )
        self.training_method = training_method

        if self.training_method in ['sdfa', 'shallow']:
            self.dfa1 = DFALayer(batch_dims=(1, ), time_dims=(0, ))
    def forward(self, x):
        if self.training_method in ['sdfa', 'shallow']:
            return self.dfa1(self.block(x))
        else:
            return self.block(x)

class ConvNet_SNN(nn.Module):
    def __init__(self, 
                 data_set,
                 channels=128, 
                 training_method='bp',
                 output_classes=11,
                 use_cupy=False,
                 spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()
        # Create a sequence of ConvBlocks
        self.training_method = training_method
        self.conv_layers = nn.Sequential(
            ConvBlock(2, channels//4, training_method, spiking_neuron, *args, **kwargs),
            ConvBlock(channels//4, channels//2, training_method, spiking_neuron, *args, **kwargs),
            ConvBlock(channels//2, channels, training_method, spiking_neuron, *args, **kwargs),
            ConvBlock(channels, channels, training_method, spiking_neuron, *args, **kwargs),
            ConvBlock(channels, channels, training_method, spiking_neuron, *args, **kwargs)
        )

        # Fully connected layers and final voting layer
        if data_set == 'dvs':
            self.fc1 = nn.Sequential(
                layer.Flatten(),
                layer.Linear(channels * 4 * 4, 1024),
                spiking_neuron(*args, **kwargs),
            )
        elif data_set == 'ncaltech':
            self.fc1 = nn.Sequential(
                layer.Flatten(),
                layer.Linear(channels * 5 * 7, 1024),
                spiking_neuron(*args, **kwargs),
            )

        self.fc2 = nn.Sequential(
            layer.Linear(1024, output_classes*10),
            spiking_neuron(*args, **kwargs)
        )
        self.vote = layer.VotingLayer(10)
        if self.training_method in ['sdfa', 'shallow']:
            self.dfa1= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            dfa_list = [self.dfa1] + [self.conv_layers[i].dfa1 for i in range(len(self.conv_layers))]
            self.dfa = DFA(dfa_list, no_training=(self.training_method=='shallow'))
            print(dfa_list)
        functional.set_step_mode(self, step_mode='m')
    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        if self.training_method in ['sdfa', 'shallow']:
            x = self.dfa1(self.fc1(x))
            return self.dfa(self.vote(self.fc2(x)).mean(0))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            return self.vote(x).mean(0)


def test_model():
    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    from module.sdfa import DFA, DFALayer
    import torch.nn.functional as F
    device = 'cpu'
    spiking_neuron = neuron.IFNode
    neuron_dict = dict(
        v_threshold=1.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=False,
    )
    net1 = MLP_SNN(input_dim=32768,
                  hidden_dim=128,
                  output_dim=11,
                  training_method='sdfa',
                  num_layers=2,
                  use_cupy=False,
                  spiking_neuron=spiking_neuron,
                  **neuron_dict)
    net1.to(device)
    print(f"DVS net test")
    print(f"---------------------------------------------")
    print(net1)
    # T N C H W
    a1, a2= torch.rand(2, 64, 2, 128, 128), torch.rand(2, 64, 2, 128, 128)
    b1, b2= torch.rand(64, 11), torch.rand(64, 11)
        
    out1 = net1(a1)
    print(out1.shape)
    loss = F.mse_loss(out1, b1)
    loss.backward()
    print(f"net1 test done")
