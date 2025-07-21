import torch
import torch.nn as nn
import ssl
import h5py
import os
import numpy as np
from torch.utils.data import Dataset, dataloader

from module.utils import get_shd_dataset

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from module.sparse_batch import sparse_batch_collate
class SparseSHDDataset(Dataset):
    def __init__(self, X, y, nb_steps, nb_units, max_time):
        self.X = X
        self.y = np.array(y, dtype=np.int64)
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.max_time = max_time
        
        # 直接全部读取到内存，如果内存不足改为hdf5 lazy模式
        self.firing_times = self.X['times']
        self.units_fired = self.X['units']
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        times = np.digitize(self.firing_times[idx], self.time_bins)
        units = self.units_fired[idx]
        coo = np.stack([times, units])  # shape: [2, nnz]
        i = torch.from_numpy(coo).long()
        v = torch.ones(len(coo[0]), dtype=torch.float32)
        X_tensor = torch.sparse_coo_tensor(i, v, (self.nb_steps, self.nb_units))
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return X_tensor, y_tensor


def shd_dataloader_from_hdf5(data_dir, time_steps, batch_size, input_dim, max_time=1.4, num_workers=0, shuffle=True):
    import os, h5py
    cache_dir = os.path.expanduser(data_dir)
    cache_subdir = "hdspikes"
    train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
    test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    print('load finished')
    print('length of y_train:', len(y_train))
    print('length of y_test:', len(y_test))

    train_dataset = SparseSHDDataset(x_train, y_train, time_steps, input_dim, max_time)
    test_dataset = SparseSHDDataset(x_test,  y_test,  time_steps, input_dim, max_time)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              num_workers=num_workers,
                              collate_fn=sparse_batch_collate)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers,
                             collate_fn=sparse_batch_collate)
    return train_loader, test_loader



def shd_dataset_generator(data_dir, 
              time_steps,
              batch_size,
              input_dim,
              max_time=1.4,
              device='cpu'):
    ssl._create_default_https_context = ssl._create_unverified_context
    # data procesing and load data
    cache_dir = os.path.expanduser(data_dir)
    cache_subdir = "hdspikes"
    #get_shd_dataset(cache_dir, cache_subdir)

    train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
    test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    print('load finished')
    print('length of y_train:', len(y_train))
    print('length of y_test:', len(y_test))
    #return x_train, y_train, x_test, y_test
    train_loader = sparse_data_generator_from_hdf5_spikes(x_train, y_train, batch_size, time_steps, input_dim, max_time, shuffle=True, device=device)
    test_loader = sparse_data_generator_from_hdf5_spikes(x_test, y_test, batch_size, time_steps, input_dim, max_time, shuffle=False, device=device)
    return train_loader, test_loader



def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True, device=None):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y,dtype=np.int64)
    number_of_batches = len(labels_)//batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']
    
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)
        
        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

# add regularization
def compute_regularization(net, l1_factor, l2_factor):
    l1_loss = 0
    l2_loss = 0
    for param in net.parameters():
        l1_loss += torch.sum(torch.abs(param))
        l2_loss += torch.sum(param ** 2)
    return l1_factor * l1_loss + l2_factor * l2_loss