import torch
import torch.nn as nn


def dataset_generator(dataset, timesteps, batch_size, num_workers=8, data_dir=None):
    if dataset == 'nmnist':
        from spikingjelly.datasets import n_mnist
        train_dataset = n_mnist.NMNIST(data_dir + 'nmnist', 
                                       train=True,
                                       data_type='frame',
                                       frames_number=timesteps,
                                       split_by='number')
        test_dataset = n_mnist.NMNIST(data_dir + 'nmnist', 
                                      train=False,
                                      data_type='frame',
                                      frames_number=timesteps,
                                      split_by='number')
    elif dataset == 'ncaltech':
        from spikingjelly.datasets import n_caltech101
        dataset = n_caltech101.NCaltech101(data_dir + 'caltech101', 
                                                  data_type='frame',
                                                  frames_number=timesteps,
                                                  split_by='number')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    elif dataset == 'dvs':
        from spikingjelly.datasets import dvs128_gesture
        train_dataset = dvs128_gesture.DVS128Gesture(data_dir + 'dvs', 
                                   train=True, 
                                   data_type='frame',
                                   frames_number=timesteps,
                                   split_by='number')
        test_dataset = dvs128_gesture.DVS128Gesture(data_dir + 'dvs', 
                                  train=False, 
                                  data_type='frame',
                                  frames_number=timesteps,
                                  split_by='number')
    elif dataset == 'BrailleLetter':
        train_dataset, test_dataset = load_braille_letter_data(data_dir)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented yet.")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def load_braille_letter_data(data_dir):
    import os, pickle, gzip
    import numpy as np
    file_name = data_dir + 'braille/tutorial5_braille_spiking_data.pkl.gz'
    with gzip.open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)
    letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # Extract data
    nb_repetitions = 50
    data = []
    labels = []
    for i, letter in enumerate(letter_written):
        for repetition in np.arange(nb_repetitions):
            idx = i * nb_repetitions + repetition
            dat = 1.0-data_dict[idx]['taxel_data'][:]/255
            data.append(dat)
            labels.append(i)
    # Crop to the same length
    data_step = l =np.min([len(d) for d in data])

    data = torch.tensor([d[:l] for d in data], dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    # Select nonzero inputs
    nzid = [1, 2, 6, 10]
    data = data[:, :, nzid]

    # Standardize data
    rshp = data.reshape(-1, data.shape[2])
    data = (data - rshp.mean(dim=0)) / (rshp.std(dim=0) + 1e-6)

    # Upsample data
    nb_upsample = 2
    data = upsample(data, n=nb_upsample)

    # Shuffle data
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # Peform train/test split
    a = int(0.8 * len(data))
    x_train, x_test = data[:a], data[a:]
    y_train, y_test = labels[:a], labels[a:]

    ds_train = torch.utils.data.TensorDataset(x_train, y_train)
    ds_test = torch.utils.data.TensorDataset(x_test, y_test)
    
    # check unique labels
    print(np.unique(labels))
    return ds_train, ds_test

def upsample(data, n=2):
    shp = data.shape
    tmp = data.reshape(shp+(1,))
    tmp = data.tile(1, 1, 1, n)
    return tmp.reshape(shp[0], n*shp[1], shp[2])