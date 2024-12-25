import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from math import ceil
import h5py


class HDFDataset(Dataset):
    def __init__(self, list_of_hdf_files, label, mode, iterations_per_epoch, batch_size):
        self.files = list_of_hdf_files
        self.label = label
        self.mode = mode
        self.iterations_per_epoch = iterations_per_epoch
        self.batch_size = batch_size
        self.num_files_to_open = ceil(batch_size / 128)

    def __len__(self):
        if self.mode == "train":
            return self.iterations_per_epoch
        else:
            return len(self.files)

    def load_and_preprocess_file(self, file_idx):
        with h5py.File(self.files[file_idx], 'r') as data:
            signals = data['signals']
            if (self.label != "apnea_hypopnea_labels"):  #since this is not stored automatically, we need to create it ourselves
                labels = data[self.label]
            else: 
                apnea_labels = data['apnea_labels'][:]
                hypopnea_labels = data['hypop_labels'][:]
                labels = np.maximum(apnea_labels, hypopnea_labels)  #by taking the max, we make sure it's either apnea or hypopnea
            num_signals = signals.shape[0]
            selection = min(self.batch_size, 128, num_signals)
            if self.mode == "train":
                random_indices = np.random.choice(num_signals, selection, replace=False)
                random_indices = np.sort(random_indices)
                selected_signals = signals[random_indices]
                selected_labels = labels[random_indices]
                return selected_signals, selected_labels
            return signals[:], labels[:]  # Validation/Testing

    def __getitem__(self, idx):
        if self.mode == "train":
            signals_batch = []
            labels_batch = []
            file_indices = np.random.choice(len(self.files), min(self.num_files_to_open, len(self.files)), replace=False)
            for file_idx in file_indices:
                signals, labels = self.load_and_preprocess_file(file_idx)
                signals_batch.append(signals)
                labels_batch.append(labels)
            signals_batch = np.concatenate(signals_batch, axis=0)
            labels_batch = np.concatenate(labels_batch, axis=0)
        else:
            signals_batch, labels_batch = self.load_and_preprocess_file(idx)

        return {
            "x": torch.from_numpy(signals_batch).float(),
            "y": torch.from_numpy(labels_batch).float(),
        }


def get_dataloader(files, mode, label="sleep_labels", iterations_per_epoch=2000, batch_size=64, num_workers=0, shuffle=False):
    dataset = HDFDataset(files, label, mode, iterations_per_epoch, batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader
