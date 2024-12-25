import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, file_list, search_label = 'sleep_label'):
        self.file_list = file_list
        self.search_label = search_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, 'r') as data:
            signal = data['signal'][...]
            label_tensor = torch.from_numpy(data[self.search_label][...])
            signal_tensor = torch.from_numpy(signal)
            return signal_tensor, label_tensor