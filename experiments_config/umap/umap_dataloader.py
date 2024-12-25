import json
import torch
import numpy as np
import h5py

from torch.utils.data import Dataset, DataLoader


class HDF5Dataset(Dataset):
    def __init__(self, file_list, search_label):
        self.file_list = file_list
        self.search_label = search_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with h5py.File(file_path, 'r') as data:
            signal = data['signal'][...]
            label = data[self.search_label][...]
            signal_tensor = torch.from_numpy(signal)
            label_tensor = torch.from_numpy(label)
            return signal_tensor, label_tensor


def produceJson(list_of_hdf_files, 
                search_label, 
                pool,
                device,
                MAEmodel,
                num_patches,
                emb_dim,
                needed_patient_ID, 
                include_all_patients,
                json_dir
               ):
    
    mydict={}
    
    batch_size = 100
    
    dataset = HDF5Dataset(list_of_hdf_files, search_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    target_value = 1000 #Basically take no more than 1000 samples per class to make sure we don't store too much that won't be used later
    
    for s, l in data_loader:
        with torch.no_grad():
            signals = s.squeeze().float().to(device)
            labels = l.squeeze().float().to(device)
            a,b,c = signals.shape
            if (a < batch_size):
                break
            encoded_features, _ = MAEmodel.encoder(signals)  
            encoded_features = encoded_features[:,:,1:,:]
            encoded_features = encoded_features[:,:,:,:] 
            encoded_reshaped = encoded_features.reshape(-1, num_patches, emb_dim)
            pooled = pool(encoded_reshaped)
            pooled = pooled.squeeze(-1)
            flattened_features = pooled.reshape(batch_size, -1)

            for class_index, class_number in enumerate(labels):
                class_key = str(int(class_number.item()))
                if class_key not in mydict:
                    mydict[class_key] = []
                if (len(mydict[class_key]) < target_value):
                    mydict[class_key].append(flattened_features[class_index].tolist())
            
            if (include_all_patients):
                all_match = True
                for key, value in mydict.items():
                    if len(value) < target_value:
                        all_match = False
                if (all_match == True):
                    print ("All classes meet their length requirement...")
                    break;
                    
    jsonfile = json_dir + "/" + search_label + "_" + needed_patient_ID +  ".json"
    with open(jsonfile, 'w') as json_file:
        json.dump(mydict, json_file)
    print ("Json successfully created and saved:", jsonfile)
