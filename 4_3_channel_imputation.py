import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy

from sklearn.neighbors import NearestNeighbors
from random import shuffle
from torch.utils.data import DataLoader
from experiments_config.imputation.MAE_model_imputation import PedSleepMAE
from experiments_config.avg_example.avg_example_dataloader import HDF5Dataset
from experiments_config.imputation.distance_calculations import *
from utils.misc import setup_seed, log_to_file
from utils.channel_info import filter_channels


def main(args):
    seed = args.seed
    directory_path = args.directory_path
    checkpoint_file = args.checkpoint_file
    log_dir = args.log_dir
    log_file_name = args.log_file_name
    batch_size = args.batch_size
    patch_size = args.patch_size
    mask_ratio = args.mask_ratio 
    emb_dim = args.emb_dim
    num_head = args.num_head
    num_layer = args.num_layer 
    
    num_channels = 16 
    num_patches = int(3840/patch_size)    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    list_of_hdf_files = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('.hdf5')]
    os.makedirs(log_dir, exist_ok = True)
    log_file = f'{log_dir}/{log_file_name}'
    setup_seed(seed)
    shuffle(list_of_hdf_files)


    MAEmodel = PedSleepMAE(
        batch_size=batch_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        emb_dim = emb_dim, 
        num_head = num_head,
        num_layer = num_layer
    ).to(device)

    checkpoint = torch.load(checkpoint_file, weights_only = True)
    MAEmodel.load_state_dict(checkpoint['state_dict'])

    print ("Running on:",device)
    print ("Total examples:", len(list_of_hdf_files))

    dataset = HDF5Dataset(list_of_hdf_files)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Adjust batch_size as needed

    all_signals = []
    all_encoded_features = []
    all_reconstructed_signals = []
    
    for remove_index in range(0,num_channels,1):
        torch.cuda.empty_cache()
        all_encoded_features_original = []
        all_encoded_features_modified = []
        all_reconstructed_signals_original = []
        all_reconstructed_signals_modified = []    
        
        log_to_file(log_file, f'Removing Channel {filter_channels[remove_index]}, index {remove_index}')
        
        #do the following to collect the embeddings in batches and not all at once to avoid any memory issues
        for signal, _ in data_loader:
            with torch.no_grad():
                signals_original = signal.squeeze().float().to(device)
                signals_modified = remove_channel(signals_original, remove_index)
                encoded_features_original, _ = MAEmodel.encoder(signals_original)
                all_encoded_features_original.append(encoded_features_original.cpu())

                encoded_features_modified, _ = MAEmodel.encoder(signals_modified)
                all_encoded_features_modified.append(encoded_features_modified.cpu())
                torch.cuda.empty_cache()

        all_encoded_features_original = torch.cat(all_encoded_features_original, dim=0)
        all_encoded_features_modified = torch.cat(all_encoded_features_modified, dim=0)
        
        #Reconstruction part:
        with torch.no_grad():
            MAEmodel.decoder.eval() 
            for i in range(0, len(all_encoded_features_original), batch_size):
                sub_original_features = all_encoded_features_original[i:min(i+batch_size, len(all_encoded_features_original))]
                reconstructed_signals_original = MAEmodel.decoder(sub_original_features.to(device), remove_index, remove = False)
                all_reconstructed_signals_original.append(reconstructed_signals_original.cpu())

                sub_modified_features = all_encoded_features_modified[i:min(i+batch_size, len(all_encoded_features_original))]
                reconstructed_signals_modified = MAEmodel.decoder(sub_modified_features.to(device), remove_index, remove = True) 
                all_reconstructed_signals_modified.append(reconstructed_signals_modified.cpu())
                torch.cuda.empty_cache() 

        all_reconstructed_signals_original = torch.cat(all_reconstructed_signals_original, dim=0)
        all_reconstructed_signals_modified = torch.cat(all_reconstructed_signals_modified, dim=0)

        mse_distances_per_sample = calculate_pairwise_mse(
            all_reconstructed_signals_original, all_reconstructed_signals_modified
        )

        dtw_distances_per_sample = calculate_pairwise_dtw(
            all_reconstructed_signals_original, all_reconstructed_signals_modified
        )

        mse_distances_per_sample = np.array(mse_distances_per_sample)
        mean_mse = np.mean(mse_distances_per_sample)
        std_mse = np.std(mse_distances_per_sample)

        dtw_distances_per_sample = np.array(dtw_distances_per_sample)
        mean_dtw = np.mean(dtw_distances_per_sample)
        std_dtw = np.std(dtw_distances_per_sample)

        log_to_file(log_file, f'Mean MSE: {mean_mse}')
        log_to_file(log_file, f'Standard Deviation MSE: {std_mse}')
        log_to_file(log_file, f'Mean DTW: {mean_dtw}')
        log_to_file(log_file, f'Standard Deviation DTW: {std_dtw}')
        log_to_file(log_file, f'====================')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Channel Imputation Experiment", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--directory_path", type=str, default='hdf5_individual_files', 
                        help="Path to the directory containing INDIVIDUAL HDF5 files")
    
    parser.add_argument("--checkpoint_file", type=str, default='checkpoint/m15p8_checkpoint.pt', 
                        help="Path to the MAE checkpoint file")
    
    parser.add_argument("--log_dir", type=str, default='logs', help="Directory to save the logs")
    parser.add_argument("--log_file_name", type=str, default='imputation_results.txt', help="File name to save the logs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for the model")
    parser.add_argument("--mask_ratio", type=int, default=15, help="Masking ratio for the model")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension for the transformer")
    parser.add_argument("--num_head", type=int, default=4, help="Number of attention heads in the transformer")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of transformer layers")

    args =  parser.parse_args()
    main(args)
    