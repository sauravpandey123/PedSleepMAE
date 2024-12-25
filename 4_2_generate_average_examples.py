import os
import sys
import argparse
import torch
import h5py

from sklearn.neighbors import NearestNeighbors
from random import shuffle
from torch.utils.data import DataLoader
from classifier_config.MAE_model_downstream import PedSleepMAE
from experiments_config.avg_example.avg_example_dataloader import HDF5Dataset
from experiments_config.avg_example.avg_example_utils import *
from utils.misc import setup_seed

def main(args):
    seed = args.seed
    directory_path = args.directory_path
    needed_patient_ID = args.needed_patient_ID
    checkpoint_file = args.checkpoint_file
    output_dir = args.output_dir
    batch_size = args.batch_size
    patch_size = args.patch_size
    mask_ratio = args.mask_ratio 
    emb_dim = args.emb_dim
    num_head = args.num_head
    num_layer = args.num_layer 
    search_label = args.search_label
    sleep_stage = args.sleep_stage 
    select_KNN_result = args.select_KNN_plot   
    
    num_channels = 16 
    num_patches = int(3840/patch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_seed(seed)
    list_of_hdf_files = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('.hdf5')]
    shuffle(list_of_hdf_files)
    
    list_of_hdf_files = [myfile for myfile in list_of_hdf_files if myfile.split("/")[-1].split("_")[0] == needed_patient_ID]

    if (len(list_of_hdf_files) == 0):
        print ("Invalid Patient ID provided.")
        sys.exit(1)

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
    print ("Patient ID:", needed_patient_ID)
    print ("Total examples for the patient:", len(list_of_hdf_files))

    dataset = HDF5Dataset(list_of_hdf_files, search_label = search_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Adjust batch_size as needed

    all_signals = []
    all_encoded_features = []
    all_reconstructed_signals = []

    #do the following to collect the embeddings in batches and not all at once to avoid any memory issues
    for signal, labels in data_loader:
        with torch.no_grad():
            sub_signals = signal.squeeze().float().to(device)
            indices = get_needed_indices(labels, search_label, sleep_stage)
            if (len(indices)>0):  #make sure indices is non-empty
                selected_signals = sub_signals[indices]
                encoded_features, _ = MAEmodel.encoder(selected_signals)
                all_signals.append(selected_signals)
                all_encoded_features.append(encoded_features)
        
    if (len(all_encoded_features) == 0):
        print (f"Sorry, no case of {search_label} found for the patient")
        sys.exit(1)
    
    all_encoded_features = torch.cat(all_encoded_features, dim = 0)    
    all_signals = torch.cat(all_signals, dim = 0)
    
    #get the average in embedding space
    average_embeddings = all_encoded_features.mean(dim = 0)
        
    #Patch indices should include all patches so that we don't add mask tokens anywhere
    patch_indices = [
        [list(range(num_patches)) for _ in range(num_channels)] 
        for _ in range(batch_size)
    ]

    #Now get the average in signal space
    reconstructed_average_embedding, _ = MAEmodel.decoder(average_embeddings.unsqueeze(0), patch_indices)
            
    print("Starting KNN algorithm...")
    number_labels = len(all_signals)
    reconstructed_flat = reconstructed_average_embedding.view(reconstructed_average_embedding.shape[0], -1).detach().cpu().numpy()
    original_flat = all_signals.view(all_signals.shape[0], -1).cpu().numpy()
    nn = NearestNeighbors(n_neighbors=number_labels)
    nn.fit(original_flat)
    distances, indices = nn.kneighbors(reconstructed_flat)
    
    closest_signal_index = indices[0][0]
    closest_signal = all_signals[closest_signal_index]
    closest_distance = distances[0][0]

    furthest_signal_index = indices[0][-1]
    furthest_signal = all_signals[furthest_signal_index]
    furthest_distance = distances[0][-1]

    average = reconstructed_average_embedding
    closest = closest_signal
    furthest = furthest_signal
    
    plot_KNN_results(
        average, 
        closest,
        furthest, 
        needed_patient_ID,
        output_dir,
        select_KNN_result,
        search_label
    )
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generating and Retrieving Representative Examples", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    label_choices = ['sleep_label', 'apnea_label', 'hypop_label', 'eeg_label', 'desat_label']
    sleep_stage_choices = [0,1,2,3,4]
    knn_plot_choices = ['average', 'closest', 'furthest']
    
    
    #ones having choice restriction
    parser.add_argument("--search_label", type=str, default='sleep_label', help="Label to search and train on", choices = label_choices)
    parser.add_argument("--sleep_stage", type=int, default=4, 
                        help="Sleep stage to analyze (only relevant if looking at sleep scoring)", choices = sleep_stage_choices)
    
    parser.add_argument("--select_KNN_plot", type=str, default='average', help="Example type to plot", choices = knn_plot_choices)
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--directory_path", type=str, default='hdf5_individual_files', 
                        help="Path to the directory containing INDIVIDUAL HDF5 files")
    parser.add_argument("--needed_patient_ID", type=str, default='4768', help="Patient ID to process")
    parser.add_argument("--checkpoint_file", type=str, default='checkpoint/m15p8_checkpoint.pt', help="Path to the MAE checkpoint file")
    parser.add_argument("--output_dir", type=str, default='avg_example_plots', help="Directory to save output plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for the model")
    parser.add_argument("--mask_ratio", type=int, default=15, help="Masking ratio for the model")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension for the transformer")
    parser.add_argument("--num_head", type=int, default=4, help="Number of attention heads in the transformer")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of transformer layers")

    args =  parser.parse_args()
    main(args)
    