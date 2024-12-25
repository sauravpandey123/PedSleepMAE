import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from random import shuffle
from torch.utils.data import DataLoader
from classifier_config.MAE_model_downstream import PedSleepMAE
from experiments_config.avg_example.avg_example_dataloader import HDF5Dataset
from experiments_config.avg_example.DTW_calculations import apply_DTW
from utils.misc import setup_seed, log_to_file


def main(args):
    seed = args.seed
    directory_path = args.dataset_directory_path
    needed_patient_ID = args.needed_patient_ID
    method = args.method 
    checkpoint_file = args.checkpoint_file
    output_dir = args.output_dir
    log_dir = args.log_dir
    log_file_name = args.log_file_name
    batch_size = args.batch_size
    patch_size = args.patch_size
    mask_ratio = args.mask_ratio 
    emb_dim = args.emb_dim
    num_head = args.num_head
    num_layer = args.num_layer 
    single_patient = args.single_patient
    total_examples = args.random_examples_to_select

    samples_to_select_plot = 2000 #points to plot to avoid cluttering
    num_channels = 16 
    num_patches = int(3840/patch_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)
    log_file = f'{log_dir}/{log_file_name}'
    
    setup_seed(seed)
    list_of_hdf_files = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('.hdf5')]
    shuffle(list_of_hdf_files)
    
    if (single_patient):
        list_of_hdf_files = [myfile for myfile in list_of_hdf_files if myfile.split("/")[-1].split("_")[0] == needed_patient_ID]
    else:
        print (f"Selecting {total_examples} random examples for calculations.")
        list_of_hdf_files = list_of_hdf_files[:total_examples]
        needed_patient_ID = 'random patients'

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
    print ("Total examples for the patient(s):", len(list_of_hdf_files))

    dataset = HDF5Dataset(list_of_hdf_files, search_label = 'apnea_label')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Adjust batch_size as needed

    signals = []
    all_encoded_features = []
    all_reconstructed_signals = []
    all_patch_indices = []

    for signal, label in data_loader:
        with torch.no_grad():
            signal = signal.squeeze().float().to(device)
            label = label.squeeze().float().to(device)
            encoded_features, _ = MAEmodel.encoder(signal)
            all_encoded_features.append(encoded_features)

    all_encoded_features = torch.cat(all_encoded_features, dim = 0)        

    encoded_features_flat = all_encoded_features.view(all_encoded_features.size(0), -1)
    encoded_features_flat = encoded_features_flat.cpu()

    if method == "euclidean":
        embedding_distances = torch.cdist(encoded_features_flat, encoded_features_flat, p=2).cpu().detach().numpy()
    else:
        print ("Applying Dynamic Time Warping (DTW).\nThis may take a while...")
        embedding_distances = apply_DTW(all_encoded_features.detach().cpu().numpy(), is_embeddings = True)

    #Patch indices should include all patches so that we don't add mask tokens anywhere
    patch_indices = [
        [list(range(num_patches)) for _ in range(num_channels)] 
        for _ in range(batch_size)
    ]

    with torch.no_grad():
        MAEmodel.decoder.eval() 
        for i in range(0, len(all_encoded_features), batch_size):
            sub_features = all_encoded_features[i:min(i+batch_size, len(all_encoded_features))]
            reconstructed_signals, _ = MAEmodel.decoder(features=sub_features, patch_indices=patch_indices)
            all_reconstructed_signals.append(reconstructed_signals)

    all_reconstructed_signals = torch.cat(all_reconstructed_signals, dim=0)

    if method!="euclidean":
        time_space_distances = apply_DTW(all_reconstructed_signals.detach().cpu().numpy())
    else: 
        #calculate euclidean distances
        reconstructed_signals_flat = all_reconstructed_signals.view(all_reconstructed_signals.size(0), -1)
        reconstructed_signals_flat = reconstructed_signals_flat.cpu()
        time_space_distances = torch.cdist(reconstructed_signals_flat, reconstructed_signals_flat, p=2).cpu().detach().numpy()

    upper_triangle_indices = np.triu_indices_from(embedding_distances, k=1)
    embedding_distances_flat = embedding_distances[upper_triangle_indices]
    time_space_distances_flat = time_space_distances[upper_triangle_indices]

    random_indices = np.random.choice(len(embedding_distances_flat), samples_to_select_plot, replace=False)
    sampled_embedding_distances = embedding_distances_flat[random_indices]
    sampled_time_space_distances = time_space_distances_flat[random_indices]

    plt.figure(figsize=(10, 6), dpi=300)  
    plt.scatter(sampled_embedding_distances, sampled_time_space_distances, alpha=0.4)

    plt.xlabel('Embedding Space Distances', fontsize=14)
    plt.ylabel('Time Space Distances', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linewidth=0.5)
    plt.savefig(f'{output_dir}/correlation_embedding_vs_real_ID_{needed_patient_ID}')

    plt.show()

    correlation = scipy.stats.pearsonr(embedding_distances_flat, time_space_distances_flat)[0]

    if (single_patient):
        message = f'Processed ID {needed_patient_ID}\n' 
    else:
        message = f'Processed random patient IDs and {total_examples} randomly selected examples\n'
    log_to_file(log_file, f'{message}Correlation between embedding space distances and time space distances ({method}): {correlation}')
    
    
if __name__ == "__main__":
    
    valid_labels = ['dtw','euclidean']
    description = "Examining relationships between Signals in Embedding versus Time Space\n"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('method', type=str, choices=valid_labels, help='Method to Calculate Distances', default='euclidean')

    parser.add_argument("--checkpoint_file", type=str, default = 'checkpoint/m15p8_checkpoint.pt', help="Path to the checkpoint file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset_directory_path", type=str, default="hdf5_individual_files", help="Path to the dataset directory containing INDIVIDUAL hdf files.")
    parser.add_argument("--output_dir", type=str, default="avg_example_plots", help="Directory to save the correlation plots.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save the logs of the program.")
    parser.add_argument("--log_file_name", type=str, default="correlation_plots_log.txt", help="File to write logs to.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size.")
    parser.add_argument("--mask_ratio", type=int, default=15, help="Mask ratio.")
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension for the transformer.')
    parser.add_argument('--num_head', type=int, default=4,help='Number of attention heads.')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of transformer layers.')
    parser.add_argument("--needed_patient_ID", type=str, default="7093", help="Patient ID to process (if just one patient)")
    parser.add_argument("--single_patient", action="store_true", help="Process just one patient?")
    parser.add_argument("--random_examples_to_select", type=int, default = 80, help="If selecting random patients, how many examples to select?")

    
    args = parser.parse_args()
    
    main(args)