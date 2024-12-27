import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import h5py
import warnings
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from random import shuffle
from classifier_config.MAE_model_downstream import PedSleepMAE
from utils.misc import setup_seed
from utils.ignore_files import ignore_files 
from experiments_config.umap import umap_dataloader
from experiments_config.umap import umap_plot


def main(args):
    seed = args.seed
    search_label = args.search_label 
    directory_path = args.directory_path
    umap_dir = args.umap_dir
    checkpoint_file = args.checkpoint_file
    patch_size = args.patch_size 
    mask_ratio = args.mask_ratio  
    emb_dim = args.emb_dim
    num_head = args.num_head
    num_layer = args.num_layer
    needed_patient_ID = args.needed_patient_ID 
    include_all_patients = args.include_all_patients 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_patches = int(3840/patch_size)  
    
    setup_seed(seed)
    os.makedirs(umap_dir, exist_ok=True)

    list_of_hdf_files = [os.path.join(directory_path, x) for x in os.listdir(directory_path) if x.endswith('.hdf5')]
    shuffle(list_of_hdf_files)

    if (not include_all_patients):
        list_of_hdf_files = [myfile for myfile in list_of_hdf_files if myfile.split("/")[-1].split("_")[0] == needed_patient_ID]
    else:
        needed_patient_ID = "Random Patients"

    if len(list_of_hdf_files) == 0:
        print("Invalid Patient ID")
        sys.exit(1)   
    
    batch_size = len(list_of_hdf_files)
    
    MAEmodel = PedSleepMAE(
        batch_size=batch_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        emb_dim=emb_dim,
        num_head=num_head,
        num_layer=num_layer
    ).to(device)


    checkpoint = torch.load(checkpoint_file, weights_only = True)
    MAEmodel.load_state_dict(checkpoint['state_dict'])
    
    print ("Running on:", device)
    print ("Patient ID:", needed_patient_ID)
    print ("Total Patient Examples to Select from:", len(list_of_hdf_files))

    pool = nn.AdaptiveMaxPool1d(1)  #define Max pooling
    
    print ("Producing Json that will be used for plotting, will take a few minutes and will print confirmation...")
    umap_dataloader.produceJson(
        list_of_hdf_files, 
        search_label, 
        pool,
        device,
        MAEmodel,
        num_patches,
        emb_dim,
        needed_patient_ID, 
        include_all_patients,
        umap_dir
    )

    #We want to use the correct legends for each of the labels
    if search_label == "apnea_label":
        class_real_names = ["No Apnea", "Central Apnea", "Obstructive Apnea", "Mixed Apnea"] #[0,1,2,3]
    elif search_label == "desat_label":
        class_real_names = ["No Oxygen Desaturation", "Oxygen Desaturation"]
    elif search_label == "eeg_label":
        class_real_names = ["No EEG Arousal", "EEG Arousal"]
    elif search_label == "hypop_label":
        class_real_names = ["No Hypopnea", "Obstructive Hypopnea", "Hypopnea"] #[0,1,2]
    elif search_label == 'sleep_label':
        class_real_names = ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R'] #[0,1,2,3,4]

    file_path = f"{umap_dir}/{search_label}_{str(needed_patient_ID)}.json"
    with open(file_path, 'r') as file:
        my_dictionary = json.load(file)

    this_title_label = f"{search_label} example plot"  #to get the correct title for the plot
    umap_plot.plot_UMAP(umap_dir, my_dictionary, search_label, needed_patient_ID, class_real_names)
    print ("Successfully saved:", file_path[:-5] + ".pdf")


if __name__ == "__main__":
    
    valid_labels = {
        "apnea_label": "Apnea",
        "hypop_label": "Hypopnea",
        "desat_label": "Oxygen Desaturation",
        "eeg_label": "EEG Arousal",
        "sleep_label": "Sleep Scoring"
    }
    
    description = "UMAP Plotter for NCH Sleep Data Bank\n"
    description += "Choose a label from the following valid labels:\n"
    description += "Label -> Meaning\n"
    description += "\n".join([f"{label} -> {meaning}\n" for label, meaning in valid_labels.items()])
    
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description=description, formatter_class=CustomFormatter)
        
    parser.add_argument('search_label', type=str, choices=valid_labels.keys(), help='Label to plot UMAP on', default='sleep_label')

    parser.add_argument("--checkpoint_file", type=str, default = '../checkpoint/m15p8_checkpoint.pt', help="Path to the MAE checkpoint file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--directory_path", type=str, default="hdf5_individual_files", help="Path to the dataset directory containing INDIVIDUAL hdf files.")
    parser.add_argument("--umap_dir", type=str, default="umap_plots", help="Directory to save UMAP plots.")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size.")
    parser.add_argument("--mask_ratio", type=int, default=15, help="Mask ratio.")
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension for the transformer.')
    parser.add_argument('--num_head', type=int, default=4,help='Number of attention heads.')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of transformer layers.')
    parser.add_argument("--needed_patient_ID", type=str, default="7093", help="Patient ID to process (if just one patient).")
    parser.add_argument("--include_all_patients", action="store_true", help="Include random selection of patients in processing?")
    
    args = parser.parse_args()
    
    main(args)
