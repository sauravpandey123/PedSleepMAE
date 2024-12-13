import mne
import os
import pandas as pd
import numpy as np
from random import shuffle, seed
import sleep_study as ss
import h5py
from glob import glob
import torch
import argparse
from loguru import logger

from utils.generate_labels import *  
from utils.ignore_files import ignore_files
from utils.preprocess_edf import *


def save_sample_to_hdf5(signal, sleep_label, apnea_label, desat_label, eeg_label, hypop_label, patID, studID, signal_number, output_dir):
    # Construct a unique filename for each sample
    filename = f"{output_dir}/{patID}_{studID}_sample_{signal_number}.hdf5"
    with h5py.File(filename, 'w') as hdf5_file:
        # Create datasets for the signal, label, and channels
        hdf5_file.create_dataset('signal', data=signal)
        hdf5_file.create_dataset('sleep_label', data=sleep_label)
        hdf5_file.create_dataset('apnea_label', data=apnea_label)
        hdf5_file.create_dataset('desat_label', data=desat_label)
        hdf5_file.create_dataset('eeg_label', data=eeg_label)
        hdf5_file.create_dataset('hypop_label', data=hypop_label)
        hdf5_file.create_dataset('patID', data=patID)
        hdf5_file.create_dataset('studID', data=studID)

        
def main():
    parser = argparse.ArgumentParser(description="Process sleep EDF files and generate HDF5 files.")
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--dataset_dir", type=str, required=True, 
                               help="Path to the main dataset directory (e.g., '/home/NCH_Sleep_data').")
    required_args.add_argument("--edf_dir", type=str, required=True, 
                               help="Path to the directory containing EDF files ((e.g., '/home/NCH_Sleep_data/Sleep_Data').")

    parser.add_argument("--random_state", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--hdf_all_dir", type=str, default="hdf5_individual_files", 
                        help="Directory to save individual HDF5 files (default: 'hdf5_individual_files').")
    parser.add_argument("--hdf_batches_dir", type=str, default="hdf_batches", 
                        help="Directory to save batched HDF5 files (default: 'hdf5_batches').")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    random_state = args.random_state
    edf_dir = args.edf_dir
    hdf_all_dir = args.hdf_all_dir
    hdf_batches_dir = args.hdf_batches_dir

    ss.init(tmp_dir = dataset_dir)
    edf_files = [f.replace(".edf","") for f in os.listdir(edf_dir) if f.endswith('.edf')]
    os.makedirs(hdf_all_dir, exist_ok = True)
    os.makedirs(hdf_batches_dir, exist_ok = True)

    filtered_edf_files = [file for file in edf_files if file not in ignore_files][:3] #this comes from the ignore_files.py file
    print (filtered_edf_files)
    seed(random_state)
    shuffle(filtered_edf_files)
    logger.info("*************************")
    logger.info("Starting by saving each example from the EDF files as an HDF5 file...")
    logger.info("*************************")
    
    for file in filtered_edf_files:
        try:
            signals, sleep_labels, channels = get_signals_and_stages(file)
            
        except Exception as e:
            logger.info(f"Error occured when reading {file}. Skipping it...")
            logger.info(f"ERROR: {e}") #some EDF files might throw an error, so we ignore them
            continue
        channels_array = np.array(channels)
        parts = file.split("_")
        patID = parts[0]
        studID = parts[1]
        
        #Get the needed labels
        apnea_labels = getCorrespondingLabel(file, "apnea")
        desat_labels = getCorrespondingLabel(file, "desat")
        eeg_labels = getCorrespondingLabel(file, "eeg")
        hypop_labels = getCorrespondingLabel(file, "hypop")

        x = filter_and_standardize_channels(signals, channels_array) #only select the 16 channels we need and standardize them

        for signal_number in range(len(x)):
            this_signal = x[signal_number] #store the signal
            #store the corresponding labels:
            sleep_label = sleep_labels[signal_number] 
            apnea_label = apnea_labels[signal_number]
            desat_label = desat_labels[signal_number]
            eeg_label = eeg_labels[signal_number]
            hypop_label = hypop_labels[signal_number]
            save_sample_to_hdf5(this_signal, sleep_label, apnea_label, desat_label, eeg_label, hypop_label, patID, studID, signal_number, hdf_all_dir)
            
        logger.info(f"File '{patID}_{studID}.edf' processed successfully.")
        
    logger.info(f"Successfully saved all examples as individual HDF5 files in the directory: {hdf_all_dir}.")
    
    logger.info("*************************")
    
    logger.info(f"Randomly grouping individual examples into batches of 128 and saving as HDF5 files for training. Files will be stored in: {hdf_batches_dir}.")
    
    individual_files = glob(os.path.join(hdf_all_dir, '*.hdf5'))

    # Ensure we only process as many files as form complete sets of 128
    num_files_to_process = len(individual_files) - (len(individual_files) % 128)

    shuffle(individual_files)

    # Initialize lists to hold combined data
    combined_signals = []
    combined_sleep_labels = []
    combined_apnea_labels = []
    combined_desat_labels = []
    combined_eeg_labels = []
    combined_hypop_labels = []
    combined_patient_ids = []
    combined_study_ids = []
    
    batch_index = 0
    for i, file_path in enumerate(individual_files[:num_files_to_process]):
        sleep_file =  (file_path.split("/")[1])
        ids = sleep_file.split("_")
        patID = int(ids[0])
        studID = int(ids[1])
        with h5py.File(file_path, 'r') as f:
            combined_signals.append(f['signal'][()])
            combined_sleep_labels.append(f['sleep_label'][()])
            combined_apnea_labels.append(f['apnea_label'][()])
            combined_desat_labels.append(f['desat_label'][()])
            combined_eeg_labels.append(f['eeg_label'][()])
            combined_hypop_labels.append(f['hypop_label'][()])
            combined_patient_ids.append(patID)
            combined_study_ids.append(studID)

        if (i + 1) % 128 == 0:
            logger.info(f"Saving batch... {batch_index}")
            combined_file_path = f'{hdf_batches_dir}/combined_{batch_index}.hdf5'
            with h5py.File(combined_file_path, 'w') as f:
                f.create_dataset('signals', data=np.array(combined_signals))
                f.create_dataset('sleep_labels', data=np.array(combined_sleep_labels))
                f.create_dataset('apnea_labels', data=np.array(combined_apnea_labels))
                f.create_dataset('desat_labels', data=np.array(combined_desat_labels))
                f.create_dataset('eeg_labels', data=np.array(combined_eeg_labels))
                f.create_dataset('hypop_labels', data=np.array(combined_hypop_labels))
                f.create_dataset('patient_ids', data=np.array(combined_patient_ids))
                f.create_dataset('study_ids', data=np.array(combined_study_ids))

            combined_signals = []
            combined_sleep_labels = []
            combined_apnea_labels = []
            combined_desat_labels = []
            combined_eeg_labels = []
            combined_hypop_labels = []
            combined_patient_ids = []
            combined_study_ids = []
            batch_index += 1
    logger.info("Success")
    logger.info("*************************")

    
    
if __name__ == "__main__":
    main() 