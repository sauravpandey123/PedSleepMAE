# PedSleepMAE (Pediatric Sleep Masked Autoencoder)

## News 
* PedSleepMAE was accepted at IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI’24)!

## Dataset Preparation and Preprocessing
For each patient, there is a corresponding EDF file consisting of the sleep signals and an annotation file consisting of the various events occuring during sleep. Once they have been downloaded, we want to convert each example into an HDF5 format, which is known for its easy and quick access during training. Out of the many modaltiies present in the files, we extract the 16 most common ones listed in the paper and standardize them. Each HDF5 file also contains its patient and study ID, along with corresponding sleep stage, apnea label, hypopnea label, oxygen desaturation label, and EEG arousal label. We then randomly select 128 individual examples and store them as an HDF5 files. These will be the files we will be using when training our model. 
