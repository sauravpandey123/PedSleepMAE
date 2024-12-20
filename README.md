# PedSleepMAE (Pediatric Sleep Masked Autoencoder)

## News 
* PedSleepMAE was accepted at IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI’24)!

## 1. Dataset Preparation and Preprocessing
Each patient has an EDF file containing raw sleep signals and an annotation file documenting events such as sleep stages, apnea, and other physiological occurrences. These files are processed into the **HDF5** format to enable efficient and scalable data access during training. 

From the raw data, the 16 most common modalities identified in the study are extracted, and all signals are standardized for consistency. Each example is first converted into an individual HDF5 file, which includes patient and study IDs, sleep stage labels, and event labels such as apnea, hypopnea, oxygen desaturation, and EEG arousal. To optimize training workflows, 128 individual examples are then grouped into a single HDF5 batch file. This pipeline ensures fast data access, efficient batching, and scalability for machine learning workflows.

`1_prepare_dataset.py` handles the above operations. 

## 2. Pretraining
We make use of a [Masked Autoencoder](https://arxiv.org/pdf/2111.06377) architecture for pretraining and learning rich representations from pediatric sleep signals. Our model is very large, with about 76 million parameters, and pretraining requires a long time - two to three weeks  - to start seeing good results. GPU support is highly recommended. 

Run `2_pretrain_model.py` to pretrain the model.

**Note:** We provide the latest checkpoint of our pretrained model at `checkpoint/m15p8_checkpoint.pt` if you want to skip this step and proceed with the following experiments.


## 3. Evaluating Diagnostic Information in the Embeddings 
Once PedSleepMAE is sufficiently pretrained, we evaluate the diagnostic information in the embeddings from the encoder of PedSleepMAE. Using rich EHR data and clinician-verified sleep events, we assess how well various sleep events are separated in the embedding space, both quantitatively and qualitatively. 


### 3.1 Training Linear Classifiers
We can perform downstream classification tasks, such as sleep scoring, apnea detection, etc, by fitting linear classifiers on top of the embeddings extracted from our pretrained model's encoder.

Run `3_1_train_classifier.py` for this step.

### 3.2 Visualization using Uniform Manifold Approximation and Projection (UMAP)
We employ UMAP to reduce our embeddings into two dimensions and visualize them. This step can be performed on just a single patient by specifying their patient ID or on randomly selected patients. To avoid cluttering in the plots, we only plot a maximum of 600 examples from each sub-label (e.g. 600 from apnea, no apnea each). 

Run `3_2_get_UMAP.py` to handle the above operations. 

### 
