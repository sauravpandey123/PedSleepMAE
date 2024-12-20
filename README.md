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
In this part, we evaluate how much diagnostic information is in the embeddings from the PedSleepMAE encoder. We have rich Electronic Health Records (EHR) and annotated clinician-verified sleep events. We fully utilize the rich Electronic Health Records (EHR) and annotated clinician-verified sleep events to quantitatively and qualitatively measure how well various sleep events are separated in the embedding space.


### 3.1 Training Linear Classifiers
Once the model is pretrained, we can perform downstream classification tasks, such as sleep scoring, apnea detection, etc, by fitting linear classifiers on top of the embeddings extracted from our pretrained model's encoder.

Run `3_train_classifier.py` for this step.
