<h1 align="center">PedSleepMAE (Pediatric Sleep Masked Autoencoder)</h1>

## :loudspeaker: News 
* PedSleepMAE was accepted at IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI’24)!
* Our [paper](https://arxiv.org/abs/2411.00718) is available on arXiv! 

## :mag: Introduction
Pediatric sleep is an important but often overlooked area in health informatics. We present PedSleepMAE, a generative model that fully leverages multimodal pediatric sleep signals including multichannel EEGs, respiratory signals, EOGs and EMG. This masked autoencoder-based model performs comparably to supervised learning models in sleep scoring and in the detection of apnea, hypopnea, EEG arousal and oxygen desaturation. Its embeddings are also shown to capture subtle differences in sleep signals coming from a rare genetic disorder. Furthermore, PedSleepMAE generates realistic signals that can be used for sleep segment retrieval, outlier detection, and missing channel imputation. This is the first general-purpose generative model trained on multiple types of pediatric sleep signals.

# :fire: Setup 
Please follow the following steps to create an environment for running PedSleepMAE

```
git clone https://github.com/sauravpandey123/PedSleepMAE.git
cd PedSleepMAE
conda env create -f environment.yml
conda activate pedsleep_env
```


# :computer: User Guide 
## :inbox_tray:  0. Downloading the Dataset
We work with the **Nationwide Children's Hospital (NCH) Sleep DataBank**, a large, public, and fairly recent pediatric Polysomnography (PSG) dataset collected in a real clinical setting. 
The dataset is available to download from [Physionet](https://physionet.org/content/nch-sleep/3.1.0/). Please take a look at the requirements that need to be met before the dataset can be downloaded. 

**Note: All the Python scripts mentioned in the following documentation are located inside the `scripts` directory.**

## :package:  1. Dataset Preparation and Preprocessing
Each sleep study has an EDF file containing raw sleep signals, electronic health records (EHR), and an annotation file documenting events such as sleep stages, apnea, and other physiological occurrences. These files are processed into the **HDF5** format to enable efficient and scalable data access during training. 

From the raw data, the **16 most common modalities** identified in the study are extracted, and all their signals are downsampled to 128 Hz and normalized to 0 mean and 1 standard deviation. Each sleep example is first converted into an individual HDF5 file, which includes patient and study IDs, sleep stage labels, and event labels such as apnea, hypopnea, oxygen desaturation, and EEG arousal. To optimize training workflows, 128 individual examples are then grouped into a single HDF5 batch file. This pipeline ensures fast data access, efficient batching, and scalability for machine learning workflows. 

`1_prepare_dataset.py` handles the above operations. 

## :arrows_counterclockwise: 2. Pretraining
We make use of a [Masked Autoencoder](https://arxiv.org/pdf/2111.06377) architecture for pretraining and learning rich representations from pediatric sleep signals. Our model is very large, with about 76 million parameters, and pretraining requires a long time - two to three weeks  - to start seeing good results. GPU support is highly recommended. 

Run `2_pretrain_model.py` to pretrain the model.

**Note: We provide the latest checkpoint of our pretrained model at `checkpoint/m15p8_checkpoint.pt` if you want to skip this step and proceed with the following experiments.**

## :brain: 3. Evaluating Diagnostic Information in the Embeddings 
Once PedSleepMAE is sufficiently pretrained, we evaluate the diagnostic information in the embeddings from the encoder of PedSleepMAE. Using rich EHR data and clinician-verified sleep events, we assess how well various sleep events are separated in the embedding space, both quantitatively and qualitatively. 

### 3.1. Training Linear Classifiers
We can perform downstream classification tasks, such as sleep scoring, apnea detection, etc, by fitting linear classifiers on top of the embeddings extracted from our pretrained model's encoder.

Run `3_1_train_classifier.py` for this step.

### 3.2. Visualization using Uniform Manifold Approximation and Projection (UMAP)
We employ UMAP to reduce our embeddings into two dimensions and visualize them. This step can be performed on just a single patient by specifying their patient ID or on randomly selected patients. 

Run `3_2_get_UMAP.py` to handle the above operations. 

### 3.3. Prader-Willi Syndrome (PWS) Cluster Analysis
To perform cluster analysis on PWS patients versus non-PWS patients, we provide the script `3_3_PWS_cluster_analysis.py`. It loads the provided maxpooled PWS and non-PWS embeddings (stored inside `experiments_config/PWS`) and uses them to compare the true silhouette score against silhouette score of randomly shuffled labels.

## :dart:  4. Accuracy of the Generated Signals
After establishing the usefulness of the embeddings, we bring the decoder back into the picture and see how it can be used to retrieve representative examples or impute missing channels.

### 4.1. Correlation between distances in embedding space and signal space
To examine the correlation between the distances, which can be calculated using either Euclidean distances or Dynamic Time Warping (DTW), of the examples in embedding space versus signal space, we provide the script `4_1_correlation_between_spaces.py`. 

### 4.2. Generating and Retrieving Representative Examples
Next, we can tackle representative signal generation and retrieval by considering a single patient, its sleep examples related to an event (e.g. REM sleep stage, apnea, etc), generating an average example, and retrieving an actual data sample closest or furthest from this mean.

Run `4_2_generate_average_examples.py` to handle the above operations. 

### 4.3. Channel Imputation 
Run `4_3_channel_imputation.py` to run the experiment for missing channel imputation, where we systematically remove one channel at a time and let PedSleepMAE reconstruct the missing signal based on the other 15 channels. 

# :memo: BibTeX
`Coming Soon after the Proceedings is out`

# :scroll: License
Please read the [MIT License](https://github.com/sauravpandey123/PedSleepMAE/blob/main/LICENSE) statement provided in the repo. 
