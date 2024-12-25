import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean 


def compute_dtw_distance(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

def apply_DTW(encoded_features, is_embeddings = False):
    if (is_embeddings):
        batch_size, channels, _, _ = encoded_features.shape
        encoded_features = encoded_features.reshape(batch_size, channels, -1)
    num_samples = encoded_features.shape[0]
    dtw_distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            distance = compute_dtw_distance(encoded_features[i], encoded_features[j])
            dtw_distances[i, j] = distance
            dtw_distances[j, i] = distance  # Symmetric matrix
            
    embedding_distances = torch.tensor(dtw_distances)
    return embedding_distances