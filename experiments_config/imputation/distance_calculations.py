from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw


def remove_channel(data, channel_index):
    modified_data = data.clone()
    modified_data[:, channel_index, :] = 0  # Set the channel to zero
    return modified_data

def calculate_pairwise_mse(original, modified):
    mse_distances_per_sample = []
    for orig_sample, mod_sample in zip(original, modified):
        mse_sum = 0
        for orig_channel, mod_channel in zip(orig_sample, mod_sample):
            mse_sum += mean_squared_error(orig_channel.cpu().numpy(), mod_channel.cpu().numpy())
        mse_avg = mse_sum / orig_sample.size(0)
        mse_distances_per_sample.append(mse_avg)
    
    return mse_distances_per_sample


def calculate_pairwise_dtw(original, modified):
    distances = []
    for orig_sample, mod_sample in zip(original, modified):
        dist = 0
        for orig_channel, mod_channel in zip(orig_sample, mod_sample):
            channel_dist, _ = fastdtw(orig_channel.cpu().numpy(), mod_channel.cpu().numpy())
            dist += channel_dist
        distances.append(dist / len(orig_sample))  # Average over channels
    return distances   