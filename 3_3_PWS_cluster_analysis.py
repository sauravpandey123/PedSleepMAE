import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import silhouette_score
from experiments_config.PWS.silhouette_calculations import *
from utils.misc import setup_seed

def main(args):
    seed = args.seed
    num_random_iterations = args.num_random_iterations
    num_iterations = args.num_iterations
    num_samples_per_class = args.num_samples_per_class
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok = True) 

    setup_seed(seed)
    
    print ("Loading embeddings...")
    encoded_features_pws = np.load('experiments_config/PWS/pws_embeddings.npz')['arr_0']
    encoded_features_non_pws = np.load('experiments_config/PWS/non_pws_embeddings.npz')['arr_0']

    print ("Loading complete...")
    print ("Shape of pooled embeddings (PWS): ", encoded_features_pws.shape)
    print ("Shape of pooled embeddings (non-PWS): ", encoded_features_non_pws.shape)
    
    random_means, random_conf_intervals = get_random_silhouette_data(
        encoded_features_pws,
        encoded_features_non_pws,
        num_random_iterations,
        num_iterations,
        num_samples_per_class
    )

    true_mean, true_conf_interval = get_true_silhouette_data(
        encoded_features_pws,
        encoded_features_non_pws,
        num_iterations,
        num_samples_per_class
    )
    

    x = [i for i in range(1,num_random_iterations + 2,1)]    
    true_error = np.array([[true_mean - true_conf_interval[0]], [true_conf_interval[1] - true_mean]])
    random_errors = np.array([[mean - ci[0], ci[1] - mean] for mean, ci in zip(random_means, random_conf_intervals)]).T
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.errorbar(x=[x[0]], y=[true_mean], yerr=true_error, fmt='o', color='blue', capsize=5, label='True Labels', markersize=3)
    ax.errorbar(x=x[1:], y=random_means, yerr=random_errors, fmt='o', color='red', capsize=5, label='Random Labels', markersize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['True Labels'] + list(np.arange(1, len(random_means) + 1)), rotation=0)

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')

    ax.axhline(y=true_mean, color='blue', linestyle='--')

    ax.tick_params(axis='x', which='major', labelsize=8)  # Adjust this value for smaller font
    ax.tick_params(axis='y', which='major', labelsize=8)

    ax.legend(fontsize=8)

    plt.tight_layout()

    output_file_path = f'{output_dir}/silhouette_random-iters_{num_random_iterations}_total-iters_{num_iterations}_samples_{num_samples_per_class}.pdf'
    plt.savefig(output_file_path, dpi=300, format='pdf')
    plt.show()

    print(f"Plot saved as {output_file_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PWS Cluster Analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_samples_per_class', type=int, default = 2500, help=f'No. of random samples from each class (PWS, non-PWS).')
    parser.add_argument('--num_iterations', type=int, default = 100, help=f'Number of times to repeat the sampling process')
    parser.add_argument('--num_random_iterations', type=int, default = 20, help=f'Number of times to repeat the above, entire cluster analysis.')
    parser.add_argument('--output_dir', type=str, default = 'PWS_plots', help=f'Directory to store analysis plot.')
    parser.add_argument('--seed', type=int, default = 42, help=f'Seed for reproducability.')
    
    args = parser.parse_args()
    main(args)