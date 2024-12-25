import numpy as np
from sklearn.metrics import silhouette_score


def get_random_silhouette_data(
    encoded_features_pws,
    encoded_features_non_pws,
    num_random_iterations, 
    num_iterations, 
    num_samples_per_class
):
    
    print ("Calculating random silhouette scores...")
    
    random_silhouette_means = []
    random_silhouette_conf_intervals = [] 
    
    for i in range(num_random_iterations):
        
        print (f"Completed random iterations: {i + 1}/{num_random_iterations}")
        
        random_silhouette_scores = []
        shuffled_labels = np.random.permutation(np.concatenate((np.ones(num_samples_per_class), np.zeros(num_samples_per_class)), axis=0))

        for _ in range(num_iterations):
            indices_pws = np.random.choice(len(encoded_features_pws), num_samples_per_class, replace=False)
            indices_non_pws = np.random.choice(len(encoded_features_non_pws), num_samples_per_class, replace=False)
            sampled_pws = encoded_features_pws[indices_pws]
            sampled_non_pws = encoded_features_non_pws[indices_non_pws]
            X_combined = np.concatenate((sampled_pws, sampled_non_pws), axis=0)
            random_score = silhouette_score(X_combined, shuffled_labels)
            random_silhouette_scores.append(random_score)
            
        random_mean_score = np.mean(random_silhouette_scores)
        random_conf_interval = np.percentile(random_silhouette_scores, [2.5, 97.5])
        random_silhouette_means.append(random_mean_score)
        random_silhouette_conf_intervals.append(random_conf_interval)
    
    print ("Calculated random silhouette scores...")
    return np.array(random_silhouette_means), np.array(random_silhouette_conf_intervals)


def get_true_silhouette_data(
    encoded_features_pws,
    encoded_features_non_pws,
    num_iterations,
    num_samples_per_class,
):
    
    print ("Getting true silhouette score...")
    true_silhouette_scores = []
    for _ in range(num_iterations):

        indices_pws = np.random.choice(len(encoded_features_pws), num_samples_per_class, replace=False)
        indices_non_pws = np.random.choice(len(encoded_features_non_pws), num_samples_per_class, replace=False)

        sampled_pws = encoded_features_pws[indices_pws]
        sampled_non_pws = encoded_features_non_pws[indices_non_pws]

        X_combined = np.concatenate((sampled_pws, sampled_non_pws), axis=0)

        labels_combined = np.concatenate((np.ones(num_samples_per_class), np.zeros(num_samples_per_class)), axis=0)
        true_score = silhouette_score(X_combined, labels_combined)
        true_silhouette_scores.append(true_score)

        # print(f'Iteration {_ + 1}/{num_iterations} - True Silhouette Score: {true_score}')

    mean_true_score = np.mean(true_silhouette_scores)
    conf_interval_true = np.percentile(true_silhouette_scores, [2.5, 97.5])

    print(f'Mean True Silhouette Score: {mean_true_score}')
    print(f'95% Confidence Interval for True Scores: {conf_interval_true}')
    return mean_true_score, conf_interval_true