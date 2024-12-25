import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings(
    "ignore", 
    message="n_jobs value 1 overridden to 1 by setting random_state"
)
warnings.filterwarnings(
    "ignore", 
    message="The palette list has more values"
)

def plot_UMAP(umap_dir, mydict, labelname, patient_ID, class_real_names):
    print ("Plotting...")
    class_names = []    
    subset_features = []
    subset_labels = []
    samples_to_take = 600  #randomly select either 600 points or less
    try:
        for class_key in sorted(mydict.keys(), key=int):
            class_names.append(int(class_key))
            feature_lists = mydict[class_key]
            sample_size = min(samples_to_take, len(feature_lists))
            random_indices = np.random.choice(len(feature_lists), sample_size, replace=False)
            feature_lists = np.array(feature_lists)[random_indices].tolist()
            subset_features.extend(feature_lists)
            subset_labels.extend([class_key] * len(feature_lists))
        
        class_names_present = [class_real_names[i] for i in class_names]
        subset_features = np.array(subset_features)
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.11, n_components=2, metric='euclidean', random_state=42)
        umap_features = reducer.fit_transform(subset_features)

        plt.figure(figsize=(12, 8))
        custom_palette = ['#46cfcf', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        scatter = sns.scatterplot(
            x=umap_features[:, 0], 
            y=umap_features[:, 1], 
            hue=subset_labels, 
            palette=custom_palette,
            s=120,
            alpha=0.8, 
            linewidth=0.5, 
            edgecolor='w'  
        )
        
        plt.xlabel('UMAP Feature 1', fontsize=14)
        plt.ylabel('UMAP Feature 2', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        palette = sns.color_palette(custom_palette, len(class_names))
        
        legend_labels = [
            plt.Line2D(
                [0], [0], marker='o', color='w', label=classname,
                markersize=10, markerfacecolor=color
            ) for classname, color in zip(class_names_present, palette)
        ]

        plt.legend(
            handles=legend_labels, 
            loc='upper right',  
            fontsize=12, 
            markerscale=0.6,  
            bbox_to_anchor=(1.2, 1) 
        )    
        
        plt.legend(handles=legend_labels, loc='upper center', fontsize=18, markerscale = 1.0)        
        plt.savefig(f'{umap_dir}/{labelname}_{str(patient_ID)}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Error occurred during UMAP plotting:", e)