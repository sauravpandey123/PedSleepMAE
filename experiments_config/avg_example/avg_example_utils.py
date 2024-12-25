import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def get_needed_indices(labels, search_label, sleep_stage):
    if search_label == 'sleep_label':
        indices = (labels == sleep_stage).nonzero(as_tuple=True)[0]
        return indices
    else:
        if search_label == 'apnea_label':
            comparison_tensor = torch.tensor([1, 2, 3], device=labels.device)
            indices = torch.isin(labels, comparison_tensor).nonzero(as_tuple=True)[0]
            return indices
        elif search_label == 'hypop_label':
            comparison_tensor = torch.tensor([1, 2], device=labels.device)
            indices = torch.isin(labels, comparison_tensor).nonzero(as_tuple=True)[0]
        else:
            comparison_tensor = torch.tensor([1], device=labels.device)
            indices = torch.isin(labels, comparison_tensor).nonzero(as_tuple=True)[0]
        return indices
        

def plot_KNN_results(average, closest, furthest, patient_ID, output_dir, choice, case): 
    sns.set(style="whitegrid")
    filter_channels = [
        'EEG C3-M2', 'EEG O1-M2', 'EEG O2-M1', 'EEG CZ-O1', 'EEG C4-M1',
        'EEG F4-M1', 'EEG F3-M2', 'CAPNO', 'SPO2', 'RESP THORACIC',
        'RESP ABDOMINAL', 'SNORE', 'C-FLOW', 'EOG LOC-M2', 'EOG ROC-M1', 'EMG CHIN1-CHIN2'
    ]

    num_channels = 16
    index = 0  

    if choice == 'average':
        selected_data = average
        label = 'Average'
        color = 'blue'
    elif choice == 'closest':
        selected_data = closest
        label = 'Closest'
        color = 'green'
    elif choice == 'furthest':
        selected_data = furthest
        label = 'Furthest'
        color = 'red'
    else:
        raise ValueError("Invalid choice. Please set 'choice' to 'closest', 'furthest', or 'average'.")

    selected_data = selected_data.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(20, 1.3 * num_channels)) 
    axes = np.array(axes).flatten()

    for i, channel_name in enumerate(filter_channels):
        axes[i].set_ylabel(f'{channel_name}', fontsize=9, fontweight='bold')
        if selected_data.shape[0] == 1:
            data_to_plot = selected_data.squeeze(0)[i]
        else:
            data_to_plot = selected_data[i]
        axes[i].plot(data_to_plot, color=color, label=label, linewidth=1)
        if i < num_channels - 1:
            axes[i].set_xlabel('')
            axes[i].set_xticks([])

        axes[i].tick_params(axis='y', which='major', labelsize=10)

    axes[num_channels - 1].set_xlabel('Time', fontsize=12, fontweight='semibold')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  
    figname = f'{output_dir}/{case}_{choice}_example_ID_{patient_ID}.pdf'
    plt.savefig(figname, dpi=300, format='pdf')
    plt.close(fig) 

    print(f"Plot saved as {figname}")