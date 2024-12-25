import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import h5py
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from random import shuffle
from classifier_config.dataloader import get_dataloader
from classifier_config.MAE_model_downstream import PedSleepMAE
from classifier_config.classifier import Multiclass_Classification
from classifier_config.class_weights import *
from classifier_config.classifier_training_utils import *
from utils.misc import *
from utils.ignore_files import ignore_files

def main(args):

    checkpoint_file = args.checkpoint_file
    search_label = args.search_label
    emb_dim = args.emb_dim
    num_head = args.num_head
    num_layer = args.num_layer 
    seed = args.seed
    batch_size = args.batch_size
    mask_ratio = args.mask_ratio
    patch_size = args.patch_size
    base_learning_rate = args.base_learning_rate
    weight_decay = args.weight_decay
    total_epochs = args.total_epochs
    iterations_per_epoch = args.iterations_per_epoch
    val_interval = args.val_interval
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    dataset_directory_path = args.dataset_dir

    num_patches = int(3840/patch_size)
    num_channels = 16
    features_shape = num_channels * num_patches 

    search_label_full_name, num_classes = get_label_info(search_label)

    setup_seed(seed)
    list_of_hdf_files = [os.path.join(dataset_directory_path, x) for x in os.listdir(dataset_directory_path) if x.endswith('.hdf5')]
    shuffle(list_of_hdf_files) 

    #all files
    total_files = len(list_of_hdf_files)
    train_files = list_of_hdf_files[:int(0.8*total_files)]
    val_files = list_of_hdf_files[int(0.8*total_files):int(0.9*total_files)]
    test_files = list_of_hdf_files[int(0.9*total_files):]

    train_dataloader = get_dataloader(
        train_files,
        mode="train",
        label=search_label_full_name,
        iterations_per_epoch=iterations_per_epoch,
        batch_size=batch_size
    )

    val_dataloader = get_dataloader(val_files, mode = "validate", label = search_label_full_name) 
    test_dataloader = get_dataloader(test_files, mode = "test", label = search_label_full_name) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAE_model = PedSleepMAE(
        batch_size=batch_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        emb_dim=emb_dim,
        num_head=num_head,
        num_layer=num_layer
    ).to(device)

    checkpoint = torch.load(checkpoint_file, weights_only = True)
    MAE_model.load_state_dict(checkpoint['state_dict'])
    linear_model = Multiclass_Classification(features_shape, num_classes).to(device)
    class_weights = get_class_weights(search_label)
    weights = torch.tensor([class_weights[i] for i in sorted(class_weights)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.AdamW(linear_model.parameters(), weight_decay=weight_decay)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{search_label_full_name}_m{mask_ratio}p{patch_size}.txt")
    
    log_parameters_to_file(log_file, args)
    print(f"Training log and parameters will be saved to: {log_file}.")
    print(f"Saving the latest model at the end of every epoch.")
    print(f"Validating the model every {val_interval} epoch(s)...")
    print(f"Running on: {device}")

    best_val_loss = float('inf')

    for epoch in range(1, total_epochs + 1):
        train_loss = train(
            MAE_model=MAE_model,
            linear_model=linear_model,
            device=device,
            label_name=search_label_full_name,
            num_patches=num_patches,
            emb_dim=emb_dim,
            train_loader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion
        )


        #When it is time to validate:
        val_loss = None
        if epoch % val_interval == 0 or epoch == total_epochs: 
            val_metrics = validate(
                MAE_model=MAE_model,
                linear_model=linear_model,
                device=device,
                label_name=search_label_full_name,
                num_patches=num_patches,
                emb_dim=emb_dim,
                val_loader=val_dataloader,
                criterion=criterion
            )

            val_loss = val_metrics['loss']
            for key in val_metrics:
                val_detail = " | ".join([f"{key}: {val_metrics[key]:.4f}" for key in val_metrics])

        save = {
            'epoch': epoch,
            'state_dict': linear_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss,
        }

        train_save_file = f"{search_label_full_name}_m{mask_ratio}p{patch_size}_last_model"
        save_checkpoint(save, checkpoint_dir, train_save_file)

        train_msg = f"Epoch {epoch}: Training Loss: {train_loss:.4f}"
        val_msg = f"\nValidation Details: \n{val_detail}" if val_loss is not None else ""
        log_to_file(log_file, f"{train_msg} {val_msg}")  

        #Save model with best val loss and print message
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_metrics['loss']
            save['loss'] = best_val_loss 
            val_save_file = f"{search_label_full_name}_m{mask_ratio}p{patch_size}_best_loss_model"
            save_checkpoint(save, checkpoint_dir, val_save_file)
            log_to_file(log_file, f"Model with lowest validation loss {val_loss:.4f} updated and saved to {checkpoint_dir}/{val_save_file}\n*********")
        else:
            log_to_file(log_file, "*********") #only print this


    log_to_file(log_file, "Training Complete!\nTest Results:")

    #Can use the best threshold that came from last validated set
    best_threshold = val_metrics['best_threshold'] if search_label!='s' else None 
    
    #Can test using the last trained model
    test_metrics = test (
                MAE_model=MAE_model,
                linear_model=linear_model,
                device=device,
                label_name=search_label_full_name,
                num_patches=num_patches,
                emb_dim=emb_dim,
                test_loader=test_dataloader,
                criterion=criterion,
                best_threshold = best_threshold
            )

    for key, value in test_metrics.items():
        log_to_file(log_file, f"{key}:\n{value}")
    
    
if __name__ == "__main__":
    valid_labels = {
    "a": "apnea",
    "h": "hypopnea",
    "ah": "apnea and hypopnea combined",
    "e": "EEG arousal",
    "d": "oxygen desaturation",
    "s": "sleep scoring"
    }
    parser = argparse.ArgumentParser(description="Linear Classifier Training Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    
    parser.add_argument('search_label', type=str, choices=valid_labels.keys(),
                        help=('Label to search and train on. Options: '
                              'a (apnea), h (hypopnea), ah (apnea & hypopnea combined), '
                              'e (EEG arousal), d (oxygen desaturation), s (sleep scoring).'))
    
    
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint/m15p8_checkpoint.pt',
                        help=f'Path to the MAE checkpoint file.')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models',
                        help=f'Directory to save model checkpoints.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs.')
    
    parser.add_argument('--dataset_dir', type=str, default='hdf_batches',
                        help='Path to the dataset directory.')

    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding dimension for the transformer.')
    
    parser.add_argument('--num_head', type=int, default=4,
                        help='Number of attention heads.')
    
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of transformer layers.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training.')
    
    parser.add_argument('--mask_ratio', type=float, default=15,
                        help='Mask Ratio.')
    
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch Size')
    
    parser.add_argument('--base_learning_rate', type=float, default=1e-3,
                        help='Base learning rate for optimizer.')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization.')
    
    parser.add_argument('--total_epochs', type=int, default=50,
                        help='Total number of training epochs.')
    
    parser.add_argument('--iterations_per_epoch', type=int, default=2000,
                        help='Number of iterations per epoch.')
    
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Interval (in epochs) to run validation.')

    args = parser.parse_args()
    if args.search_label not in valid_labels:
        print(f"Error: Invalid search_label '{args.search_label}'. "
              f"Choose from: {', '.join(f'{k} ({v})' for k, v in valid_labels.items())}")
        sys.exit(1)
        
    main(args)