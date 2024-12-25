import os
import argparse
import torch
from random import shuffle

from pretrain_config.MAE_dataloader import get_dataloader
from pretrain_config.MAE_pretraining_utils import *
from pretrain_config.MAE_model import PedSleepMAE  
from utils.misc import *


def main(args):
    seed = args.seed
    batch_size = args.batch_size
    mask_ratio = args.mask_ratio
    patch_size = args.patch_size
    num_head = args.num_head
    num_layer = args.num_layer
    emb_dim = args.emb_dim
    base_learning_rate = args.base_learning_rate
    weight_decay = args.weight_decay
    total_epochs = args.total_epochs
    iterations_per_epoch = args.iterations_per_epoch
    checkpoint_dir = args.checkpoint_dir
    val_interval = args.val_interval
    log_dir = args.log_dir
    dataset_directory_path = args.dataset_dir
    
    setup_seed(seed)
    list_of_hdf_files = [os.path.join(dataset_directory_path, x) for x in os.listdir(dataset_directory_path) if x.endswith('.hdf5')]
    shuffle(list_of_hdf_files) 
    
    #all files
    total_files = len(list_of_hdf_files)
    train_files = list_of_hdf_files[:int(0.8*total_files)]
    val_files = list_of_hdf_files[int(0.8*total_files):]
    
    train_dataloader = get_dataloader(train_files, mode = "train", iterations_per_epoch = iterations_per_epoch, batch_size = batch_size) 
    val_dataloader = get_dataloader(val_files, mode = "validate") 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Define model architecture here
    model = PedSleepMAE(
        batch_size=batch_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        emb_dim=emb_dim,
        num_layer=num_layer,
        num_head=num_head
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print ("Total parameters in this model:", pytorch_total_params)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"log_MAE_m_{mask_ratio}_p_{patch_size}.txt")
        
    log_parameters_to_file(log_file, args)

    print(f"Training log and parameters will be saved to: {log_file}")
    print(f"Saving the latest model at the end of every epoch...")
    print(f"Validating the model every {val_interval} epoch(s)...")
    print(f"Running on: {device}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, total_epochs + 1):
        train_loss = train(model, device, train_dataloader, optimizer)

        val_loss = None
        if epoch % val_interval == 0 or epoch == total_epochs:  # Validate on last epoch as well
            val_loss = validate(model, device, val_dataloader)

        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss,
        }

        train_save_file = f"MAE_mask_{mask_ratio}_patch_{patch_size}_last_model"
        save_checkpoint(save, checkpoint_dir, train_save_file)
        
        train_msg = f"Epoch {epoch}: Training Loss: {train_loss:.4f}"
        val_msg = f"\nValidation Loss: {val_loss:.4f}" if val_loss is not None else ""
        log_to_file(log_file, f"{train_msg} {val_msg}")   
        
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save['loss'] = val_loss
            val_save_file = f"MAE_mask_{mask_ratio}_patch_{patch_size}_best_loss_model"
            save_checkpoint(save, checkpoint_dir, val_save_file)
            log_to_file(log_file, f"Model with lowest validation loss {val_loss:.4f} updated and saved to {checkpoint_dir}/{val_save_file}\n*********")
        else:
            log_to_file(log_file, "*********")
            
    log_to_file(log_file, "Training Complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for MAE with configurable parameters.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset_dir', type=str, help="Path to the grouped dataset directory.", default = 'hdf_batches')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--mask_ratio', type=int, default=15, help="Mask ratio for training.")
    parser.add_argument('--patch_size', type=int, default=8, help="Patch size for training.")
    parser.add_argument('--emb_dim', type=int, default=64, help="Dimensionality of the Embedding Space.")
    parser.add_argument('--num_layer', type=int, default=3, help="Layers of Vision Transformer(ViT) attention blocks.")
    parser.add_argument('--num_head', type=int, default=4, help="Number of Attention Heads")
    parser.add_argument('--base_learning_rate', type=float, default=1e-4, help="Base learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for the optimizer.")
    parser.add_argument('--total_epochs', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--iterations_per_epoch', type=int, default=4, help="Number of iterations per epoch.")
    parser.add_argument('--val_interval', type=int, default=4, help="Run validation every N epochs.")
    parser.add_argument('--checkpoint_dir', type=str, default="saved_models", help="Directory to save model checkpoints.")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory to save training logs.")
    
    args = parser.parse_args()
    main(args)