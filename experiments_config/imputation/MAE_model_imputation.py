#This model has been specially designed just for the decoder part to add mask tokens to the patches of the removed channels
#It does not take in patch indices and does not return a mask, only the final signals
#All the other functionality is the same as the other MAE models for downstream tasks

import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import random


class PatchProjection(torch.nn.Module):
    def __init__(self, patch_size, emb_dim):
        super().__init__()
        self.projection = torch.nn.Linear(patch_size, emb_dim)

    def forward(self, x):
        x = self.projection(x)
        return x
    
    
class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.samples = ratio

    def forward(self, patches : torch.Tensor):
        batch_size, num_channels, seq_length, emb_dim = patches.shape
        sample_seconds_to_keep = self.samples
        channels = num_channels
        unmasked_data = []
        patch_indices = []
        num_patches_to_mask = int(sample_seconds_to_keep/30 * seq_length)  #how many to mask
        num_patches_to_unmask = seq_length - num_patches_to_mask   #how many to unmask
        batch_unmasked_data = []  #collects data for the whole process
        batch_indices = []  
        for batch in range(batch_size):
            channel_data = []
            channel_indices = []
            for channel in range(channels):
                unmasked_patches_indices = random.sample(range(seq_length), num_patches_to_unmask)
                mask = torch.zeros(seq_length, dtype=torch.bool)
                mask[unmasked_patches_indices] = True
                channel_unmasked_data = patches[batch, channel, mask, :]          
                channel_data.append(channel_unmasked_data)
                channel_indices.append(unmasked_patches_indices) 
            channel_data = torch.stack(channel_data)
            batch_unmasked_data.append(channel_data)
            batch_indices.append(channel_indices)
            
        unmasked_data = torch.stack(batch_unmasked_data)
        unmasked_data = unmasked_data.reshape(batch_size, num_channels, num_patches_to_unmask, emb_dim)
        return unmasked_data, batch_indices


    
class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 batch_size, 
                 num_channels, 
                 patch_size, 
                 emb_dim, 
                 num_layer, 
                 num_head, 
                 mask_ratio = 0
                 ) -> None:
        super().__init__()
        
        self.patch_projection = PatchProjection(patch_size, emb_dim)
        self.channels = num_channels
        new_sequence_length = int(3840/patch_size) #3840 is the samples per channel     

        self.patch_size = patch_size 
        self.mask_ratio = 0 #set mask ratio to 0 since we need all embeddings
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.num_head = num_head
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1,1,1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.channels, new_sequence_length + 1, emb_dim)) 
        
        self.shuffle = PatchShuffle(self.mask_ratio)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim*num_channels, num_head) for _ in range(num_layer)]) 
        self.layer_norm = torch.nn.LayerNorm(emb_dim*num_channels)
        
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = img.unfold(2, self.patch_size, self.patch_size)  #Break image into patches and embed them, then add positional information
        patches = self.patch_projection(patches)
        batch_size, channels, seq_length, emb_dim = patches.shape
        excluded_pos_encoding = self.pos_embedding[:,:,1:,:] #remove the cls token
        
        patches = patches + excluded_pos_encoding # Add positional embeddings
        patches, patch_indices = self.shuffle(patches) 
        
        cls_token = self.cls_token + self.pos_embedding[:,:,:1,:]
        cls_token_expanded = cls_token.expand(patches.size(0), patches.size(1), -1, patches.size(3))
        
        patches = torch.cat([cls_token_expanded, patches], dim=2)   
        batch_size, channels, seq_length, emb_dim = patches.shape
        transformed_sequence = patches.view(batch_size, seq_length, channels*emb_dim)

        features = self.layer_norm(self.transformer(transformed_sequence))    
        features = features.view(batch_size, channels, seq_length, emb_dim)
        return features, patch_indices  
    

    
class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                    batch_size, 
                    num_channels, 
                    patch_size, 
                    emb_dim,
                    num_layer, 
                    num_head, 
                    mask_ratio
                 ) -> None:
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        self.channels = num_channels
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.emb_dim = emb_dim
        self.patch_size = patch_size 
        
        self.new_sequence_length = int(3840/patch_size)   
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 1, emb_dim)) 

        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.channels , self.new_sequence_length + 1, emb_dim)) 
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim*num_channels, num_head) for _ in range(num_layer)])
        output_dimension = self.patch_size
        self.head = torch.nn.Linear(emb_dim, output_dimension)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, remove_index, remove):        
        channels = self.channels
        cls_token = features[:,:,:1,:]
        features = features[:,:,1:,:]

        batch_size, _, seq_length, emb_dim = features.shape
        features_flat = features.view(batch_size, channels, seq_length*emb_dim)
        original_seq_length = self.new_sequence_length #when you define the final output, it should be of the same size as the original
        mask_token_expanded = self.mask_token.expand(batch_size, channels, original_seq_length, -1)
        
        full_sequence = torch.zeros(batch_size, channels, original_seq_length, emb_dim, device = features.device)  
                
        for batch_idx in range(batch_size):
            for channel in range(channels):
                channel_features = features[batch_idx, channel, :, :]
                full_sequence[batch_idx, channel, :, :] = channel_features
            if (remove):
                full_sequence[batch_idx, remove_index, :, :] = mask_token_expanded[batch_idx, remove_index]

        full_sequence = torch.cat([cls_token, full_sequence], dim=2) 
        batch_size, channels, seq_length, emb_dim = full_sequence.shape

        expanded_pos_encoding = self.pos_embedding
        full_sequence = full_sequence + expanded_pos_encoding
        
        batch_size, channels, seq_length, emb_dim = full_sequence.shape
        transformed_sequence = full_sequence.view(batch_size, seq_length, emb_dim*channels) 
        reconstructed_output = self.transformer(transformed_sequence)
        reconstructed_output = reconstructed_output.view(batch_size, channels, seq_length, emb_dim)

        reconstructed_output = reconstructed_output[:, :, 1:, :]
        batch_size, channels, seq_length, emb_dim = reconstructed_output.shape

        patches = self.head(reconstructed_output)
        final_output = patches.view(batch_size, channels, -1)
        
        return final_output
    


class PedSleepMAE(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 patch_size,
                 mask_ratio,
                 emb_dim = 64,
                 num_channels = 16,
                 num_head = 4,
                 num_layer = 3,
                 ) -> None:
        super().__init__()
        

        self.encoder = MAE_Encoder(batch_size, num_channels, patch_size, emb_dim, num_layer, num_head, mask_ratio)
        self.decoder = MAE_Decoder(batch_size, num_channels, patch_size, emb_dim, num_layer, num_head, mask_ratio)

    def forward(self, img):
        features, all_kept_channels = self.encoder(img)
        predicted_img, mask = self.decoder(features, all_kept_channels)
        return predicted_img, mask