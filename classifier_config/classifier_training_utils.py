import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix


pool = torch.nn.AdaptiveAvgPool1d(1)

def process_apnea_hypopnea_cases(labels, searchlabel):
    if (searchlabel == 'apnea_labels' or searchlabel == 'apnea_hypopnea_labels'):
        all_possible_labels = [1,2,3] #[1 = central apnea, 2 = obstructive apnea, 3 = mixed apnea]
        for label in all_possible_labels:
            labels[labels == label] = 1        
    elif searchlabel == 'hypop_labels':
        all_possible_labels = [1,2] #[1 = obstructive hypopnea, 2 = hypopnea]
        for label in all_possible_labels:
            labels[labels == label] = 1
    return labels


def get_label_info(labels):
    labels = labels.lower()
    if (labels == "a"):
        return "apnea_labels",2
    elif (labels == "h"):
        return "hypop_labels",2
    elif (labels == "ah"): 
        return "apnea_hypopnea_labels",2
    elif (labels == "e"):
        return "eeg_labels",2
    elif (labels == "d"):
        return "desat_labels",2
    elif (labels == "s"):
        return "sleep_labels",5
    

    
def train(MAE_model, linear_model, device, label_name, num_patches, emb_dim, train_loader, optimizer, criterion):
    MAE_model.eval()
    linear_model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_tensor = batch["x"].squeeze().float().to(device) 
        batch_size =  input_tensor.shape[0]
        output_tensor = batch["y"].squeeze().to(device)
        with torch.no_grad():
            encoded_features, _ = MAE_model.encoder(input_tensor) 
            encoded_features = encoded_features[:,:,1:,:]
            encoded_reshaped = encoded_features.reshape(-1, num_patches, emb_dim)
            pooled = pool(encoded_reshaped)
            pooled = pooled.squeeze(-1)
            flattened_features = pooled.reshape(batch_size, -1)
            labels = output_tensor
            if label_name in ['apnea_labels', 'hypop_labels', 'apnea_hypopnea_labels']:
                labels = process_apnea_hypopnea_cases(output_tensor, label_name)
        optimizer.zero_grad()
        outputs = linear_model(flattened_features)
        loss = criterion(outputs, labels.long())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    return avg_loss



def validate(MAE_model, linear_model, device, label_name, num_patches, emb_dim, val_loader, criterion):
    MAE_model.eval()
    linear_model.eval()
    running_loss = 0.0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_tensor = batch["x"].squeeze().float().to(device) 
            batch_size = input_tensor.shape[0]
            output_tensor = batch["y"].squeeze().to(device)
            
            # Feature extraction
            encoded_features, _ = MAE_model.encoder(input_tensor)
            encoded_features = encoded_features[:, :, 1:, :]
            encoded_reshaped = encoded_features.reshape(-1, num_patches, emb_dim)
            pooled = pool(encoded_reshaped)
            pooled = pooled.squeeze(-1)
            flattened_features = pooled.reshape(batch_size, -1)

            labels = output_tensor
            if label_name in ['apnea_labels', 'hypop_labels', 'apnea_hypopnea_labels']:
                labels = process_apnea_hypopnea_cases(output_tensor, label_name)

            outputs = linear_model(flattened_features)

            if outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  
                preds = (probs > 0.5).astype(int)  
            else: 
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    avg_loss = running_loss / len(val_loader)

    # Binary classification metrics
    if outputs.shape[1] == 2:
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_index = f1_scores.argmax()
        best_threshold = thresholds[best_index] if len(thresholds) > 0 else 0.5
        best_f1 = f1_scores[best_index]
        preds = (all_probs > best_threshold).astype(int)

        auc = roc_auc_score(all_labels, all_probs)
        return {
            "loss": avg_loss,
            "auc": auc,
            "best_f1": best_f1,
            "best_threshold": best_threshold
        }

    # Multi-class classification metrics
    else:
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        # Weighted F1-score
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        try:
            weighted_auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
        except ValueError:
            weighted_auc = 0.0

        return {
            "loss": avg_loss,
            "weighted_f1": weighted_f1,
            "weighted_auc": weighted_auc
        }

    
    
    
def test(MAE_model, linear_model, device, label_name, num_patches, emb_dim, test_loader, criterion, best_threshold=None):
    MAE_model.eval()
    linear_model.eval()
    running_loss = 0.0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_tensor = batch["x"].squeeze().float().to(device)
            batch_size = input_tensor.shape[0]
            output_tensor = batch["y"].squeeze().to(device)
            
            encoded_features, _ = MAE_model.encoder(input_tensor)
            encoded_features = encoded_features[:, :, 1:, :]
            encoded_reshaped = encoded_features.reshape(-1, num_patches, emb_dim)
            pooled = pool(encoded_reshaped)
            pooled = pooled.squeeze(-1)
            flattened_features = pooled.reshape(batch_size, -1)

            labels = output_tensor
            if label_name in ['apnea_labels', 'hypop_labels', 'apnea_hypopnea_labels']:
                labels = process_apnea_hypopnea_cases(output_tensor, label_name)

            outputs = linear_model(flattened_features)

            if outputs.shape[1] == 2:  # Binary classification (2 neurons)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Positive class probabilities
                preds = (probs > (best_threshold if best_threshold is not None else 0.5)).astype(int)
            else:  # Multi-class classification
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs if outputs.shape[1] == 2 else probs)
            all_preds.extend(preds)
    
    avg_loss = running_loss / len(test_loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    if outputs.shape[1] == 2:  # Binary classification
        accuracy = round(accuracy_score(all_labels, all_preds), 4)
        f1 = round(f1_score(all_labels, all_preds), 4)
        conf_matrix = np.round(confusion_matrix(all_labels, all_preds, normalize='true'), 4)
        auc = round(roc_auc_score(all_labels, all_probs), 4)

        return {
            "loss": round(avg_loss, 4),
            "accuracy": accuracy,
            "f1_score ": f1,
            "confusion_matrix": conf_matrix,
            "auroc": auc
        }

    else:  # Handle sleep scoring (multi-class)
        accuracy = round(accuracy_score(all_labels, all_preds), 4)
        weighted_f1 = round(f1_score(all_labels, all_preds, average='weighted'), 4)
        conf_matrix = np.round(confusion_matrix(all_labels, all_preds, normalize='true'), 4)
        try:
            weighted_auc = round(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted'), 4)
        except ValueError:
            weighted_auc = 0.0  # If AUROC cannot be computed

        return {
            "loss": round(avg_loss, 4),
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "confusion_matrix": conf_matrix,
            "weighted_auroc": weighted_auc
        }

        

def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    f_path = os.path.join(checkpoint_dir, f"{filename}.pt")
    torch.save(state, f_path)