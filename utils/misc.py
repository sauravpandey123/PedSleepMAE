import random
import torch
import numpy as np




def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def log_parameters_to_file(log_file, args):
    with open(log_file, 'a') as f:
        f.write("Training Parameters:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")
        
def log_to_file(log_file, message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)