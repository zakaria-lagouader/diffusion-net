import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from rna_mesh_dataset import RNAMeshDataset 

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default='xyz')
args = parser.parse_args()

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_outputs = 1  # For regression, the number of output values per vertex

# model 
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 100
lr = 1e-4
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(base_path, "data/saved_models/sal_model-4.pth")
model_save_path = os.path.join(base_path, "data/saved_models/sal_model.pth")
dataset_path = os.path.join(base_path, "data/saliency_data")


# === Load datasets

# Load the test dataset
test_dataset = RNAMeshDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = RNAMeshDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# === Create the model

C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_outputs,
                                          C_width=128, 
                                          N_block=4, 
                                          last_activation=None,  # No softmax for regression
                                          outputs_at='vertices', 
                                          dropout=True)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()  # Use MSELoss for regression

def train_epoch(epoch):
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    for data in tqdm(train_loader):
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device, dtype=torch.float32)  # For regression, labels should be float
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        preds = preds.squeeze()

        # Evaluate loss
        loss = criterion(preds, labels)
        loss.backward()
        
        total_loss += loss.item()  # Accumulate loss
        total_num += 1  # Accumulate number of samples

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_num  # Compute average loss
    return avg_loss

# Do an evaluation pass on the test dataset 
def test():
    model.eval()
    
    total_loss = 0.0
    total_num = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device, dtype=torch.float32)  # For regression, labels should be float
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            preds = preds.squeeze()

            # Evaluate loss
            loss = criterion(preds, labels)
            
            total_loss += loss.item()  # Accumulate loss
            total_num += 1 # Accumulate number of samples

    avg_loss = total_loss / total_num  # Compute average loss
    return avg_loss

if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_loss = train_epoch(epoch)
        test_loss = test()
        print("Epoch {} - Train Loss: {:06.4f}  Test Loss: {:06.4f}".format(epoch, train_loss, test_loss))

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)

# Test
test_loss = test()
print("Overall test loss: {:06.4f}".format(test_loss))
