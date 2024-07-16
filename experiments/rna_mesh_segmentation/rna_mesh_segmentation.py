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
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'xyz')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 1

# model 
input_features = args.input_features # one of ['xyz', 'hks']
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
pretrain_path = os.path.join(base_path, "pretrained_models/rna_mesh_seg_{}_4x128.pth".format(input_features))
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

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128, 
                                          N_block=4, 
                                        #   last_activation=torch.nn.functional.sigmoid,
                                          outputs_at='vertices', 
                                          dropout=True)


model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

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
    
    total_loss = 0
    total_num = 0
    for data in tqdm(train_loader):
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, targets = data

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
        targets = targets.to(device)
        
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

        preds = preds.squeeze()  # Remove the extra dimension if present
        targets = targets.float()  # Convert targets to float

        # Evaluate loss
        loss = criterion(preds, targets)
        loss.backward()
        
        # Track loss
        this_loss = loss.item() * targets.shape[0]
        this_num = targets.shape[0]
        total_loss += this_loss
        total_num += this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_mse = total_loss / total_num
    return train_mse


# Do an evaluation pass on the test dataset 
def test():
    model.eval()
    
    total_mse = 0
    total_num = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, targets = data

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
            targets = targets.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            preds = preds.squeeze()  # Remove the extra dimension if present
            targets = targets.float()  # Convert targets to float

            # Calculate MSE
            mse = criterion(preds, targets)
            total_mse += mse.item()
            total_num += targets.shape[0]

    test_mse = total_mse / total_num
    return test_mse 

if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_mse = train_epoch(epoch)
        test_mse = test()
        print("Epoch {} - Train MSE: {}  Test MSE: {}".format(epoch, train_mse, test_mse))

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)

# Test
test_mse = test()
print("Overall test MSE: {}".format(test_mse))
