import torch
import torch.nn as nn

import torchdiffeq as tde
from torchdiffeq import odeint as odeint

import torch_geometric as tg
from torch_geometric.data import Data, Batch

import numpy as np
from pprint import pprint
from peptide_model import PeptODE_uncond

import utils
import peptide_utils
import prepare_cremp

"""
- dataloaders
- training loop
"""

# peptide_data = peptide_utils.get_graph_data_pyg(peptide_utils.process_data_mda("peptide_data/pdb_with_atom_connectivity_water/peptides/"))
cremp_data = torch.load("cremp_pyg_data_small_coordsonly.pt")
n_instances = len(cremp_data)
train_size = int(0.8 * n_instances)
peptide_data_test = cremp_data[train_size:]
peptide_data_train = cremp_data[:train_size]

avg_first_coordinates = np.mean([data.y[0, :][55:64].numpy() for data in peptide_data_train], axis=0) # get first residue from every instance
avg_last_coordinates = np.mean([data.y[-1, :][55:64].numpy() for data in peptide_data_train], axis=0) # get last residue from every instance
test_loader = tg.loader.DataLoader(peptide_data_test, batch_size=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"scipy_solver","adaptive_heun"]

model = PeptODE_uncond(c_in=37, n_layers=4)
model = model.to(device) # 37 features (28 for amino acids, 9 for spatial features)
optim = torch.optim.Adam(model.parameters())

# model = torch.load("peptode_cremp_ckpt/peptode_cremp_model_epoch_140.pt").to(device)
model = torch.load("peptode_cremp_ckpt_lossv3/peptode_cremp_model_epoch_40.pt").to(device)

t_begin = 0
t_end = 1
t_nsamples = 100
t_space = np.linspace(t_begin, t_end, t_nsamples)
t = torch.tensor(t_space).to(device)

def generate(model, N_peptides, N_residues, pos_emb_dim):
    batch = []
    for n in range(N_peptides):
        
        n_res = N_residues[n]
        peptide_pos_features = torch.randn(n_res, 9)
        # input_peptide_labels = float(1 / 28) * torch.ones(size=(n_res, 28))
        input_peptide_labels = torch.rand(size=(n_res, 55))
        input_peptide_labels = input_peptide_labels.view(-1, 55)
        amino_index = torch.tensor([i for i in range(n_res)]).view(-1, 1).float()
        
        input_ab_coords = torch.from_numpy(np.linspace(avg_first_coordinates, avg_last_coordinates, n_res)).view(-1, 9)
        final_input_features = torch.cat([input_peptide_labels, input_ab_coords], dim=1) # z_i(t) = [a_i(t), s_i(t)]

        edges = []
        for i in range(n_res):
            for j in range(n_res):
                if i != j:
                    edges.append([i, j])
        n_edges = len(edges)
        edges = torch.tensor(edges).long().view(2, n_edges)

        aa_idx = torch.tensor([i for i in range(n_res)]).view(-1, 1).float()

        data = Data(x=final_input_features, edge_index=edges, a_index=aa_idx.view(1,-1))
        batch.append(data)

    for ii, data in enumerate(batch): # generate 1 molecule at a time (TODO: batching is buggy)
        batch_data = Batch.from_data_list([data])
        params = [batch_data.edge_index, batch_data.a_index]
        model.update_param(params)
        pos_emb = peptide_utils.cyclic_positional_encoding(batch.a_index.view(-1), d=pos_emb_dim)
        x = torch.cat([batch.x, pos_emb], dim=1)
        batch_data.x = x
        batch_data = batch_data.to(device)

        options = {
            'dtype': torch.float64,
            # 'first_step': 1.0e-9,
            # 'grid_points': t,
        }
        
        y_pd = odeint(
            model, batch_data.x, t, 
            method="rk4", 
            rtol=5e-1, atol=5e-1,
            options=options
        )

        y_pd = y_pd[-1].cpu().detach() # get final timestep z(T)
        amino_acids_ids = torch.softmax(y_pd[:, :55], 1)
        amino_acids_ids = amino_acids_ids.argmax(dim=1)
        polar_coords = y_pd[:, 55:58]

        peptide_utils.convert_to_mda_writer(amino_acids_ids, polar_coords, pep_idx=ii, save_dir="generated_cremp_peptides_ca_only/")

size_dist = np.load("cremp_size_dist.npy")
N_peptides = 10
sampled_sizes = np.random.choice(a=size_dist, size=(N_peptides,))
sampled_sizes = sampled_sizes.reshape(-1).tolist()
generate(model, N_peptides, sampled_sizes, pos_emb_dim=16)

