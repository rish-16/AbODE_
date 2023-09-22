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

"""
- dataloaders
- training loop
"""

peptide_data = peptide_utils.get_graph_data_pyg(peptide_utils.process_data_mda("peptide_data/pdb_with_atom_connectivity_water/peptides/"))
n_instances = len(peptide_data)
train_size = int(0.8 * n_instances)
peptide_data_test = peptide_data[train_size:]
test_loader = tg.loader.DataLoader(peptide_data_test, batch_size=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"scipy_solver","adaptive_heun"]

model = PeptODE_uncond(c_in=37, n_layers=4)
model = model.to(device) # 37 features (28 for amino acids, 9 for spatial features)
optim = torch.optim.Adam(model.parameters())

model = torch.load("peptode_ckpt/peptode_model_epoch_final.pt").to(device)

t_begin = 0
t_end = 1
t_nsamples = 200
t_space = np.linspace(t_begin, t_end, t_nsamples)
t = torch.tensor(t_space).to(device)

def generate(model, N_peptides, N_residues):
    batch = []
    for n in range(N_peptides):
        n_res = N_residues[n]
        input_peptide_labels = float(1 / 28) * torch.ones(size=(N_res, 28))
        input_peptide_labels = input_peptide_labels.view(-1, 28)
        amino_index = torch.tensor([i for i in range(N_res)]).view(-1, 1).float()
        temp_coords = peptide_pos_features.view(-1, 3, 3)

        input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(), temp_coords[-1].numpy(), N_res)).view(-1, 9)
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

    batch = Batch.from_data_list(batch)
    params = [batch.edge_index, batch.a_index]
    model.update_param(params)
    x = batch.x

    options = {
        'dtype': torch.float64,
        # 'first_step': 1.0e-9,
        # 'grid_points': t,
    }
    
    y_pd = odeint(
        model, x, t, 
        method="adaptive_heun", 
        rtol=5e-1, atol=5e-1,
        options=options
    )

    y_pd = y_pd[-1] # get final timestep z(T)
    amino_acids_ids = torch.softmax(y_pd[:, :28], 1).argmax()
    polar_coords = y_pd[:, 28:37]

    print (amino_acids.shape, polar_coords.shape)

    peptide_utils.convert_to_mda_writer(amino_acids_ids, polar_coords, N_residues)

generate(model, 1, 6)