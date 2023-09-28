import torch
import torch.nn as nn
import torch_geometric as tg
import torchdiffeq as tde
from torchdiffeq import odeint as odeint

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

model = torch.load("peptode_ckpt/peptode_model_epoch_final.pt")

t_begin = 0
t_end = 1
t_nsamples = 100
t_space = np.linspace(t_begin, t_end, t_nsamples)
t = torch.tensor(t_space).to(device)

with torch.no_grad():
    model.eval()
    for idx, batch_data in enumerate(test_loader):
        # batch_data is a pyg.data.DataBatch object
        batch_data = batch_data.to(device)
        optim.zero_grad()

        # print (batch_data.edge_index.shape, batch_data.edge_index.max(), batch_data.edge_index.min())
        params_list = [batch_data.edge_index, batch_data.a_index]
        model.update_param(params_list)
        # print (model.edge_index.shape, model.edge_index.max(), model.edge_index.min())

        eval_metrics = peptide_utils.evaluate_model(model, test_loader, device, odeint, time=t)
        pprint (eval_metrics, indent=2)