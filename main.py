import torch
import torch.nn as nn
import torch_geometric as tg
import torchdiffeq as tde
from torchdiffeq import odeint as odeint

import numpy as np
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
peptide_data_train, peptide_data_test = peptide_data[:train_size], peptide_data[train_size:]
train_loader = tg.loader.DataLoader(peptide_data_train, batch_size=64)
test_loader = tg.loader.DataLoader(peptide_data_test, batch_size=64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"scipy_solver","adaptive_heun"]

model = PeptODE_uncond(c_in=37, n_layers=4)
model = model.to(device) # 37 features (28 for amino acids, 9 for spatial features)
optim = torch.optim.Adam(model.parameters())

t_begin = 0
t_end = 1
t_nsamples = 200
t_space = np.linspace(t_begin, t_end, t_nsamples)

EPOCHS = 1000
for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()
    for idx, batch_data in enumerate(train_loader):
        # batch_data is a pyg.data.DataBatch object
        optim.zero_grad()
        t = torch.tensor(t_space).to(device)

        params_list = [batch_data.edge_index, batch_data.a_index]
        model.update_param(params_list)

        options = {
            'dtype': torch.float64,
            # 'first_step': 1.0e-9,
            # 'grid_points': t,
        }

        # The ODE-function to solve the ODE-system
        print ("ODE")
        y_pd = odeint(
            model, batch_data.x, t, 
            method="adaptive_heun", 
            rtol=5e-1, atol=5e-1,
            options=options
        )

        break
    break
    #     # pred = model(batch_data)
    #     # loss = utils.loss_function_vm_with_side_chains_v2(y_pd, batch_data.y)
    #     loss = peptide_utils.my_custom_peptide_loss(y_pd, batch.y)
    #     loss.backward()
    #     optim.step()

    #     epoch_loss += loss.cpu().detach.item()

    # # with torch.no_grad():
    #     # model.eval()
    #     # get metrics

    # print (f"epoch: {epoch} | train loss: {epoch_loss:.5f}")


