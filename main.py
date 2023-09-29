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
import prepare_cremp

"""
- dataloaders
- training loop
"""

# peptide_data = peptide_utils.get_graph_data_pyg(peptide_utils.process_data_mda("peptide_data/pdb_with_atom_connectivity_water/peptides/"))

cremp_data = torch.load("cremp_data_ca_only.pt")
print ("Loaded dataset ...")
n_instances = len(cremp_data)
train_size = int(0.8 * n_instances)
peptide_data_train, peptide_data_test = cremp_data[:train_size], cremp_data[train_size:][:70] # test size of 50 peptides
train_loader = tg.loader.DataLoader(peptide_data_train, batch_size=300)
test_loader = tg.loader.DataLoader(peptide_data_test, batch_size=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams',"scipy_solver","adaptive_heun"]

pos_emb_dim = 16
model = PeptODE_uncond(c_in=58+pos_emb_dim, n_layers=4)
model = model.to(device) # 37 features (28 for amino acids, 9 for spatial features)
optim = torch.optim.Adam(model.parameters())

t_begin = 0
t_end = 1
t_nsamples = 100
t_space = np.linspace(t_begin, t_end, t_nsamples)
t = torch.tensor(t_space).to(device)

EPOCHS = 1000
for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()
    for idx, batch_data in enumerate(train_loader):
        # batch_data is a pyg.data.DataBatch object
        batch_data = batch_data.to(device)
        optim.zero_grad()

        # print (batch_data.edge_index.shape, batch_data.edge_index.max(), batch_data.edge_index.min())
        params_list = [batch_data.edge_index, batch_data.a_index]
        model.update_param(params_list)
        # print (model.edge_index.shape, model.edge_index.max(), model.edge_index.min())

        options = {
            'dtype': torch.float64,
            # 'first_step': 1.0e-9,
            # 'grid_points': t,
        }

        # The ODE-function to solve the ODE-system
        # print ("ODE")
        pos_emb = peptide_utils.cyclic_positional_encoding(batch_data.a_index.view(-1), d=pos_emb_dim)
        batch_data.x = torch.cat([torch.rand_like(batch_data.x[:, :55]), batch_data.x[:, 55:], pos_emb], dim=1)
        batch_data.x = batch_data.x.to(device)
        y_pd = odeint(
            model, batch_data.x, t, 
            method="adaptive_heun", 
            rtol=5e-1, atol=5e-1,
            options=options
        )

        y_pd = y_pd[-1, :, :].reshape(-1, y_pd.size(-1)) # get last timestep z(T)
        loss = peptide_utils.loss_ca_only(y_pd, batch_data.y)
        loss.backward()
        optim.step()

        epoch_loss += loss.cpu().detach().item()

    print (f"epoch: {epoch} | train loss: {epoch_loss:.5f}")
    if epoch % 20 == 0:
        eval_metrics = peptide_utils.evaluate_model_ca_only(model, test_loader, device, odeint, time=t, pos_emb_dim=pos_emb_dim)
        pprint (eval_metrics, indent=2)

        torch.save(model, f"peptode_cremp_ckpt_caonly/peptode_cremp_model_epoch_{epoch}.pt")
        torch.save(eval_metrics, f"peptode_cremp_ckpt_caonly/peptode_cremp_metrics_epoch_{epoch}.pt")

torch.save(model, f"peptode_cremp_ckpt_caonly/peptode_cremp_model_epoch_final.pt")
