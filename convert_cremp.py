import torch
import torch_geometric as tg
from pprint import pprint

PATH = "cremp_pyg_data_small_coordsonly.pt"
dataset = torch.load(PATH)

new_dataset = []
for data in dataset:
    x = data.x
    aa_ids = x[:, :55]
    coords = x[:, 55:64]
    ca_coords = coords[:, 3:6]
    x_ = torch.cat([aa_ids, ca_coords], dim=1)
    data.x = x_

    first_res = data.first_res
    first_res_ = first_res[:, 1] # only Ca coordinates
    data.first_res = first_res_

    y = data.y
    y_aa_ids = y[:, :55]
    y_coords = y[:, 55:64]
    y_ca_coords = y_coords[:, 3:6]
    y_ = torch.cat([y_aa_ids, y_ca_coords], dim=1)
    data.y = y_

    new_dataset.append(data)

torch.save(new_dataset, "cremp_data_ca_only.pt")