import torch
import torch_geometric as tg
from pprint import pprint

PATH = "cremp_pyg_data_small_coordsonly.pt"
dataset = torch.load(PATH)

new_dataset = []
for data in dataset[:5]:
    x = data.x
    aa_ids = x[:, :55]
    coords = x[:, 55:64]
    ca_coords = coords[:, 3:6]
    x_ = torch.cat([aa_ids, ca_coords], dim=1)
    data.x = x_
    new_dataset.append(data)

pprint (new_dataset)

# torch.save(new_dataset, "cremp_data_ca_only.pt")