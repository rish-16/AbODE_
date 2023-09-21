import MDAnalysis as mda
import os, re, subprocess
import pickle as pkl

import torch
import torch_geometric as tg

"""
- convert CREMP dataset samples (amino acid IDs, 3D coords) into pyg.Data instances
- compute angles and extra spatial features s_i = (r_i, alpha_i, gamma_i)
    - store in pyg.Data object
"""

CREMP_PATH = "/data/rishabh/pickle/CREMP/"

class CREMP_PeptODE_Dataset(tg.data.Dataset):
    def __init__(self, molecules):
        super().__init__()
        self.molecules = molecules
        self.transform()

    def transform(self):
        pass

    def __getitem__(self, idx):
        return self.molecules[idx]