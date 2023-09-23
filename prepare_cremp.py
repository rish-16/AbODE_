import torch
import torch_geometric as tg

import math, random, sys, argparse, os, json, csv, time
import numpy as np
import MDAnalysis as mda
from pprint import pprint
from tqdm import tqdm
import pickle as pkl
from rdkit import Chem

import peptide_utils
import featurise_cremp

CREMP_PATH = "/data/rishabh/pickle/"
pdb_sequences = os.listdir(CREMP_PATH) 

def get_cremp_data():
    data = []
    size_dist = []
    for pdb in pdb_sequences[:4]:
        fp = CREMP_PATH + pdb
        residues_in_mol = [aa.strip("[]") for aa in pdb.replace("Sar", "MeG").split(".")[:-1]] # ignore 'pickle' at the end
        size_dist.append(len(residues_in_mol))
        ohe_aa, all_conf_coords = featurise_cremp.featurize_macrocycle_atoms_from_file(path=fp, residues_in_mol=residues_in_mol) # return_mol = False
        ohe_aa = torch.tensor(ohe_aa)

        for conf_coords in all_conf_coords:
            coords_n, coords_ca, coords_c = conf_coords
            combined_coords = torch.from_numpy([np.concatenate([coords_n, coords_ca, coords_c], axis=1)]) # (N_res, 9)

            final_target_features = torch.cat([ohe_aa, combined_coords], dim=1)

            first_coord = all_coords[0].view(-1, 3, 3)
            edge_index = radius_graph(coords_ca, r=5, loop=False) # connect residues by C_alpha coordinates
            data.append(
                tg.data.Data(
                    x=ohe_aa,
                    y=,
                    edge_index=edge_index,
                    first_res=,

                )
            )

    return data, size_dist, 