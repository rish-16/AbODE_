import torch
import torch_geometric as tg
from torch_geometric.nn import knn_graph, radius_graph

import math, random, sys, argparse, os, json, csv, time
import numpy as np
import MDAnalysis as mda
from pprint import pprint
from tqdm import tqdm
import pickle as pkl
from rdkit import Chem

import peptide_utils
import featurise_cremp

NUM_UNIQUE_AAs = len(featurise_cremp.AMINO_ACID_RESNAMES)

def get_cremp_data(CREMP_PATH):
    final_data = []
    size_dist = []
    pdb_sequences = os.listdir(CREMP_PATH) 
    print ("number of sequences:", len(pdb_sequences))
    
    for pdb in pdb_sequences:
        try:
            fp = CREMP_PATH + pdb
            residues_in_mol = [aa.strip("[]") for aa in pdb.replace("Sar", "MeG").split(".")[:-1]] # ignore 'pickle' at the end
            size_dist.append(len(residues_in_mol))
            ohe_aa, all_conf_coords = featurise_cremp.featurize_macrocycle_atoms_from_file(path=fp, residues_in_mol=residues_in_mol) # return_mol = False
            ohe_aa = torch.tensor(ohe_aa)
            print (pdb, len(all_conf_coords))
            for conf_coords in all_conf_coords:
                coords_n, coords_ca, coords_c = conf_coords
                combined_coords = torch.from_numpy(np.concatenate([coords_n, coords_ca, coords_c], axis=1)) # (N_res, 9)

                converted_rag_polar_coords, input_rand_coords = peptide_utils.convert_coords_to_polar(coords_n, coords_ca, coords_c)
                final_target_features = torch.cat([ohe_aa, converted_rag_polar_coords], dim=1)

                input_peptide_labels = float(1/NUM_UNIQUE_AAs) * torch.ones(len(residues_in_mol), NUM_UNIQUE_AAs + 1) # uniform array
                input_peptide_labels = input_peptide_labels.view(-1, NUM_UNIQUE_AAs + 1)
                amino_index = torch.tensor([i for i in range(len(residues_in_mol))]).view(-1, 1).float()

                final_input_features = torch.cat([input_peptide_labels, input_rand_coords], dim=1)
                
                first_coord = combined_coords[0].view(-1, 3, 3)
                edge_index = radius_graph(torch.from_numpy(coords_ca), r=5, loop=False) # connect residues by C_alpha coordinates
                
                final_data.append(
                    tg.data.Data(
                        x=final_input_features,
                        y=final_target_features,
                        edge_index=edge_index,
                        first_res=first_coord,
                        a_index=amino_index.view(1,-1)
                    )
                )
        except Exception as e:
            print (pdb)
            print (e)

    return final_data, size_dist

# if __name__ == "__main__":
CREMP_PATH = "/data/rishabh/pickle/"
final_data, size_dist = get_cremp_data(CREMP_PATH)
print (len(final_data))
pprint (final_data[:10])
torch.save(final_data, "cremp_pyg_data_small.pt")