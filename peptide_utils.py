import torch
import torch.nn as nn
import torch.nn.functional as F

import math, random, sys, argparse, os, json, csv
import numpy as np
import MDAnalysis as mda
from pprint import pprint
from tqdm import tqdm

from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.data import Data, DataLoader

import astropy
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian

from rmsd import *
from scipy.stats import vonmises
from scipy.special import softmax

def loss_function_polar(y_pred, y_true):
    pred_labels = y_pred[:,:28].view(-1, 28)
    truth_labels = y_true[:,:28].view(-1, 28)
    
    celoss = nn.CrossEntropyLoss()
    loss_ce = celoss(pred_labels,truth_labels)
    
    pred_r = y_pred[:,28].view(-1,1)
    true_r = y_true[:,28].view(-1,1)
    
    r_loss = nn.SmoothL1Loss(reduction='mean')
    loss_val = r_loss(pred_r,true_r)
    
    pred_angle = y_pred[:,29:31].view(-1,2)
    true_angle = y_true[:,29:31].view(-1,2)
    
    diff_angle = pred_angle - true_angle
    loss_per_angle = torch.mean(1 - torch.square(torch.cos(diff_angle)),0).view(-1,2)
    total_angle_loss  = loss_per_angle.sum(dim=1)
    
    total_loss = loss_ce + 0.8*(loss_val + total_angle_loss)
    
    return total_loss
    

def process_data_mda(path):
    directory = os.listdir(path)
    peptide_data = []

    atoms = ["N", "H", "O", "C", "S"]
    atom_type_map = {sym : i for i, sym in enumerate(atoms)}
    possible_residues = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'ALA', 'LEU', 'MLE', 'NAL', 'NLE', 'PHE', 'PRO']
    res_mapper = {sym : i for i, sym in enumerate(possible_residues)}
        
    def onehot_encoder(sym, mapper):
        vec = [0 for _ in range(len(mapper) + 1)] # extra +1 for unknown element
        if sym not in mapper:
            vec[-1] = 1 # set final component as 1 if unknown element
        else:
            idx = mapper[sym]
            vec[idx] = 1
        return vec

    for file in directory:
        fp = path + file
        uni = mda.Universe(fp, format="PDB")
        residues = list(uni.residues)
        residues = [res.resname for res in residues]
        
        backbone_atoms = uni.select_atoms("(name CA or name N or name C)")
        coords = backbone_atoms.positions
        coords = torch.from_numpy(coords) # (3N, 3)

        shape_converter = lambda x,i : x[i:][::3]

        coords_concat_n = shape_converter(coords, 0)
        coords_concat_ca = shape_converter(coords, 1)
        coords_concat_c = shape_converter(coords, 2)

        # group the (N, Ca, C) coordinates to create a (N, 1, 3) tensor
        coords_final = torch.cat([coords_concat_n, coords_concat_ca, coords_concat_c], dim=1)

        node_features = []
        for res in residues:
            node_features.append(onehot_encoder(res, res_mapper))

        node_features = torch.tensor(node_features)

        edge_index = radius_graph(coords_concat_ca, r=5, loop=False)
        
        # data = Data(x=coords_final, h=node_features, y=target_features, edge_index=edge_index)
        data = {
            "x": coords_final,
            "aa": node_features,
            "edge_index": edge_index,
            "residues": residues
        }
        peptide_data.append(data)

    return peptide_data

def get_graph_data_pyg(mda_data):
    all_data = []
    for record in mda_data:
        N_res = len(record['residues'])
        all_coords = record['x'].reshape(-1, 9)

        peptide_coords_forward_rolled = torch.roll(all_coords, 1, 0)
        peptide_diff_forward = all_coords - peptide_coords_forward_rolled

        peptide_coords_backward_rolled = torch.roll(all_coords, -1, 0)
        peptide_diff_backward = peptide_coords_backward_rolled - all_coords
        
        # peptide_diff_forward = peptide_diff_forward[1:-1]
        # peptide_diff_backward = peptide_diff_backward[1:-1]
        r_norm = torch.norm(peptide_diff_backward.view(-1, 3, 3), dim=2).view(-1, 3, 1) # r_i
        
        mid_angle = torch.acos(F.cosine_similarity(peptide_diff_forward.view(-1, 3, 3), peptide_diff_backward.view(-1, 3, 3),dim=2)).view(-1, 3, 1) # alpha_i
        cross_vector = torch.cross(peptide_diff_forward.view(-1, 3, 3), peptide_diff_backward.view(-1, 3, 3), dim=2).view(-1, 3, 3) # gamma_i
        normal_angle = torch.acos(F.cosine_similarity(cross_vector, peptide_diff_backward.view(-1, 3, 3), dim=2)).view(-1, 3, 1) # n_i

        # s_i = <r_i, a_i, g_i>
        peptide_pos_features = torch.cat((r_norm, mid_angle, normal_angle), dim=2).view(-1, 9) # s_i

        # edge_s = []
        # edge_f = []
        # #edges_ab = radius_graph(torch.tensor(C_alpha_ab),r=10,loop=True)
        # for idx_start in range(len(peptide_pos_features)):
        #     for idx_end in range(len(peptide_pos_features)):
        #         if idx_start != idx_end:
        #             edge_s.append(idx_start)
        #             edge_f.append(idx_end)
        
        # edges_ab = torch.tensor([edge_s,edge_f])

        label_features = record['aa']
        assert label_features.size(0) == peptide_pos_features.size(0), f"size mismatch {label_features.shape} AND {peptide_pos_features.shape}"
        final_target_features = torch.cat([label_features, peptide_pos_features], dim=1)

        input_peptide_labels = float(1 / 28) * torch.ones(size=(N_res, 28))
        input_peptide_labels = input_peptide_labels.view(-1, 28)
        amino_index = torch.tensor([i for i in range(N_res)]).view(-1, 1).float()
        temp_coords = peptide_pos_features.view(-1, 3, 3)

        input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(), temp_coords[-1].numpy(), N_res)).view(-1, 9)
        final_input_features = torch.cat([input_peptide_labels, input_ab_coords], dim=1) # z_i(t) = [a_i(t), s_i(t)]
        
        d = Data(x=final_input_features, y=final_target_features, edge_index=record['edge_index'], a_index=amino_index.view(1,-1), )
        all_data.append(d)

    return all_data

def get_angles(coordinates, atom_ids):
    pass

def process_data_rdk(rdkit_mols):
    """
    - extract N, Ca, C backbone coordinates
    - extra residue identities
    """

    pass

if __name__ == "__main__":
    peptide_data = get_graph_data_pyg(process_data_mda("peptide_data/pdb_with_atom_connectivity_water/peptides/"))
    pprint (peptide_data[:5])