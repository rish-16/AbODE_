import torch
import torch.nn as nn
import torch.nn.functional as F

import math, random, sys, argparse, os, json, csv, time
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

        shape_converter = lambda x,i : x[i:][::3] # 3 because we only care about grouping N, Ca, C separately

        coords_concat_n = shape_converter(coords, 0)
        coords_concat_ca = shape_converter(coords, 1)
        coords_concat_c = shape_converter(coords, 2)

        # group the (N, Ca, C) coordinates to create a (N, 1, 3) tensor
        coords_final = torch.cat([coords_concat_n, coords_concat_ca, coords_concat_c], dim=1)

        node_features = []
        for res in residues:
            node_features.append(onehot_encoder(res, res_mapper))

        node_features = torch.tensor(node_features)

        edge_index = radius_graph(coords_concat_ca, r=5, loop=False) # connect residues by C_alpha coordinates
        
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
    for ii, record in enumerate(mda_data):
        N_res = len(record['residues']) 
        all_coords = record['x'].reshape(-1, 9)

        peptide_coords_forward_rolled = torch.roll(all_coords, 1, 0)
        peptide_diff_forward = all_coords - peptide_coords_forward_rolled

        peptide_coords_backward_rolled = torch.roll(all_coords, -1, 0)
        peptide_diff_backward = peptide_coords_backward_rolled - all_coords

        first_coord = all_coords[0].view(-1, 3, 3)
        
        # peptide_diff_forward = peptide_diff_forward[1:-1]
        # peptide_diff_backward = peptide_diff_backward[1:-1]
        r_norm = torch.norm(peptide_diff_backward.view(-1, 3, 3), dim=2).view(-1, 3, 1) # r_i
        
        mid_angle = torch.acos(F.cosine_similarity(peptide_diff_forward.view(-1, 3, 3), peptide_diff_backward.view(-1, 3, 3),dim=2)).view(-1, 3, 1) # alpha_i
        cross_vector = torch.cross(peptide_diff_forward.view(-1, 3, 3), peptide_diff_backward.view(-1, 3, 3), dim=2).view(-1, 3, 3) # gamma_i
        normal_angle = torch.acos(F.cosine_similarity(cross_vector, peptide_diff_backward.view(-1, 3, 3), dim=2)).view(-1, 3, 1) # n_i

        # s_i = <r_i, a_i, g_i>
        peptide_pos_features = torch.cat((r_norm, mid_angle, normal_angle), dim=2).view(-1, 9) # s_i

        # if ii == 0: 
            # print (peptide_pos_features) 

        label_features = record['aa']
        assert label_features.size(0) == peptide_pos_features.size(0), f"size mismatch {label_features.shape} AND {peptide_pos_features.shape}"
        final_target_features = torch.cat([label_features, peptide_pos_features], dim=1)

        input_peptide_labels = float(1 / 28) * torch.ones(size=(N_res, 28))
        input_peptide_labels = input_peptide_labels.view(-1, 28)
        amino_index = torch.tensor([i for i in range(N_res)]).view(-1, 1).float()
        temp_coords = peptide_pos_features.view(-1, 3, 3)

        input_ab_coords = torch.from_numpy(np.linspace(temp_coords[0].numpy(), temp_coords[-1].numpy(), N_res)).view(-1, 9)
        final_input_features = torch.cat([input_peptide_labels, input_ab_coords], dim=1) # z_i(t) = [a_i(t), s_i(t)]
        
        d = Data(x=final_input_features, y=final_target_features, edge_index=record['edge_index'], a_index=amino_index.view(1,-1), first_residue=first_coord)
        all_data.append(d)

    return all_data

def process_data_rdk(rdkit_mols):
    """
    - extract N, Ca, C backbone coordinates
    - extra residue identities
    """

    pass

def _transform_to_cart(coords_r, coords_theta, coords_phi):
    x_coord_n_true  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.cos(coords_phi[:,0]).view(-1,1)
    y_coord_n_true  = coords_r[:,0].view(-1,1)*torch.sin(coords_theta[:,0]).view(-1,1)*torch.sin(coords_phi[:,0]).view(-1,1)
    z_coord_n_true  = coords_r[:,0].view(-1,1)*torch.cos(coords_theta[:,0]).view(-1,1)
        
    x_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.cos(coords_phi[:,1]).view(-1,1)
    y_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.sin(coords_theta[:,1]).view(-1,1)*torch.sin(coords_phi[:,1]).view(-1,1)
    z_coord_ca_true   = coords_r[:,1].view(-1,1)*torch.cos(coords_theta[:,1]).view(-1,1).view(-1,1)
        
    x_coord_c_true   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.cos(coords_phi[:,2]).view(-1,1)
    y_coord_c_true   = coords_r[:,2].view(-1,1)*torch.sin(coords_theta[:,2]).view(-1,1)*torch.sin(coords_phi[:,2]).view(-1,1)
    z_coord_c_true   = coords_r[:,2].view(-1,1)*torch.cos(coords_theta[:,2]).view(-1,1)
    
    Cart = torch.cat([x_coord_n_true, y_coord_n_true, z_coord_n_true, x_coord_ca_true, y_coord_ca_true, z_coord_ca_true, x_coord_c_true, y_coord_c_true, z_coord_c_true], dim=1).view(-1, 9)
    
    return Cart

def _get_cartesian(pred_polar_coord, truth_polar_coord):
    pred_r = pred_polar_coord[:, [0,3,6]]
    pred_theta = pred_polar_coord[:, [1,4,7]]
    pred_phi = 3.14/2 - pred_polar_coord[:, [2,5,8]]
        
    coords_r =  truth_polar_coord[:, [0,3,6]]
    coords_theta = truth_polar_coord[:, [1,4,7]]
    coords_phi = 3.14/2 -truth_polar_coord[:, [2,5,8]]
    
    Cart_true = _transform_to_cart(coords_r,coords_theta,coords_phi)
    Cart_pred = _transform_to_cart(pred_r,pred_theta,pred_phi)
    
    return Cart_true, Cart_pred

def evaluate_model(model, loader, device, odeint, time):
    model.eval()
    
    perplexity = []
    calpha_rmsd = []
    rmsd_pred = []
    RMSD_test_n = []
    RMSD_test_ca = []
    RMSD_test_ca_cart = []
    RMSD_test_c = []    

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        params = [batch.edge_index, batch.a_index]
        model.update_param(params)
        x = batch.x

        options = {
            'dtype': torch.float64,
            # 'first_step': 1.0e-9,
            # 'grid_points': t,
        }
        
        y_pd = odeint(
            model, x, time, 
            method="adaptive_heun", 
            rtol=5e-1, atol=5e-1,
            options=options
        )

        y_pd = y_pd[-1] # get final timestep z(T)
        y_truth = batch.y
        
        pred_labels = y_pd[:, :28].view(-1, 28)
        truth_labels = y_truth[:, :28].view(-1, 28)

        celoss = nn.CrossEntropyLoss()
        loss_ce = celoss(pred_labels, truth_labels)
        ppl = torch.exp(loss_ce)

        first_residue = batch.first_residue

        pred_polar_coord = y_pd[:,28:37].cpu().detach().numpy().reshape(-1, 3, 3)
        truth_polar_coord = y_truth[:,28:37].cpu().detach().numpy().reshape(-1, 3, 3)
        first_residue_coord = first_residue[:, 1, :].cpu().detach().numpy().reshape(-1, 3)

        rmsd_N = kabsch_rmsd(pred_polar_coord[:][:,0][:], truth_polar_coord[:][:,0][:])
        rmsd_Ca = kabsch_rmsd(pred_polar_coord[:][:,1][:], truth_polar_coord[:][:,1][:])
        rmsd_C = kabsch_rmsd(pred_polar_coord[:][:,2][:], truth_polar_coord[:][:,2][:])

        Cart_pred,Cart_truth = _get_cartesian(torch.tensor(pred_polar_coord).view(-1, 9), torch.tensor(truth_polar_coord).view(-1, 9))
        Cart_pred[0] = Cart_truth[0]
        Cart_pred[-1] = Cart_truth[-1]
        
        C_alpha_pred = Cart_pred[:,3:6].numpy()
        C_alpha_truth = Cart_truth[:,3:6].numpy()
        
        for entry in range(len(C_alpha_pred)):
            if entry == 0: 
                C_alpha_pred[entry] = C_alpha_pred[entry] + first_residue_coord
                C_alpha_truth[entry] = C_alpha_truth[entry] + first_residue_coord
            else:
                C_alpha_pred[entry] = C_alpha_pred[entry] + C_alpha_pred[entry-1]
                C_alpha_truth[entry] = C_alpha_truth[entry] + C_alpha_truth[entry-1]

        # Calculating the Kabsch RMSD with reconstructed features
        rmsd_cart_Ca = kabsch_rmsd(C_alpha_pred,C_alpha_truth)

        perplexity.append(ppl.item())
        rmsd_pred.append(rmsd_Ca)
        RMSD_test_n.append(rmsd_N)
        RMSD_test_ca.append(rmsd_Ca)
        RMSD_test_ca_cart.append(rmsd_cart_Ca)
        RMSD_test_c.append(rmsd_C)

    metrics = {
        'mean_perplexity': np.array(perplexity).reshape(-1, 1).mean(axis=0)[0],
        'std_perplexity': np.array(perplexity).reshape(-1, 1).std(axis=0)[0],
        'mean_rmsd': np.array(RMSD_test_ca).reshape(-1, 1).mean(axis=0)[0],
        'std_rmsd': np.array(RMSD_test_ca).reshape(-1, 1).std(axis=0)[0],
        'mean_rmsd_n': np.array(RMSD_test_n).reshape(-1, 1).mean(axis=0)[0],
        'mean_rmsd_ca': np.array(RMSD_test_ca).reshape(-1, 1).mean(axis=0)[0],
        'mean_rmsd_ca_cart': np.array(RMSD_test_ca_cart).reshape(-1, 1).mean(axis=0)[0],
        'mean_rmsd_c': np.array(RMSD_test_c).reshape(-1, 1).mean(axis=0)[0]
    }

    return metrics

def decode_polar_coords(bb_combined):
    # (N_res, (r,a,g)_n + (r,a,g)_ca + (r,a,g)_c)
    coords_r =  bb_combined[:, [0,3,6]]
    coords_theta = bb_combined[:, [1,4,7]]
    coords_phi = 3.14/2 - bb_combined[:, [2,5,8]]

    cart_true = _transform_to_cart(coords_r, coords_theta, coords_phi)
    return cart_true

def convert_to_mda_writer(res_ids, bb_coords, save_dir="generated_peptides/"):
    """
    res_ids are (N, 1)
    bb_coords are (N, 9) for {N, Ca, C}, each with xyz coordinates
    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # res_ids_split = torch.split(res_ids, chunk_sizes, dim=0)
    # bb_coords_split = torch.split(bb_coords, chunk_sizes, dim=0)

    # res_split = res_ids_split[sp]
    # bb_split = bb_coords_split[sp]

    decoded_coords = decode_polar_coords(bb_coords) # (N_res, 9)
    decoded_coords = decoded_coords.reshape(len(res_ids)*3, 3)

    atom_names = []
    for _ in range(len(res_ids)):
        atom_names.append("N")
        atom_names.append("CA")
        atom_names.append("C")

    possible_residues = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'ALA', 'LEU', 'MLE', 'NAL', 'NLE', 'PHE', 'PRO']
    res_mapper = {i : sym for i, sym in enumerate(possible_residues)}
    decoded_residues = [res_mapper[idx.item()] for idx in res_ids]
    print (decoded_residues)

    atom_names = ["N", "CA", "C"] * len(res_ids)
    atom2res = []
    for i in range(len(res_ids)):
        atom2res.append(i)
        atom2res.append(i)
        atom2res.append(i)

    uni = mda.Universe.empty(n_atoms=len(atom_names), n_residues=len(res_ids), atom_resindex=atom2res, trajectory=True)
    uni.add_TopologyAttr("name", atom_names)
    uni.add_TopologyAttr("resname", decoded_residues)

    bonds = []
    for o in range(0, len(atom_names)-1):
        bonds.append([i, i+1]) # ij
        bonds.append([i+i, i]) # ji
    uni.add_TopologyAttr('bonds', bonds)

    uni.atoms.positions = decoded_coords.numpy()

    with mda.Writer(f"{save_dir}/generated_peptide_{int(time.time())}.pdb", len(uni.atoms)) as w:
        w.write(uni)

    print (f"Saved molecule")

if __name__ == "__main__":
    peptide_data = get_graph_data_pyg(process_data_mda("peptide_data/pdb_with_atom_connectivity_water/peptides/"))
    pprint (peptide_data[:5])
