import torch
import torch.nn as nn
import torch_geometric as tg
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *
import gvpgnn


class PeptODE_uncond(nn.Module):
    def __init__(self, c_in, n_layers, params=None):
        super().__init__()
        layers_mlp = []
        activation_fns = []
        layers_mlp.append(tg.nn.TransformerConv(c_in+1, 128, heads=1, edge_dim=1))
        layers_mlp.append(tg.nn.TransformerConv(128, 512, heads=1, edge_dim=1))
        layers_mlp.append(tg.nn.TransformerConv(512, 512, heads=1, edge_dim=1))
        layers_mlp.append(tg.nn.TransformerConv(512, 512, heads=1, edge_dim=1))
        layers_mlp.append(tg.nn.TransformerConv(512, 64, heads=1, edge_dim=1))
        layers_mlp.append(tg.nn.TransformerConv(64, c_in, heads=1, edge_dim=1))

        # layers_mlp.append(gvpgnn.EGNNLayer(c_in+1))
        # layers_mlp.append(gvpgnn.EGNNLayer(128))
        # layers_mlp.append(gvpgnn.EGNNLayer(128))
        # layers_mlp.append(gvpgnn.EGNNLayer(c_in))
        
        activation_fns.append(nn.ReLU())
        activation_fns.append(nn.ReLU())
        activation_fns.append(nn.ReLU())
        activation_fns.append(nn.ReLU())

        self.layer_mlp = nn.ModuleList(layers_mlp)
        self.activation_mlp = nn.ModuleList(activation_fns)

        self.edge_index = 0
        self.edge_ab = 0
        self.antigen_coords = 0
        self.antigen_labels = 0
        self.order = 0
        self.amino_index = 0
        
    
    # def _get_quaternion(self,R):
    #     # Taken from https://github.com/wengong-jin/RefineGNN/blob/main/structgen/protein_features.py
    #     diag = torch.diagonal(R, dim1=-2, dim2=-1)
    #     Rxx, Ryy, Rzz = diag.unbind(-1)
    #     magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
    #           Rxx - Ryy - Rzz, 
    #         - Rxx + Ryy - Rzz, 
    #         - Rxx - Ryy + Rzz
    #     ], -1)))
    #     _R = lambda i,j: R[:,i,j]
    #     signs = torch.sign(torch.stack([
    #         _R(2,1) - _R(1,2),
    #         _R(0,2) - _R(2,0),
    #         _R(1,0) - _R(0,1)
    #     ], -1))
    #     xyz = signs * magnitudes
    #     # The relu enforces a non-negative trace
    #     w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    #     Q = torch.cat((xyz, w), -1)
    #     Q = F.normalize(Q, dim=-1)
    #     return Q
    
    def update_param(self, params):
        self.edge_index = params[0]
        self.amino_index = params[1]
        
    def _get_pairwise_scores(self,total_scores,edge_index):
        starting = torch.index_select(total_scores,0,edge_index[0].long())
        ending = torch.index_select(total_scores,0,edge_index[1].long())
        diff = starting - ending
        
        return diff
    
    
    def _get_pairwise(self,euclid_coords,edge_index):
            starting = torch.index_select(euclid_coords,0,edge_index[0].long())
            ending = torch.index_select(euclid_coords,0,edge_index[1].long())
            diff_vector = starting - ending
            diff_norm = torch.norm(diff_vector,dim=2).view(-1,3)
            diff_vector = F.normalize(diff_vector.view(-1,3,3), dim=2)
            
            
            #start_r = starting.view(-1,3,3)[:,:1,:1]
            #end_r = ending.view(-1,3,3)[:,:1,:1]
            #distance_vec = starting.view(-1,3,3)-ending.view(-1,3,3)
            #diff_dist = distance_vec[:,:,:1].view(-1,3)
            
            return diff_norm,diff_vector.view(-1,9)
    
    def _rbf_weight(self,d_ij):
        
        alpha_dist = d_ij.view(len(d_ij),3)
        diff_scaled = torch.div(alpha_dist,1*torch.ones(3*len(alpha_dist)).view(len(d_ij),3))
        RBF = torch.exp(-diff_scaled).view(-1,3)
        return RBF
        
    def _get_orientation_vector(self,O,r_ij,edge_index):
        # Taken from https://github.com/wengong-jin/RefineGNN/blob/main/structgen/protein_features.py
        starting_node_orientation = torch.index_select(O,0,edge_index[0].long()).view(-1,3,3)
        vector_mat = F.normalize(r_ij[:,3:6].view(-1,3,1),dim=-1)
        transposed_starting_mat = torch.transpose(starting_node_orientation,1,2)
        final_orient_vector = torch.matmul(transposed_starting_mat,vector_mat)
        return final_orient_vector.view(-1,3)
        
    def _get_orientations(self,coords,eps=1e-6):
        
        
        # Taken from https://github.com/wengong-jin/RefineGNN/blob/main/structgen/protein_features.py
        coords = coords.reshape(len(coords),3,3)
        X = coords[:,:3,:].reshape(1,3*coords.shape[0],3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(torch.tensor(dX), dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
    
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        
        O = F.pad(O, (0,0,1,2), 'constant', 0).view(len(coords),3,9)
        O_final = O[:,1,:]
        #print(O_final.shape)
        
        return O_final
    
    
    def _get_neighbour_orientation(self,O,edge_index):
        
        starting_node_orientation = torch.index_select(O,0,edge_index[0].long()).view(-1,3,3)
        ending_node_orientation = torch.index_select(O,0,edge_index[1].long()).view(-1,3,3)
        transposed_mat = torch.transpose(starting_node_orientation,1,2)
        final_combined_orient = torch.matmul(transposed_mat,ending_node_orientation)
        
        return final_combined_orient.view(-1,9)
        
        

    def forward(self, t, data):
        
        # Node_label, Node_coord  = data[:,:55], data[:,55:58]
        Node_coord  = data[:,0:3]
        
        Edge_index = self.edge_index.long()
        
        # r_ij,r_ij_vector = self._get_pairwise(Node_coord.view(-1,3,3),Edge_index_ag)
        
        # rbf_weight = self._rbf_weight(r_ij.to(data.device)).float().view(-1,3)
        
        # node_label_ag = self._get_pairwise_scores(Node_label,Edge_index_ag).view(-1,20)
        
        # spatial_diff = self._get_pairwise_scores(self.amino_index.view(-1,1),Edge_index_ag)
        
        # Orientations_node = self._get_orientations(Node_coord.view(-1,3,3))
        
        # orient_features = self._get_neighbour_orientation(Orientations_node,Edge_index_ag)
        
        # oriented_vector = self._get_orientation_vector(Orientations_node,r_ij_vector.view(-1,9),Edge_index_ag)
        
        # final_edge_feature = torch.cat([spatial_diff,node_label_ag,rbf_weight,r_ij_vector,orient_features,oriented_vector],dim=1).float()

        Node_coord = Node_coord.view(-1, 3)
        r_ij = torch.norm(Node_coord[Edge_index[0]] - Node_coord[Edge_index[1]], dim=1).view(-1, 1).float() # [E, 1])
        
        dx = data.float()
        # h = dx.float()
        tt = torch.ones_like(dx[:, :1]) * t
        dx_final = torch.cat([tt.float(), dx], 1)
        
        for l,layer in enumerate(self.layer_mlp):
            dx_final = layer(dx_final, edge_index=Edge_index, edge_attr=r_ij)
            
            if l != len(self.layer_mlp)-1: 
                dx_final = torch.relu(dx_final)

        return dx_final.float()