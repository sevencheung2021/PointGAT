# from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class PointGAT(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout,xyz_feature_dim):
        super(PointGAT, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
#         self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
#         self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
#         self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num)


        self.radius = radius
        self.T = T

        self.relu = nn.ReLU()
        
        # self.transform = nn.Linear(xyz_feature_dim, 64)
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv1 = torch.nn.Conv1d(6, 64, 1)   # feature_dim =5 [x,y,z]
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, fingerprint_dim)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(fingerprint_dim)
        

        self.fc3 = nn.Linear(fingerprint_dim*2, fingerprint_dim)
        self.fc4 = nn.Linear(fingerprint_dim,1)


    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask,xyz_feature):
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))
        
       
        identity = atom_feature
        # print(atom_feature.shape)

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
#             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
        attention_weight = attention_weight * attend_mask
#         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        #do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
#             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
        #     print('atom_features shape')
        #     print(atom_feature.shape)

            # do nonlinearity
            activated_features = F.relu(atom_feature)
        # print('activated_features shape')
        # print(activated_features.shape)


        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        # print('features_mol from sum atom feature shape')
        # print(mol_feature.shape)

        atom_level_mol_feature = mol_feature  # ZR add


        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)           
        # print('activated_features_mol from sum atom feature shape')
        # print(activated_features_mol.shape)


        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
#             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
#             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)

        #     print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)           

        mol_feature = mol_feature + atom_level_mol_feature  # ZR add

        ##### xyz_feature #####
        batchsize = batch_size

        xyz_feature = xyz_feature.permute(0,2,1)
        
        xyz = F.leaky_relu(self.bn1(self.conv1(xyz_feature)))
        xyz = F.leaky_relu(self.bn2(self.conv2(xyz)))
        xyz = F.leaky_relu(self.bn3(self.conv3(xyz)))
        xyz = torch.max(xyz, 2, keepdim=True)[0]
        # xyz = xyz.view(-1, 1024)
        xyz = xyz.view(-1, 1024)
        

        xyz = F.relu(self.fc1(xyz))
        xyz = F.relu(self.fc2(xyz))
        

        cat = torch.cat([mol_feature,xyz],dim=1) #(N,256*2)
        
        final_mol = F.relu(self.fc3(cat))
        
        mol_prediction = self.fc4(final_mol)
        
        return mol_prediction
        
        

