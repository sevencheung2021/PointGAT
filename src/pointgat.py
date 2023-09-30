'''
Description: 
version: 
Email: rongzhangthu@yeah.net
Author: Rong Zhang
Date: 2022-06-13 18:10:11
LastEditors: Seven Rong Cheung
LastEditTime: 2023-09-28 11:06:13
'''


import os
import copy
import json
import pickle
import time
import sys
import numpy as np
import pandas as pd


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


sys.setrecursionlimit(50000)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True


from PointGAT.PointGAT_Layers import PointGAT
from PointGAT import save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# log files to write
class Logger_print():
    def __init__(self,filename = 'default log',stream = sys.stdout):
        self.terminal = stream
        self.log = open(filename,'w')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger_print('C10_mmff_lowest_2023.log',sys.stdout)


##  generate feature rb files 
def generateGeoFeatureFile(file_path,file_name):
    raw_filename = os.path.join(file_path,file_name)
    feature_filename = raw_filename.replace('.csv','.pickle')
    filename = raw_filename.replace('.csv','')
    smiles_tasks_df = pd.read_csv(raw_filename,index_col=False)
    smiles_tasks_df.sample().reset_index(drop=True)
    smilesList = smiles_tasks_df.smiles.tolist()
    
    print("number of {} smiles {}: ".format(filename,len(smilesList)))
    
    if os.path.isfile(feature_filename):
        print('loading*****************************************')
        feature_dicts = pickle.load(open(feature_filename, "rb" ))
        # print(feature_dicts)
    else:
        print('generating*****************************************')
        feature_dicts = save_smiles_dicts(smilesList,filename)
    return smiles_tasks_df,feature_dicts


## train
def train(model, dataset, optimizer, loss_function,feature_dicts,tasks,batch_size,xyz_feature):
    model.train()
    np.random.seed(8)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.smiles.values
        y_val = batch_df[tasks[0]].values
        
        
        # print(data)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_lis = get_smiles_array(smiles_list,feature_dicts)
        
        x_xyz = batch_df[xyz_feature[0]].values
        x_xyz = [json.loads(val) for val in x_xyz]   
        x_xyz = np.asarray(x_xyz)
        
        # print(x_atom)
        mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask),torch.Tensor(x_xyz))
        
        model.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1))     
        loss.backward()
        optimizer.step()


## eval 
def eval(model, dataset,feature_dicts,tasks,batch_size,xyz_feature):
    model.eval()
    eval_MAE_list = []
    eval_MSE_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch) 
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.smiles.values
#         print(batch_df)
        y_val = batch_df[tasks[0]].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list= get_smiles_array(smiles_list,feature_dicts)

        x_xyz = batch_df[xyz_feature[0]].values
        x_xyz = [json.loads(val) for val in x_xyz]   
        x_xyz = np.asarray(x_xyz)

        mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask),torch.Tensor(x_xyz))
        
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
#      
        eval_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        # print(len(eval_MAE_list))
        eval_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
    return np.array(eval_MAE_list).mean(), np.array(eval_MSE_list).mean()


def main():

    ## model parameters
    start_time = str(time.ctime()).replace(':','-').replace(' ','_')
    tasks = ['normalize_energy']
    xyz_feature = ['xyz_feature']
    batch_size = 32
    epochs = 800
    p_dropout= 0.1
    fingerprint_dim = 215
    weight_decay = 6.0 # also known as l2_regularization_lambda
    learning_rate = 3.5
    output_units_num = 1 # for regression model
    radius = 6
    T = 4
    xyz_feature_dim = 6

    ## data path
    data_file_path = '../Data/'
    train_file_name = 'c10_train.csv'
    valid_file_name = 'c10_valid.csv'
    test_file_name = 'c10_test.csv'
    
    ## load data
    train_df,train_feature_dicts = generateGeoFeatureFile(file_path=data_file_path,file_name=train_file_name)
    valid_df,valid_feature_dicts = generateGeoFeatureFile(file_path=data_file_path,file_name=valid_file_name)
    test_df,test_feature_dicts = generateGeoFeatureFile(file_path=data_file_path,file_name=test_file_name)
    
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([train_df.smiles.values[0]],train_feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    
    
    # model setup
    loss_function = nn.MSELoss()
    model = PointGAT(radius, T, num_atom_features, num_bond_features,
            fingerprint_dim, output_units_num, p_dropout,xyz_feature_dim)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('**** model params:{}'.format(params))

    ## perform training process
    best_param ={}
    best_param["train_epoch"] = 0
    best_param["valid_epoch"] = 0
    best_param["train_MAE"] = 9e8
    best_param["valid_MAE"] = 9e8

    collector= {k: [] for k in ['epoch',
                                    'train_loss',
                                    'valid_loss',
                                    'train_RMSE',
                                    'valid_RMSE',
                                    'train_MAE',
                                    'valid_MAE']}
    print('training*************************************************************')
    for epoch in range(epochs):
        
        train_MAE, train_MSE = eval(model, train_df,train_feature_dicts,tasks,batch_size,xyz_feature)
        valid_MAE, valid_MSE = eval(model, valid_df,valid_feature_dicts,tasks,batch_size,xyz_feature)

        if train_MAE < best_param["train_MAE"]:
            best_param["train_epoch"] = epoch
            best_param["train_MAE"] = train_MAE
        if valid_MAE < best_param["valid_MAE"]:
            best_param["valid_epoch"] = epoch
            best_param["valid_MAE"] = valid_MAE
            
            if valid_MAE < 0.025:

                torch.save(model, '../weights/model_'+'C10_'+start_time+'_'+str(best_param["valid_epoch"])+'.pt')

        if (epoch - best_param["train_epoch"] >100) and (epoch - best_param["valid_epoch"] >100):        
            break

        print("Epoch",epoch)
        print('train_RMSE and valid_RMSE')
        print(epoch, np.sqrt(train_MSE), np.sqrt(valid_MSE))
        print('train_MAE and valid_MAE')
        print(epoch, train_MAE,valid_MAE)

        ## training 
        train(model, train_df, optimizer, loss_function,train_feature_dicts,tasks,batch_size,xyz_feature)

        collector['epoch'].append(epoch+1)
        collector['train_loss'].append(train_MSE)
        collector['train_RMSE'].append(np.sqrt(train_MSE))
        collector['train_MAE'].append(train_MAE)
        
        collector['valid_loss'].append(valid_MSE)
        collector['valid_RMSE'].append(np.sqrt(valid_MSE))
        collector['valid_MAE'].append(valid_MAE)

        pd.DataFrame(collector).to_csv('C10_pointgat_training_result.csv', index=False)
    

    # evaluate model
    print('evaluate*************************************************************')
    best_model = torch.load('../weights/model_'+'C10_'+start_time+'_'+str(best_param["valid_epoch"])+'.pt')

    best_model_dict = best_model.state_dict()
    best_model_wts = copy.deepcopy(best_model_dict)

    model.load_state_dict(best_model_wts)
    (best_model.align[0].weight == model.align[0].weight).all()
    test_MAE, test_MSE = eval(model, test_df,test_feature_dicts,tasks,batch_size,xyz_feature)

    print("best epoch:",best_param["valid_epoch"],"\n","test RMSE:",np.sqrt(test_MSE),"test MAE:",test_MAE)
    


if __name__ =='__main__':

    main()

