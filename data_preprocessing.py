#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''==============================================
# @Project : mpnn_mol2vec_mlp
# @File    : data_preprocessing.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2022/7/18 21:25
================================================'''
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

data_path = '../data/meta_data/'
data_name = 'AID_720552'
df = pd.read_csv(data_path+data_name+'.csv',usecols=['smiles','HIV_active'])
df['cid'] = df.index
names = ['smiles','label','cid']
df.columns = names

# clean smiles
df_no_smiles = df.drop(columns='smiles')

smiles = []
for i in df.smiles.tolist():
    cpd = str(i).split('.')
    cpd_longest = max(cpd, key=len)
    smiles.append(cpd_longest)

smiles = pd.Series(smiles, name='smiles')

df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)
df_clean_smiles.to_csv(data_path+data_name+'_cleaned-smiles.csv', index=False)

# WeightedRandomSampler权重随机采样
df = pd.read_csv(data_path+data_name+'_cleaned-smiles.csv', header=0)
dataset = list(zip(df['cid'], df['smiles'], df['label']))

BATCH_SIZE = 100

labels_unique, counts = np.unique(df['label'], return_counts=True)
print("Unique labels : {},counts :{}".format(labels_unique, counts))

class_weights = [sum(counts) / c for c in counts]
print(class_weights)

example_weights = [class_weights[e] for e in df['label']]
sampler = WeightedRandomSampler(example_weights, len(df['label']))

train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

if os.path.exists(data_path+data_name+'_balanced.csv'):
    os.remove(data_path+data_name+'_balanced.csv')

for batch_idx, (cid, smiles, labels) in enumerate(train_dataloader):
    pd_output = pd.DataFrame({'cid': cid, 'smiles': smiles, 'label': labels})
    pd_output.to_csv(data_path+data_name+'_balanced.csv',
                     index=False, mode='a', header=None)

table = pd.read_csv(data_path+data_name+'_balanced.csv', header=0)
table.columns = ['cid', 'smiles', 'label']
labels_unique, counts = np.unique(table['label'], return_counts=True)
print("Unique labels : {},counts :{}".format(labels_unique, counts))