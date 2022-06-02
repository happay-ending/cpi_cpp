#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''==============================================
# @Project : CPI_CPP
# @File    : optuna_CPI_CPP.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2022/3/6 15:08
================================================'''
'''
Dependencies:
-	python	3.7.13
-	torch	1.11.0+cu102
-	dgl-cuda10.2	0.8.1
-	dgllife	0.2.9
-	rdkit	2018.09.3
-	gensim	4.2.0
-	networkx	2.2
-	numpy	1.21.6
-	pandas	1.3.5
-	scikit-learn	1.0.2
-	scipy	1.7.3
-   optuna 2.10.0
'''
import random
import sys
import warnings
import time

import dgl
import optuna
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from dgl.nn.pytorch import Set2Set
from dgllife.model.gnn.mpnn import MPNNGNN


from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph, CanonicalBondFeaturizer
from matplotlib import pyplot as plt
from optuna.samplers import TPESampler
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from word2vec.features import mol2alt_sentence, DfVec, sentences2vec

from word2vec.word2vec import seq_to_kmers, get_protein_sequence_embedding
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")


# Define predictor layer
class mlp_layer(nn.Sequential):
    def __init__(self, in_feats, out_feats, dropout=0.):
        super(mlp_layer, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.BatchNorm1d(out_feats),
        )

    def forward(self, feats):
        return self.predict(feats)

# Define predictor
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims_lst, dropout=0.):
        '''
        input_dim (int)
        output_dim (int)
        hidden_dims_lst (list, each element is a integer, indicating the hidden size)
        '''
        super(MLP, self).__init__()
        layer_size = len(hidden_dims_lst)
        dims = [input_dim] + hidden_dims_lst + [output_dim]

        self.mlp_layers = nn.ModuleList()
        for i in range(layer_size):
            self.mlp_layers.append(mlp_layer(dims[i], dims[i + 1], dropout))
        self.predictor = nn.Linear(dims[-2], dims[-1])

    def forward(self, feats):
        for mlp in self.mlp_layers:
            feats = mlp(feats)
        return self.predictor(feats)

# Define deep learning encoder module
class cpi_cpp_net(nn.Module):
    def __init__(self, node_in_feats_drug, edge_in_feats_drug, node_out_feats_drug=64, edge_hidden_feats_drug=128,
                 num_step_message_passing_drug=6, num_step_set2set_drug=6, num_layer_set2set_drug=3,
                 in_feats_protein=100, mlp_hidden_feats=[64, 64]):
        super(cpi_cpp_net, self).__init__()

        self.gnn_drug = MPNNGNN(node_in_feats=node_in_feats_drug, node_out_feats=node_out_feats_drug,
                                edge_in_feats=edge_in_feats_drug, edge_hidden_feats=edge_hidden_feats_drug,
                                num_step_message_passing=num_step_message_passing_drug)
        self.readout = Set2Set(input_dim=node_out_feats_drug, n_iters=num_step_set2set_drug,
                               n_layers=num_layer_set2set_drug)

        self.predictor = MLP(2 * node_out_feats_drug + in_feats_protein, 2, mlp_hidden_feats)


    def forward(self, bg_drug, node_feats_drug, edge_feats_drug, feats_protein):
        node_feats = self.gnn_drug(bg_drug, node_feats_drug, edge_feats_drug)
        graph_feats_drug = self.readout(bg_drug, node_feats)
        cat_vector = torch.cat((graph_feats_drug, feats_protein), 1)
        interaction = self.predictor(cat_vector)
        return interaction


# Define metric function
def metric_function(y_true, y_pred, y_score):
    acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    # fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    return acc, auc, precision, recall, f1, mcc


# Define train function
def train_epoch(model, device, dataloader, criterion, optimizer):
    train_y_true = []
    train_y_pred = []
    train_y_score = []
    train_epoch_loss = 0.0

    model.train()
    for batch_idx, (drug_graph, protein_vector, labels) in enumerate(dataloader):
        labels = labels.to(device)
        atom_feats = drug_graph.ndata.pop('h').to(device)
        bond_feats = drug_graph.edata.pop('e').to(device)
        drug_graph = drug_graph.to(device)
        protein_vector = protein_vector.to(device)

        optimizer.zero_grad()
        logits = model(drug_graph, atom_feats, bond_feats, protein_vector)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.detach().item()
        pred = logits.argmax(dim=1)
        score = torch.select(logits, 1, 1)

        train_y_true += labels.to('cpu').numpy().flatten().tolist()
        train_y_pred += pred.to('cpu').detach().numpy().flatten().tolist()
        train_y_score += score.to('cpu').detach().numpy().flatten().tolist()

    train_epoch_loss /= (batch_idx + 1)
    train_acc, train_auc, train_precision, train_recall, train_f1, train_MCC = metric_function(
        y_true=train_y_true, y_pred=train_y_pred, y_score=train_y_score)

    return train_epoch_loss, train_acc, train_auc, train_precision, train_recall, train_f1, train_MCC


# Define valid function
def valid_epoch(model, device, dataloader, criterion):
    test_y_true = []
    test_y_pred = []
    test_y_score = []
    test_epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for batch_idx, (drug_graph, protein_vector, labels) in enumerate(dataloader):

            labels = labels.to(device)
            atom_feats = drug_graph.ndata.pop('h').to(device)
            bond_feats = drug_graph.edata.pop('e').to(device)
            drug_graph = drug_graph.to(device)
            protein_vector = protein_vector.to(device)

            logits = model(drug_graph, atom_feats, bond_feats, protein_vector)
            loss = criterion(logits, labels)

            test_epoch_loss += loss.detach().item()
            pred = logits.argmax(dim=1)
            score = torch.select(logits, 1, 1)

            test_y_true += labels.to('cpu').numpy().flatten().tolist()
            test_y_pred += pred.to('cpu').detach().numpy().flatten().tolist()
            test_y_score += score.to('cpu').detach().numpy().flatten().tolist()

    test_epoch_loss /= (batch_idx + 1)
    test_acc, test_auc, test_precision, test_recall, test_f1, test_MCC = metric_function(
        y_true=test_y_true, y_pred=test_y_pred, y_score=test_y_score)

    return test_epoch_loss, test_acc, test_auc, test_precision, test_recall, test_f1, test_MCC


# Define collate function
def collate(sample):
    drug_graph, protein_vector, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(drug_graph)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(np.array(protein_vector)), torch.tensor(labels)

# Seed function
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def drawPlot(name, param):
    plt.title(name + ' in cross entropy averaged over minibatches')
    plt.plot(param)
    plt.show()


def define_model(trial):

    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e', self_loop=True)

    node_in_feats_drug = node_featurizer.feat_size('h')
    edge_in_feats_drug = edge_featurizer.feat_size('e')
    node_out_feats_drug = trial.suggest_int("node_out_feats_compound", 32, 128, step=32)
    edge_hidden_feats_drug = trial.suggest_int(
        "edge_hidden_feats_compound", 64, 256, step=64)
    num_step_set2set_drug = trial.suggest_int("num_step_set2set_compound", 3, 9, step=3)
    num_layer_set2set_drug = trial.suggest_int("num_layer_set2set_compound", 1, 3)
    num_step_message_passing_drug = trial.suggest_int(
        "num_step_message_passing_compound", 2, 6)

    mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
    mlp_hidden_dims_lst = []
    for i in range(mlp_layers):
        mlp_features = trial.suggest_int(
            "mlp_n_units_l{}".format(i), 32, 256, step=16)
        mlp_hidden_dims_lst.append(mlp_features)

    _, _, type = file_doc_info()
    if type == 'cpp':
        in_feats_protein = 300
    else:
        in_feats_protein = 100

    cpi_cpp_model = cpi_cpp_net(node_in_feats_drug=node_in_feats_drug, edge_in_feats_drug=edge_in_feats_drug,
                                node_out_feats_drug=node_out_feats_drug, edge_hidden_feats_drug=edge_hidden_feats_drug,
                                num_step_message_passing_drug=num_step_message_passing_drug,
                                num_step_set2set_drug=num_step_set2set_drug,
                                num_layer_set2set_drug=num_layer_set2set_drug, in_feats_protein=in_feats_protein,
                                mlp_hidden_feats=mlp_hidden_dims_lst)

    return cpi_cpp_model


def read_cpi_raw_data(file_path, file_name):
    if 'kiba' in file_name:
        df = pd.read_csv(file_path + file_name + '.csv')

        labels = np.array(df['label'])
        labels = np.array(labels, dtype=np.int64)

        df_ligands = pd.read_csv(file_path + 'ligands_smiles.csv')
        df_compound_info = pd.merge(left=df['cid'], right=df_ligands, on='cid', how='left')  # cid,smiles

        df_proteins = pd.read_csv(file_path + 'protein_seq.csv')

        df_proteins_info = pd.merge(left=df['pid'], right=df_proteins, on='pid', how='left')  # pid,sequence

        smiles = np.array(df_compound_info['smiles'])
        sequence = np.array(df_proteins_info['sequence'])
        data_x = np.concatenate((smiles[:, np.newaxis], sequence[:, np.newaxis]), axis=1)
    else:
        df = pd.read_csv(file_path + file_name + '.txt', header=None, sep=' ')
        df.columns = ['smiles', 'sequence', 'label']
        labels = np.array(df['label'])
        labels = np.array(labels, dtype=np.int64)
        smiles = np.array(df['smiles'])
        sequence = np.array(df['sequence'])
        data_x = np.concatenate((smiles[:, np.newaxis], sequence[:, np.newaxis]), axis=1)
    return data_x, labels


def read_cpp_raw_data(file_path, file_name):
    df = pd.read_csv(file_path + file_name + '.csv')
    labels = np.array(df['label'])
    labels = np.array(labels, dtype=np.int64)
    smiles = np.array(df['smiles'])
    return smiles, labels


def get_compound_graph(train_smiles):
    mols = [Chem.MolFromSmiles(x) for x in train_smiles]
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e', self_loop=True)
    data_g = [mol_to_bigraph(m, add_self_loop=True, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
              for m in mols]

    return data_g


def get_protein_vec(train_sequence):
    proteins = []
    model = Word2Vec.load("./word2vec/models/word2vec_30.model")
    for each in train_sequence:
        protein_embedding = get_protein_sequence_embedding(model, seq_to_kmers(each))
        proteins.append(protein_embedding)

    return np.array(proteins, dtype=np.float32)


def get_compound_sequenc_vec(train_sequence):
    aas = [Chem.MolFromSmiles(x) for x in train_sequence]
    model = Word2Vec.load('./word2vec/models/model_300dim.pkl')
    aa_sentences = [mol2alt_sentence(x, 1) for x in aas]
    mol2vec = [DfVec(x) for x in sentences2vec(aa_sentences, model, unseen='UNK')]
    sequence_vec = np.array([x.vec for x in mol2vec], dtype=np.float32)
    return sequence_vec


def objective(trial):
    file_path, file_name, type = file_doc_info()

    EPOCHS = 150
    K = 5  # k-fold cross-validation

    criterion = CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type == "cpi":
        data_X, data_y = read_cpi_raw_data(file_path, file_name)
    else:
        data_X, data_y = read_cpp_raw_data(file_path, file_name)

    scv = StratifiedKFold(n_splits=K, random_state=set_seed(42), shuffle=True)
    fold = 0
    fold_acc_avg = []
    for train_index, val_index in scv.split(data_X, data_y):

        train_X, val_X = data_X[train_index], data_X[val_index]
        train_y, val_y = data_y[train_index], data_y[val_index]

        if type == "cpi":
            train_smiles, train_sequence = train_X[:, 0], train_X[:, 1]
            train_drug_graph = get_compound_graph(train_smiles)
            train_sequence_vec = get_protein_vec(train_sequence)
            val_smiles, val_sequence = val_X[:, 0], val_X[:, 1]
            val_drug_graph = get_compound_graph(val_smiles)
            val_sequence_vec = get_protein_vec(val_sequence)
        else:
            train_smiles, train_sequence = train_X, train_X
            train_drug_graph = get_compound_graph(train_smiles)
            train_sequence_vec = get_compound_sequenc_vec(train_sequence)
            val_smiles, val_sequence = val_X, val_X
            val_drug_graph = get_compound_graph(val_smiles)
            val_sequence_vec = get_compound_sequenc_vec(val_sequence)

        train_data = list(zip(train_drug_graph, train_sequence_vec, train_y))
        val_data = list(zip(val_drug_graph, val_sequence_vec, val_y))

        model = define_model(trial).to(device)

        batch_size = trial.suggest_int("batch_size", 64, 128, step=64)
        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"])
        # Create Adjustable Learning Rates
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
        test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_precision': [], 'train_recall': [],
                   'train_f1': [], 'train_MCC': [
            ], 'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
                   'test_MCC': []}

        for epoch in range(EPOCHS):
            train_loss, train_acc, train_auc, train_precision, train_recall, train_f1, train_MCC = train_epoch(
                model, device, train_loader, criterion, optimizer)
            test_loss, test_acc, test_auc, test_precision, test_recall, test_f1, test_MCC = valid_epoch(
                model, device, test_loader, criterion)

            ###Save the cross-validation test epoch results for each trial
            columns_name = ['trial', 'fold', 'epoch', 'loss', 'acc', 'auc', 'precision', 'recall', 'f1', 'MCC']
            columns_list = [trial.number, fold, epoch, test_loss, test_acc, test_auc, test_precision, test_recall,
                            test_f1, test_MCC]
            save_data = pd.DataFrame(data=[columns_list], columns=columns_name)
            save_data.to_csv(file_path + 'output/'+ file_name + '_optuna_results.csv', index=False, mode='a', header=None)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['train_MCC'].append(train_MCC)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_auc'].append(test_auc)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_f1'].append(test_f1)
            history['test_MCC'].append(test_MCC)

        print('Performance of No.{} fold '.format(fold + 1))
        print(
            '{:5} | Fold:  {:2}/{:2} | Loss: {:.5f} | Acc: {:.5f} | AUC:{:.5f} | Precision:{:.5f} | Recall:{:.5f} | F1:{:.5f} | MCC:{:.5f}'.format(
                "Train",
                fold + 1, K, np.mean(history['train_loss']), np.mean(history['train_acc']),
                np.mean(history['train_auc']), np.mean(history['train_precision']), np.mean(history['train_recall']),
                np.mean(history['train_f1']), np.mean(history['train_MCC'])))
        print(
            '{:5} | Fold:  {:2}/{:2} | Loss: {:.5f} | Acc: {:.5f} | AUC:{:.5f} | Precision:{:.5f} | Recall:{:.5f} | F1:{:.5f} | MCC:{:.5f}'.format(
                "Test",
                fold + 1, K, np.mean(history['test_loss']), np.mean(history['test_acc']), np.mean(history['test_auc']),
                np.mean(history['test_precision']), np.mean(history['test_recall']), np.mean(history['test_f1']),
                np.mean(history['test_MCC'])))
        fold_acc_avg.append(np.mean(history['test_acc']))
        fold += 1
        trial.report(np.mean(fold_acc_avg), fold + 1)
        # trial.report(np.mean(history['test_acc']), fold + 1)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # torch.save(model.state_dict(), file_path + 'output/model/' + file_name +
    #            '_trial_' + str(trial.number) + '_params.pkl')
    # return np.mean(history['test_acc'])

    return np.mean(fold_acc_avg)


def file_doc_info():
    file_path = './datasets/'
    # file_path = './datasets/KIBA/'
    file_name = 'human'
    # file_name = 'celegans'
    # file_name = 'kiba'

    # file_name = 'bace'
    # file_name = 'hERG'
    # file_name = 'P53'
    type = 'cpi'  # cpi or cpp

    return file_path, file_name, type


if __name__ == '__main__':
    file_path, file_name, type = file_doc_info()
    if type == 'cpi':
        print("CPI prediction classifier running...!")
    elif type == 'cpp':
        print("Molecular property prediction classifier running...!")
    else:
        print("Type error, enter type again!")
        sys.exit()
    start_time = time.time()
    set_seed(42)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name='CPP_and_CPI_Predictor', direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Number: ", trial.number)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    end_time = time.time()
    print("Code runs in {} seconds!".format(end_time - start_time))

    print(study.best_params)
    print(study.best_trial.intermediate_values)
    best_params = pd.DataFrame.from_dict([study.best_params])
    best_params['trial_number'] = study.best_trial.number
    best_params.to_csv(file_path + 'output/'+ file_name + '_best_params.csv', index=False)
    intermediate_values = pd.DataFrame.from_dict([study.best_trial.intermediate_values])
    intermediate_values.to_csv(file_path +'output/'+ file_name + '_intermediate_values.csv', index=False)

