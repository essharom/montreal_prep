# Use this as a source folder for all custom functions we create throughout the different sections. 
# The idea is that we use this as a source from which we can pull functions we introduced in an earlier section

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import astropy.stats
from palettable import wesanderson
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import torch.nn as nn
import sklearn as sk
import seaborn as sns
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

def pearson_corr(data) : 

    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.cov(mat)

    for i in range(K) : 
        correl[i , : ] = cov[i , :] / np.sqrt(cov[i,i] * np.diag(cov))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def abs_bicorr(data) : 

    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    bicorr = astropy.stats.biweight_midcovariance(mat)

    for i in range(K) : 
        correl[i , : ] = bicorr[i , :] / np.sqrt(bicorr[i,i] * np.diag(bicorr))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def get_k_neighbours(df , k ) : 
    k_neighbours = {}
    if abs(df.max().max()) > 1 : 
        print('Dataframe should be a similarity matrix of max value 1')
    else:
        np.fill_diagonal(df.values , 1)
        for node in df.index : 
            neighbours = df.loc[node].nlargest(k+1).index.to_list()[1:] #Exclude the node itself
            k_neighbours[node] = neighbours
        
    return k_neighbours

def plot_knn_network(df , K , labels , node_colours = ['skyblue'] , node_size = 300) : 
    # Get K-nearest neighbours for each node
    k_neighbours = get_k_neighbours(df , k = K)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(df.index)
    
    nx.set_node_attributes(G , labels.astype('category').cat.codes , 'label')
    nx.set_node_attributes(G , pd.Series(np.arange(len(df.index)) , index=df.index) , 'idx')

    # Add edges based on the k-nearest neighbours
    for node, neighbours in k_neighbours.items():
        for neighbor in neighbours:
            G.add_edge(neighbor, node)

    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
    
    return G

def gen_graph_legend(node_colours , G , attr) : 
    
    patches = []
    for col , lab in zip(node_colours.drop_duplicates() , pd.Series(nx.get_node_attributes(G , attr)).drop_duplicates()) : 
        patches.append(mpatches.Patch(color=col, label=lab))
    
    return patches

def train(g, h, train_split , val_split , device ,  model , labels , epochs , lr):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    
    # training loop
    epoch_progress = tqdm(total=epochs, desc='Loss : ', unit='epoch')
    for epoch in range(epochs):
        model.train()

        logits  = model(g, h)

        loss = loss_fcn(logits[train_split], labels[train_split].float())
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        if (epoch % 5) == 0 :
            
            _, predicted = torch.max(logits[train_split], 1)
            _, true = torch.max(labels[train_split] , 1)
            train_acc.append((predicted == true).float().mean().item())

            valid_loss , valid_acc , *_ = evaluate(val_split, device, g , h, model , labels)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc)
            
            epoch_desc = (
                "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                    epoch, np.mean(train_loss[-5:]) , np.mean(train_acc[-5:]), np.mean(val_acc[-5:])
                )
            )
            
            epoch_progress.set_description(epoch_desc)
            epoch_progress.update(5)

    fig1 , ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(train_loss , label = 'Train Loss')
    ax1.plot(range(5 , len(train_loss)+1 , 5) , val_loss  , label = 'Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    fig2 , ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(train_acc  , label = 'Train Accuracy')
    ax2.plot(val_acc  , label = 'Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_ylim(0,1)
    ax2.legend()
    
    # Close tqdm for epochs
    epoch_progress.close()

    return fig1 , fig2

def evaluate(split, device, g , h, model , labels):
    model.eval()
    loss_fcn = nn.CrossEntropyLoss()
    acc = 0
    
    with torch.no_grad() : 
        logits = model(g, h)
        
        loss = loss_fcn(logits[split], labels[split].float())

        _, predicted = torch.max(logits[split], 1)
        _, true = torch.max(labels[split] , 1)
        acc = (predicted == true).float().mean().item()
        
        logits_out = logits[split].cpu().detach().numpy()
        binary_out = (logits_out == logits_out.max(1).reshape(-1,1))*1
        
        labels_out = labels[split].cpu().detach().numpy()
        
        PRC =  average_precision_score(labels_out , binary_out , average="weighted")
        SNS = recall_score(labels_out , binary_out , average="weighted")
        F1 = 2*((PRC*SNS)/(PRC+SNS))
        
    
    return loss , acc , F1 , PRC , SNS
    
class GCN(nn.Module):
    def __init__(self, input_dim,  hidden_feats, num_classes):
        
        super().__init__()
        
        self.gcnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gcnlayers.append(
                        GraphConv(input_dim , hidden_feats[layers])
                    )
                else :
                    self.gcnlayers.append(
                        GraphConv(hidden_feats[layers-1] , hidden_feats[layers])
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.gcnlayers.append(
                    GraphConv(hidden_feats[layers-1] , num_classes)
                )
                
        self.drop = nn.Dropout(0.05)

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        
        for layers in range(self.num_layers) : 
            if layers == self.num_layers - 1 : 
                h = self.gcnlayers[layers](g , h)
            else : 
                h = self.gcnlayers[layers](g, h)
                h = self.drop(F.relu(h))
            
        score = self.drop(h)
            
        return score