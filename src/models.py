import os
import math
import pymetis
import collections
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        features = torch.mm(features, self.weight)
        output = torch.spmm(adj, features)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SageConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.linear = nn.Linear(in_features*2, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)

    def forward(self, features, adj):
        neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        features = torch.cat([features, neigh_feature], dim=-1)
        output = self.linear(features)

        return output


class SemanticLayer(nn.Module):

    def __init__(self, in_features, out_features, nheads, graph_mode=1):
        super(SemanticLayer, self).__init__()
        
        self.nheads = nheads
        self.graph_mode = graph_mode

        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.graph_mode == 0:
            self.classifier = nn.Linear(out_features, nheads, bias=False)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.graph_mode == 1:
            self.classifier = nn.Linear(out_features, nheads, bias=False)
        else:
            self.classifier = nn.Linear(out_features, 1, bias=False)
            self.loss_fn = nn.MSELoss()

        self.layers = nn.ModuleList()
        self.atts = nn.ModuleList()
        self.convs_1 = nn.ModuleList()
        self.convs_2 = nn.ModuleList()

        for _ in range(nheads):
            self.layers.append(nn.Linear(in_features, int(out_features / nheads), bias=False))
            self.atts.append(nn.Linear(out_features*2, 1, bias=False))
            self.convs_1.append(GraphConvolution(out_features, int(out_features / 2)))
            self.convs_2.append(GraphConvolution(int(out_features / 2), out_features))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        nn.init.normal_(self.classifier.weight)
        for linear in self.layers:
            nn.init.xavier_uniform_(linear.weight, gain=1.414)
        for att in self.atts:
            nn.init.xavier_uniform_(att.weight, gain=1.414)

    def forward(self, x, adj):

        h = self.linear(x)
        edge_list = adj.nonzero().t()
        edge_h = torch.cat((h[edge_list[0, :], :], h[edge_list[1, :], :]), dim=1)

        out_features = []
        self.out_descriptor = []

        for i in range(self.nheads):
            e = torch.sigmoid(self.atts[i](edge_h).squeeze(1))
            new_adj = torch.sparse.FloatTensor(edge_list, e, torch.Size([h.shape[0], h.shape[0]])).to_dense()

            features = torch.matmul(new_adj, x)
            out = self.layers[i](features)
            out_features.append(out)
            
            descriptor = torch.tanh(self.convs_1[i](h.detach(), new_adj))
            descriptor = torch.tanh(self.convs_2[i](descriptor, new_adj))
            self.out_descriptor.append(torch.mean(descriptor, dim=0, keepdim=True))

        out = torch.cat(tuple([rst for rst in out_features]), -1)

        return out


    def compute_semantic_loss(self):

        labels = [torch.ones(1)*i for i in range(self.nheads)]
        labels = torch.cat(tuple(labels), 0).long().to(device)

        factors_feature = torch.cat(tuple(self.out_descriptor), 0)
        pred = self.classifier(factors_feature)

        if self.graph_mode == 0:
            pred = nn.Softmax(dim=1)(pred)
            loss_sem = self.loss_fn(pred, labels)
        elif self.graph_mode == 1:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(torch.cosine_similarity(pred[i], pred[j], dim=0))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = torch.mean(loss_sem_list)
        else:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(self.loss_fn(pred[i], pred[j]))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = - torch.mean(loss_sem_list)    

        return loss_sem
                

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):

        h = torch.mm(x, self.W)
        edge_list = adj.nonzero().t()
        edge_h = torch.cat((h[edge_list[0, :], :], h[edge_list[1, :], :]), dim=1)

        e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze(1))
        new_adj = torch.sparse.FloatTensor(edge_list, e, torch.Size([h.shape[0], h.shape[0]])).to_dense()

        zero_vec = -9e15*torch.ones_like(new_adj)
        attention = torch.where(adj > 0, new_adj, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)


class GCN_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_En2(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_Classifier(nn.Module):
    
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class Sage_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_En2(nn.Module):
    
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_Classifier(nn.Module):

    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class SEM_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4, graph_mode=1):
        super(SEM_En, self).__init__()

        self.sem1 = SemanticLayer(nfeat, nhid, nheads=nheads, graph_mode=graph_mode)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sem1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        sem_loss = self.sem1.compute_semantic_loss()

        return x, sem_loss


class SEM_En2(nn.Module):
    
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4, graph_mode=1):
        super(SEM_En2, self).__init__()

        self.sem1 = SemanticLayer(nfeat, nhid, nheads=nheads, graph_mode=graph_mode)
        self.sem2 = SemanticLayer(nhid, nembed, nheads=nheads, graph_mode=graph_mode)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sem1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sem2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        sem_loss = self.sem1.compute_semantic_loss()
        sem_loss += self.sem2.compute_semantic_loss()

        return x, sem_loss


class SEM_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(SEM_Classifier, self).__init__()

        self.sem1 = SageConv(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):

        x = F.relu(self.sem1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, x, adj):

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear(x))

        return x


class GAT_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4):
        super(GAT_En2, self).__init__()

        self.attentions_1 = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention1_{}'.format(i), attention)

        self.linear_1 = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.attentions_2 = [GraphAttentionLayer(nembed, int(nembed / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention2_{}'.format(i), attention)

        self.linear_2 = nn.Linear(nembed, nembed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_1.weight, std=0.05)
        nn.init.normal_(self.linear_2.weight, std=0.05)


    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear_1(x))

        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear_2(x))

        return x


class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, nheads=4):
        super(GAT_Classifier, self).__init__()

        self.attentions = [GraphAttentionLayer(nembed, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.linear(x))
        x = self.mlp(x)

        return x


class Decoder(nn.Module):

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out


class NodeDistance(object):

    def __init__(self, adj, nclass):

        self.graph = nx.from_numpy_matrix(adj.detach().cpu().numpy())
        self.nclass = nclass

    def get_label(self):
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass-1))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1
        distance = np.triu(distance)

        return torch.LongTensor(distance) - 1

# Local-Path Prediction        
class PairwiseDistance(nn.Module):
    def __init__(self, nhid, adj, device, param):
        super(PairwiseDistance, self).__init__()

        self.linear = nn.Linear(nhid, param['dis_nclass'])
        self.agent = NodeDistance(adj, param['dis_nclass'])

        if param['dataset'] != 'cora' and os.path.exists("../save/{}/labels_dis_nclass_{}.npy".format(param['dataset'], param['dis_nclass'])):
            self.pseudo_labels = torch.LongTensor(np.load("../save/{}/labels_dis_nclass_{}.npy".format(param['dataset'], param['dis_nclass']))).to(device)
        else:
            self.pseudo_labels = self.agent.get_label().to(device)
            np.save("../save/{}/labels_dis_nclass_{}.npy".format(param['dataset'], param['dis_nclass']), self.pseudo_labels.detach().cpu().numpy())
        self.node_pairs = self.sample(self.pseudo_labels.detach().cpu().numpy() + 1.0, k=param['k_num'])

    def sample(self, labels, k):

        node_pairs = []

        for i in range(1, int(labels.max()) + 1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])

        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()

        return node_pairs[0], node_pairs[1]


    def forward(self, embeddings):
        
        embeddings_0 = embeddings[self.node_pairs[0]]
        embeddings_1 = embeddings[self.node_pairs[1]]
        embeddings = self.linear(torch.abs(embeddings_0 - embeddings_1))

        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels[self.node_pairs])

        return loss


class ClusteringMachine(object):
    def __init__(self, adj, features, nclass):

        self.adj = adj.detach().cpu().numpy()
        self.features = features.detach().cpu().numpy()
        self.nclass = nclass
        self.graph = nx.from_numpy_matrix(self.adj)
        
    def decompose(self):

        print("Metis graph clustering started ...")
        self.metis_clustering()
        self.central_nodes = self.get_central_nodes()
        self.shortest_path_to_clusters(self.central_nodes)
        self.dis_matrix = torch.FloatTensor(self.dis_matrix)

    def metis_clustering(self):

        (st, parts) = pymetis.part_graph(adjacency=self.graph, nparts=self.nclass)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):

        self.sg_nodes = {}
        self.sg_edges = {}

        for cluster in self.clusters:

            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]

            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]

        print('Number of nodes in clusters:', {x: len(y) for x, y in self.sg_nodes.items()})

    def get_central_nodes(self):

        self.general_data_partitioning()
        central_nodes = {}

        for cluster in self.clusters:

            counter = {}
            for node, _ in self.sg_edges[cluster]:
                counter[node] = counter.get(node, 0) + 1

            sorted_counter = sorted(counter.items(), key=lambda x:x[1])
            central_nodes[cluster] = sorted_counter[-1][0]

        return central_nodes

    def shortest_path_to_clusters(self, central_nodes, transform=True):

        self.dis_matrix = -np.ones((self.adj.shape[0], self.nclass))

        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])

            while q:
                node_cur, depth = q.popleft()

                if node_cur in visited:
                    continue
                visited.add(node_cur)

                if transform:
                    self.dis_matrix[node_cur][cluster] = 1 / depth
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                    
                for node_next in self.graph.neighbors(node_cur):
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2

        return self.dis_matrix

# Global-Path Prediction  
class DistanceCluster(nn.Module):
    def __init__(self, nhid, adj, features, device, param):
        super(DistanceCluster, self).__init__()

        self.linear = nn.Linear(nhid, param['clu_nclass']).to(device)
        self.cluster_agent = ClusteringMachine(adj, features, param['clu_nclass'])
        self.cluster_agent.decompose()
        self.pseudo_labels = self.cluster_agent.dis_matrix.to(device)

    def forward(self, embeddings):
        
        output = self.linear(embeddings)
        loss = F.mse_loss(output, self.pseudo_labels, reduction='mean')

        return loss


class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)

        return x