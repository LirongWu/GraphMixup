import random
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn.functional as F


def evaluation(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    accuracy = correct.item() * 1.0 / len(labels)

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy(), average='macro')

    macro_F = f1_score(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(), average='macro')

    return accuracy, auc_score, macro_F

# Interpolation in the input space
def src_upsample(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale

        for j in range(c_up_scale):
            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            
        if up_scale_rest != 0:
            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()

# Interpolation in the embedding space
def src_smote(features, labels, idx_train, adj, up_scale=1.0, im_class_num=3):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    chosen = None
    new_features = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / new_chosen.shape[0]) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/new_chosen.shape[0] - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale
            
        for j in range(c_up_scale):

            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
        
        if up_scale_rest != 0.0 and int(new_chosen.shape[0] * up_scale_rest)>=1:

            num = int(new_chosen.shape[0] * up_scale_rest)
            new_chosen = new_chosen[:num]
            chosen_embed = features[new_chosen, :]

            distance = squareform(pdist(chosen_embed.detach()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
                
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num)))
    new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
    new_adj[adj.shape[0]:, :adj.shape[0]] = adj[chosen,:]
    new_adj[:adj.shape[0], adj.shape[0]:] = adj[:,chosen]
    new_adj[adj.shape[0]:, adj.shape[0]:] = adj[chosen,:][:,chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_train_append = idx_train.new(np.arange(adj.shape[0], adj.shape[0] + add_num))

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return features, labels, idx_train, new_adj.detach()

# Mixup in the semantic relation space
def mixup(embed, labels, idx_train, adj=None, up_scale=1.0, im_class_num=3, scale=0.0):

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]

        if up_scale == 0:
            c_up_scale = int(avg_number / chosen.shape[0] + scale) - 1
            if c_up_scale >= 0:
                up_scale_rest = avg_number/chosen.shape[0] + scale - 1 - c_up_scale
            else:
                c_up_scale = 0
                up_scale_rest = 0
            # print(round(scale, 2), round(c_up_scale, 2), round(up_scale_rest, 2))
        else:
            c_up_scale = int(up_scale)
            up_scale_rest = up_scale - c_up_scale
            

        for j in range(c_up_scale):

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()

            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0]))

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_new), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

        if up_scale_rest != 0.0:

            num = int(chosen.shape[0] * up_scale_rest)
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().cpu().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()

            new_embed = embed[chosen, :] + (embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = idx_train.new(np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0]))

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_new), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        if adj_new is not None:
            add_num = adj_new.shape[0]
            new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
            new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
            new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

            return embed, labels, idx_train, new_adj.detach()
        else:
            return embed, labels, idx_train, adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt, param):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    if param['dataset'] == 'cora':
        return loss * 1e-3
    else:
        return loss / adj_tgt.shape[0]




