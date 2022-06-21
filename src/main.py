import csv
import nni
import time
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F

import models
import utils
import data_load
import QLearning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch):

    global last_k
    global current_k
    global last_acc
    global current_acc
    global action
    global k_record
    global Endepoch

    encoder.train()
    classifier.train()
    decoder.train()
    pairdis.train()
    clusterdis.train()

    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    if param['model'] == 'sem':
        embed, loss_sem = encoder(features, adj)
    else:
        embed = encoder(features, adj)

    if param['setting'] == 'joint' or param['setting'] == 'pre-train' or param['setting'] == 'fine-tune':
        # Feature Mixup, Label Mixup, and Edge Mixup in the semantic relation space
        embed, labels_new, idx_train_new, adj_up = utils.mixup(embed, labels, idx_train, adj=adj.detach(), up_scale=param['up_scale'], im_class_num=param['num_im_class'], scale=current_k)

        n_num = labels.shape[0]
        adj_rec = decoder(embed)
        # Three Losses for training edge predictor
        loss_rec = utils.adj_mse_loss(adj_rec[:n_num, :][:, :n_num], adj.detach(), param)
        loss_dis = pairdis(embed[:n_num])
        loss_clu = clusterdis(embed[:n_num])

        # Obtain threshold binary edges or soft continuous edges
        if param['mode'] == 'discrete_edge':
            adj_new = copy.deepcopy(adj_rec.detach())
            threshold = 0.5
            adj_new[adj_new < threshold] = 0.0
            adj_new[adj_new >= threshold] = 1.0
        else:
            adj_new = adj_rec

        adj_new = torch.mul(adj_up, adj_new)
        adj_new[:n_num, :][:, :n_num] = adj.detach()

        if param['mode'] == 'discrete_edge':
            adj_new = adj_new.detach()

    elif param['setting'] == 'embed_smote':
        embed, labels_new, idx_train_new = utils.mixup(embed, labels, idx_train, up_scale=param['up_scale'], im_class_num=param['num_im_class'])
        adj_new = adj

    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj

    output = classifier(embed, adj_new)

    # The Re-weight method assign larger weight to losses of samples on minority classes
    if param['setting'] == 're-weight':
        weight = features.new((labels.max().item() + 1)).fill_(1)
        c_largest = labels.max().item()
        avg_number = int(idx_train.shape[0] / (c_largest + 1))

        for i in range(param['num_im_class']):
            if param['up_scale'] != 0:
                weight[c_largest-i] = 1 + param['up_scale']
            else:
                chosen = idx_train[(labels == (c_largest - i))[idx_train]]
                c_up_scale = int(avg_number / chosen.shape[0]) - 1
                if c_up_scale >= 0:
                    weight[c_largest-i] = 1 + c_up_scale
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new], weight=weight)
    else:
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

    acc_train, auc_train, f1_train = utils.evaluation(output[idx_train], labels[idx_train])

    if param['setting'] == 'joint':
        loss = loss_train + loss_rec
        if param['dis_weight'] != 0:
            loss += loss_dis * param['dis_weight']
        if param['clu_weight'] != 0:
            loss += loss_clu * param['clu_weight']
        if param['model'] == 'sem':
            loss += loss_sem
        else:
            loss_sem = loss_train
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

        if epoch >= 50 and (not QLearning.isTerminal(k_record)):
            last_k, current_k, action = QLearning.Run_QL(env, RL, current_acc=current_acc, last_acc=last_acc, last_k=last_k, current_k=current_k, action=action)
            k_record.append(current_k)
            Endepoch = epoch
        else:
            k_record.append(current_k)

    # Perform joint training
    elif param['setting'] == 'pre-train':
        loss = loss_rec + 0 * loss_train
        if param['dis_weight'] != 0:
            loss += loss_dis * param['dis_weight']
        if param['clu_weight'] != 0:
            loss += loss_clu * param['clu_weight']
        if param['model'] == 'sem':
            loss += loss_sem
        else:
            loss_sem = loss_train
        loss.backward()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

    # Perform pre-training
    elif param['setting'] == 'fine-tune':
        loss = loss_train
        if param['model'] != 'sem':
            loss_sem = loss_train
        loss.backward()
        optimizer_en.step()
        optimizer_de.zero_grad()
        optimizer_cls.step()

        if epoch >= 50 and (not QLearning.isTerminal(k_record, delta_k=param['delta_k'])):
            last_k, current_k, action = QLearning.Run_QL(env, RL, current_acc=current_acc, last_acc=last_acc, last_k=last_k, current_k=current_k, action=action)
            k_record.append(current_k)
            Endepoch = epoch
        else:
            k_record.append(current_k)

    # Perform fine-tuning or training with original settings
    else:
        loss = loss_train
        loss_rec = loss_train
        loss_dis = loss_train
        loss_clu = loss_train
        if param['model'] == 'sem':
            loss += loss_sem
        else:
            loss_sem = loss_train
        loss.backward()  
        optimizer_en.step()     
        optimizer_cls.step()

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val, auc_val, f1_val = utils.evaluation(output[idx_val], labels[idx_val])
    last_acc = current_acc
    current_acc = f1_val

    print('\033[0;30;46m Epoch: {:04d}, loss_train: {:.4f}, loss_rec: {:.4f}, loss_dis: {:.4f}, loss_clu: {:.4f}, loss_sem: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}\033[0m'.format(
                        epoch, loss_train.item(), loss_rec.item(), loss_dis.item(), loss_clu.item(), loss_sem.item(), acc_train, loss_val.item(), acc_val))

    return f1_val


def test(epoch):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    pairdis.eval()
    clusterdis.eval()

    if param['model'] == 'sem':
        embed, _ = encoder(features, adj)
    else:
        embed = encoder(features, adj)
    output = classifier(embed, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test, auc_test, f1_test = utils.evaluation(output[idx_test], labels[idx_test])

    print("\033[0;30;41m [{}] Loss: {}, Accuracy: {:f}, Auc-Roc score: {:f}, Macro-F1 score: {:f}\033[0m".format(epoch, loss_test.item(), acc_test, auc_test, f1_test))

    return acc_test, auc_test, f1_test


def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()
    saved_content['pairdis'] = pairdis.state_dict()
    saved_content['clusterdis'] = clusterdis.state_dict()
    
    torch.save(saved_content, '../checkpoint/{}/{}_{}.pth'.format(param['dataset'], param['setting'], epoch))


def load_model(filename):
    loaded_content = torch.load('../checkpoint/{}/{}.pth'.format(param['dataset'], filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])
    pairdis.load_state_dict(loaded_content['pairdis'])
    clusterdis.load_state_dict(loaded_content['clusterdis'])

    print("successfully loaded: "+ filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','BlogCatalog', 'wiki-cs'])
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--num_im_class', type=int, default=3, choices=[3, 14, 10])

    parser.add_argument('--model', type=str, default='sem', choices=['sage','gcn', 'sem', 'gat'])
    parser.add_argument('--setting', type=str, default='pre-train', choices=['raw', 'pre-train', 'fine-tune', 'joint', 'over-sampling', 'smote', 'embed_smote', 're-weight'])
    parser.add_argument('--mode', type=str, default='continuous_edge', choices=['discrete_edge','continuous_edge'])
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--graph_mode', type=int, default=1)
    parser.add_argument('--dis_weight', type=float, default=1.0)
    parser.add_argument('--clu_weight', type=float, default=1.0)
    parser.add_argument('--up_scale', type=float, default=0)
    parser.add_argument('--delta_k', type=float, default=0.05)

    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--dis_nclass', type=int, default=6)
    parser.add_argument('--clu_nclass', type=int, default=15)
    parser.add_argument('--k_num', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=2010)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    random.seed(param['seed'])
    np.random.seed(param['seed'])
    torch.manual_seed(param['seed'])
    torch.cuda.manual_seed(param['seed'])

    if param['dataset'] == 'BlogCatalog':
        param['num_im_class'] = 14
        param['epochs'] = 4010
        param['clu_weight'] = 1e-3
        param['dis_nclass'] = 3
        param['clu_nclass'] = 30
        param['k_num'] = 5000
    if param['dataset'] == 'wiki-cs':
        param['num_im_class'] = 10
        param['dis_nclass'] = 5
        param['clu_nclass'] = 12
        param['k_num'] = 10000
        param['dropout'] = 0.5

    # Load Dataset
    if param['dataset'] == 'cora':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_cora(num_per_class=20, num_im_class=param['num_im_class'], im_ratio=param['im_ratio'])
    elif param['dataset'] == 'BlogCatalog':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_BlogCatalog()
    elif param['dataset'] == 'wiki-cs':
        idx_train, idx_val, idx_test, adj, features, labels = data_load.load_wiki_cs()
    else:
        print("no this dataset: {param['dataset']}")

    # For over-sampling and smote methods, they directly upsampling data in the input space
    if param['setting'] == 'over-sampling':
        features, labels, idx_train, adj = utils.src_upsample(features, labels, idx_train, adj, up_scale=param['up_scale'], im_class_num=param['num_im_class'])
    if param['setting'] == 'smote':
        features, labels, idx_train, adj = utils.src_smote(features, labels, idx_train, adj, up_scale=param['up_scale'], im_class_num=param['num_im_class'])

    # Load different bottleneck encoders and classifiers
    if param['setting'] != 'embed_smote':
        if param['model'] == 'sage':
            encoder = models.Sage_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'])
            classifier = models.Sage_Classifier(nembed=param['nhid'], nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif param['model'] == 'gcn':
            encoder = models.GCN_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'])
            classifier = models.GCN_Classifier(nembed=param['nhid'], nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif args.model == 'sem':
            encoder = models.SEM_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.SEM_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.GAT_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'], nheads=param['nhead'])
    else:
        if args.model == 'sage':
            encoder = models.Sage_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif args.model == 'gcn':
            encoder = models.GCN_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])       
        elif args.model == 'sem':
            encoder = models.SEM_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'], dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1, dropout=param['dropout'])
          
    # Load edge predictor and modules for Local-Path and Global-Path Prediction
    decoder = models.Decoder(nembed=param['nhid'], dropout=param['dropout'])
    pairdis = models.PairwiseDistance(nhid=param['nhid'], adj=adj, device=device, param=param)
    clusterdis = models.DistanceCluster(nhid=param['nhid'], adj=adj, features=features, device=device, param=param)

    # Load three optimizer for the semantic feature extractor, edge predictor, and node classifier
    optimizer_en = torch.optim.Adam(encoder.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    optimizer_de = torch.optim.Adam(list(decoder.parameters()) + list(pairdis.parameters()) + list(clusterdis.parameters()), lr=param['lr'], weight_decay=param['weight_decay'])

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    decoder = decoder.to(device)
    pairdis = pairdis.to(device)
    clusterdis = clusterdis.to(device)

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    if param['load'] is not None:
        load_model(param['load'])

    # Initialize the RL agent
    env = QLearning.GNN_env(action_value=0.05)
    RL = QLearning.QLearningTable(actions=list(range(env.n_actions)))

    last_k = 0.0
    current_k = 0.0
    last_acc = 0.0
    current_acc = 0.0
    action = None
    k_record = [0]
    Endepoch = 0

    es = 0
    f1_val_best = 0
    metric_test_val = [0, 0, 0, 0]
    metric_test_best = [0, 0, 0]

    # Run training and testing for maximum epochs with early stopping
    for epoch in range(param['epochs']):
        f1_val = train(epoch)

        if epoch % 5 == 0:
            acc_test, roc_test, f1_test = test(epoch)
            if f1_val > f1_val_best:
                f1_val_best = f1_val
                metric_test_val[0] = acc_test
                metric_test_val[1] = roc_test
                metric_test_val[2] = f1_test
                metric_test_val[3] = epoch
                es = 0
            elif param['setting'] == 'fine-tune':
                es += 1
                if es >= 20:
                    print("Early stopping!")
                    break

            if f1_test > metric_test_best[2]:
                metric_test_best[0] = acc_test
                metric_test_best[1] = roc_test
                metric_test_best[2] = f1_test

        if epoch % 500 == 0 and param['setting'] == 'pre-train':
            save_model(epoch)


    if param['setting'] == 'pre-train':
        param['setting'] = 'fine-tune'

        es = 0
        f1_val_best = 0
        metric_test_val = [0, 0, 0, 0]
        metric_test_best = [0, 0, 0]

        for epoch in range(param['epochs']):
            f1_val = train(epoch)

            if epoch % 5 == 0:
                acc_test, roc_test, f1_test = test(epoch)
                if f1_val > f1_val_best:
                    f1_val_best = f1_val
                    metric_test_val[0] = acc_test
                    metric_test_val[1] = roc_test
                    metric_test_val[2] = f1_test
                    metric_test_val[3] = epoch
                    es = 0
                else:
                    es += 1
                    if es >= 20:
                        print("Early stopping!")
                        break

                if f1_test > metric_test_best[2]:
                    metric_test_best[0] = acc_test
                    metric_test_best[1] = roc_test
                    metric_test_best[2] = f1_test

    # Save all classification results
    nni.report_final_result(metric_test_val[2])
    outFile = open('../PerformMetrics_{}.csv'.format(param['dataset']),'a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    results.append(str(f1_val_best))
    results.append(str(metric_test_val[0]))
    results.append(str(metric_test_val[1]))
    results.append(str(metric_test_val[2]))
    results.append(str(metric_test_best[0]))
    results.append(str(metric_test_best[1]))
    results.append(str(metric_test_best[2]))
    results.append(str(acc_test))
    results.append(str(roc_test))
    results.append(str(f1_test))
    results.append(str(metric_test_val[3]))
    results.append(Endepoch)
    results.append(k_record[-1])
    writer.writerow(results)

    # np.save("../result/{}/RL_process_{}.npy".format(param['dataset'], Endepoch), np.array(k_record))