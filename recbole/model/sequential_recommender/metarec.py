# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""
from curses import raw
from tqdm import tqdm
import pickle
from collections import defaultdict, Counter
from copy import copy, deepcopy
import enum
import random
from re import I
from tkinter import N
from tkinter.tix import Tree
import numpy as np
from os import path
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CLLayer

import dgl
from dgl.nn.pytorch import GATConv, GraphConv

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def graph_dual_neighbor_readout(g:dgl.DGLGraph, aug_g:dgl.DGLGraph, node_ids, features):
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)
    all_neighbors = [set(t.tolist()) for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist()) for t in foreign_neighbors.split(for_nbr_num.tolist())]
    # sample foreign neighbors
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i] for i in range(len(all_neighbors))]
    # sample civil neighbors
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t) for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat([torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat([torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]
    # insert zero vector for zero-length neighbors
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)

def graph_neighbor_readout(g:dgl.DGLGraph, node_ids, features):
    # Readout the neighbors of the given node.
    _, neighbors = g.out_edges(node_ids)
    neighbor_nums = g.out_degrees(node_ids)
    neighbor_features = features[neighbors].split(neighbor_nums.tolist())
    neighbor_features = [t.mean(dim=0) for t in neighbor_features]
    return torch.stack(neighbor_features, dim=0)

def hierarchical_infomax(item_embedding, isim_embedding, iadj_embedding):
    def score(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)
    def row_shuffle(embedding):
        """
        Randomly shuffle the rows of the embedding matrix.
        """
        perm = torch.randperm(embedding.shape[0])
        return embedding[perm]
    def row_column_shuffle(embedding):
        """
        Randomly shuffle the rows and columns of the embedding matrix.
        """
        idx = torch.randperm(embedding.nelement())
        embedding = embedding.view(-1)[idx].view(embedding.size())
        return embedding

    # Local Infomax with item sim embedding
    pos = score(item_embedding, isim_embedding)
    neg1 = score(row_shuffle(item_embedding), isim_embedding)
    neg2 = score(row_column_shuffle(item_embedding), isim_embedding)
    local_loss = torch.log(torch.sigmoid(pos-neg1))-torch.log(torch.sigmoid(neg1-neg2))

    # Global Infomax with item adj embedding
    pos = score(item_embedding, iadj_embedding)
    neg1 = score(row_shuffle(item_embedding), iadj_embedding)
    neg2 = score(row_column_shuffle(item_embedding), iadj_embedding)
    global_loss = torch.log(torch.sigmoid(pos-neg1))-torch.log(torch.sigmoid(neg1-neg2))

    return local_loss, global_loss


def collect_user_sequence(iter_data):
    # Collect user sequences from the dataset.
    user_sequences = defaultdict(list)
    for interaction in iter_data:
        item_seq = interaction["item_id_list"].tolist()
        last_item = interaction["item_id"].tolist()
        seq_length = interaction["item_length"].tolist()
        uid = interaction["session_id"].tolist()
        for i, u in enumerate(uid):
            user_sequences[u].extend(item_seq[i][:seq_length[i]] + [last_item[i]])
    return user_sequences


def build_usim(user_sequences, n_items):
    row = []
    col = []
    for usr, itms in user_sequences.items():
        col.extend(list(itms))
        row.extend([usr]*len(itms))
    row = np.array(row)
    col = np.array(col)
    feature_mtx = csr_matrix(([1]*len(row),(row,col)),shape=(max(user_sequences.keys())+1, n_items+1))
    similarity = cosine_similarity(feature_mtx)
    # n_friends + 1
    return similarity.argsort()[:,-(4+1):]


def graph_mask(g : dgl.DGLGraph, mask_indices):
    edge_ids = g.edge_ids(mask_indices[0], mask_indices[1])
    masked_g = deepcopy(g)
    masked_g.edata["w"][edge_ids] = 0.
    return masked_g


def graph_augment(g: dgl.DGLGraph, batch, mask_token):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    node_indicies_a = []
    node_indicies_b = []
    for seq in batch:
        for i in range(len(seq)):
            if seq[i] == 0:
                break
            if seq[i] == mask_token:
                continue
            if i > 0 and seq[i-1] != mask_token:
                node_indicies_a.append(seq[i])
                node_indicies_b.append(seq[i-1])
            if seq[i+1] != 0 and seq[i+1] != mask_token:
                node_indicies_a.append(seq[i])
                node_indicies_b.append(seq[i+1])
    node_indicies_a = torch.tensor(node_indicies_a, dtype=torch.int64, device=g.device)
    node_indicies_b = torch.tensor(node_indicies_b, dtype=torch.int64, device=g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g

def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8, device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=2):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, out_dim, num_heads, feat_drop=0.2, attn_drop=0.2)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(out_dim * num_heads, out_dim, 1, feat_drop=0.2, attn_drop=0.2)

    def forward(self, graph, feature):
        origin_w, graph = graph_dropout(graph, 0.7)
        h1 = self.layer1(graph, feature)
        h1 = F.elu(h1)
        h2 = self.layer2(graph, h1.view(h1.size(0), -1))
        h1 = torch.mean(h1, dim=1)

        # recover edge weight
        graph.edata['w'] = origin_w
        return torch.mean(torch.stack([h1, h2]), dim=0)

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.layer = GraphConv(in_dim, out_dim, weight=False, bias=False, allow_zero_in_degree=True)

    def forward(self, graph, feature):
        origin_w, graph = graph_dropout(graph, 0.7)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            # TODO:这里需不需要dropout??
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb

class MetaRec(SequentialRecommender):

    def __init__(self, config, dataset, iter_traindata):
        super(MetaRec, self).__init__(config, dataset)

        self.config = config
        # load parameters info
        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.weight_mlp_3x = nn.Sequential(
            # nn.Linear(3*self.hidden_size, 3*self.hidden_size),
            # nn.ReLU(),
            nn.Linear(3*self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.weight_mlp_2x = nn.Sequential(
            nn.Linear(2*self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Contrastive Learning
        self.contrastive_learning_layer = CLLayer(self.hidden_size, tau=0.2)

        # Fusion Attn
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # task length lable
        self.task_length_label = nn.Embedding(6, self.hidden_size)
        nn.init.normal_(self.task_length_label.weight, std=0.02)

        # Global Graph Learning
        # self.cooccurrence_graph = self._build_graph_dgl()
        self.item_adjgraph = self.__build_graph(iter_traindata)
        self.user_sequences = collect_user_sequence(iter_traindata)
        self.item_simgraph = self.__build_isim()
        
        self.gcn = GCN(self.hidden_size, self.hidden_size)
        
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.loss_fct = nn.CrossEntropyLoss()

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")

        # parameters initialization
        self.apply(self._init_weights)

        # dropout
        F.dropout(self.item_embedding.weight.data, p=0.5, inplace=True)
    
    def _subgraph_agreement(self, aug_g, raw_output_all, raw_output_seq, valid_items_flatten):
        # here it firstly removes items of the sequence in the cooccurrence graph, and then performs the gnn aggregation, and finally calculates the item-wise agreement score.
        aug_output = self.gcn_forward(g = aug_g)
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(self.item_adjgraph, aug_g, valid_items_flatten, raw_output_all)
        neighbor_readout = graph_neighbor_readout(aug_g, valid_items_flatten, aug_output)
        view1_sim = F.cosine_similarity(raw_output_seq, self.weight_mlp_2x(torch.cat((aug_output[valid_items_flatten], neighbor_readout), dim=1)), eps=1e-12)
        view2_sim = F.cosine_similarity(civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        agreement = (view1_sim+view2_sim)/2
        agreement = torch.exp(agreement)
        agreement = (agreement - agreement.min()) / (agreement.max() - agreement.min())
        agreement = (self.config["weight_mean"] / agreement.mean()) * agreement
        return agreement

    def __build_graph(self, iter_data):
        graph_file = "/data1/yuh/meta-rec/dataset/"+self.config["dataset"]+"/dgl_graph.bin"
        try:
            g = dgl.load_graphs(graph_file, [0])
            print("loading graph from DGL binary file...")
            return g[0][0].to(self.device)
        except:
            print("constructing DGL graph...")
            adj_dict = defaultdict(list)
            for batch in iter_data:
                item_seq = batch[self.ITEM_SEQ].tolist()
                last_item = batch[self.ITEM_ID].tolist()
                for seq_id, seq in enumerate(item_seq):
                    for i in range(len(seq)):
                        # if seq[i] == 25722:
                        #     print("f")
                        if seq[i] == 0:
                            break
                        if i>0:
                            adj_dict[seq[i]].append(seq[i-1])
                            adj_dict[seq[i-1]].append(seq[i])
                        if seq[i+1] != 0:
                            adj_dict[seq[i]].append(seq[i+1])
                            adj_dict[seq[i+1]].append(seq[i])
                    # i 指向了第一个0
                    adj_dict[seq[i-1]].append(last_item[seq_id])
                    adj_dict[last_item[seq_id]].append(seq[i-1])
            # print(adj_dict[25722])
            cols = []
            rows = []
            values = []
            for item in adj_dict:
                adj = adj_dict[item]
                adj_count = Counter(adj)

                rows.extend([item]*len(adj_count))
                cols.extend(adj_count.keys())
                values.extend(adj_count.values())

            adj_mat = csr_matrix((values, (rows, cols)), shape=(self.n_items + 1, self.n_items + 1))
            adj_mat = adj_mat.tolil()
            adj_mat.setdiag(np.ones((self.n_items + 1,)))
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            g = dgl.from_scipy(norm_adj, 'w', device=self.device)
            g.edata['w'] = g.edata['w'].float()
            print("saving DGL graph to binary file...")
            dgl.save_graphs("/data1/yuh/meta-rec/dataset/"+self.config["dataset"]+"/dgl_graph.bin", [g])
            return g

    def __build_isim(self):
        k=self.config["sim_group"]
        graph_file = "/data1/yuh/meta-rec/dataset/"+self.config["dataset"]+"/dgl_isim_graph.bin"
        try:
            g = dgl.load_graphs(graph_file, [0])
            print("loading isim graph from DGL binary file...")
            return g[0][0].to(self.device)
        except:
            print("building isim graph...")
            row = []
            col = []
            for usr, itms in self.user_sequences.items():
                col.extend(list(itms))
                row.extend([usr]*len(itms))
            row = np.array(row)
            col = np.array(col)
            # n_users, n_items
            cf_graph = csr_matrix(([1]*len(row),(row,col)),shape=(max(self.user_sequences.keys())+1, self.n_items+1), dtype=np.float32)
            similarity = cosine_similarity(cf_graph.transpose())
            # filter topk connections
            sim_items_slices = []
            sim_weights_slices = []
            i = 0
            while i < similarity.shape[0]:
                similarity = similarity[i:, :]
                sim = similarity[:256, :]
                sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
                sim_weights = np.take_along_axis(sim, sim_items, axis=1)
                sim_items_slices.append(sim_items)
                sim_weights_slices.append(sim_weights)
                i = i + 256
            sim = similarity[256:, :]
            sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
            sim_weights = np.take_along_axis(sim, sim_items, axis=1)
            sim_items_slices.append(sim_items)
            sim_weights_slices.append(sim_weights)

            sim_items = np.concatenate(sim_items_slices, axis=0)
            sim_weights = np.concatenate(sim_weights_slices, axis=0)
            row = []
            col = []
            for i in range(len(sim_items)):
                row.extend([i]*len(sim_items[i]))
                col.extend(sim_items[i])
            values = sim_weights / sim_weights.sum(axis=1, keepdims=True)
            values = np.nan_to_num(values).flatten()
            adj_mat = csr_matrix((values, (row, col)), shape=(self.n_items + 1, self.n_items + 1))
            g = dgl.from_scipy(adj_mat, 'w', device=self.device)
            g.edata['w'] = g.edata['w'].float()
            print("saving isim graph to binary file...")
            dgl.save_graphs("/data1/yuh/meta-rec/dataset/"+self.config["dataset"]+"/dgl_isim_graph.bin", [g])
            return g

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        # 0在后面的mask, 和原版BERT4Rec不同.
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len
        return sequence

    def reconstruct_train_data(self, item_seq, last_item = None, max_mask_num = None, generate_graph_mask = False):
        """
        Mask item sequence for training.
        """

        device = item_seq.device
        batch_size = item_seq.size(0)
        zero_last_padding = torch.zeros((item_seq.size(0), 1), device=item_seq.device, dtype=torch.long)
        item_seq = torch.cat((item_seq, zero_last_padding), dim=1)
        seq_lens = torch.count_nonzero(item_seq, dim=1)

        if max_mask_num is None:
            max_mask_num = self.max_seq_length

        if last_item is not None:
            item_seq[list(range(batch_size)), seq_lens] = last_item

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        # node pairs for indexing edge idx.
        masked_graph_indicies_a = []
        masked_graph_indicies_b = []

        for i, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0 or len(pos_item) >= max_mask_num:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token

                    # generate graph edge index for graph masking here:
                    if generate_graph_mask:
                        if index_id < len(instance) - 1 and instance[index_id + 1] != 0:
                            masked_graph_indicies_a.append(item)
                            masked_graph_indicies_b.append(instance[index_id + 1])
                        if index_id > 0 and instance[index_id - 1] != self.mask_token:
                            masked_graph_indicies_a.append(item)
                            masked_graph_indicies_b.append(instance[index_id - 1])

            # 每个序列都必须被mask
            if len(pos_item) == 0:
                item = 0 
                while item == 0:
                    index_id = random.randint(0, seq_lens[i])
                    item = instance[index_id]
                pos_item.append(item)
                masked_sequence[index_id] = self.mask_token

                # generate graph edge index for graph masking here:
                if generate_graph_mask:
                    if index_id < len(instance) - 1 and instance[index_id + 1] != 0:
                        masked_graph_indicies_a.append(item)
                        masked_graph_indicies_b.append(instance[index_id + 1])
                    if index_id > 0 and instance[index_id - 1] != self.mask_token:
                        masked_graph_indicies_a.append(item)
                        masked_graph_indicies_b.append(instance[index_id - 1])
            
            masked_item_sequence.append(masked_sequence)
            pos_items.extend(pos_item)

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B*mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)

        if generate_graph_mask:
            masked_graph_indicies_a = torch.tensor(masked_graph_indicies_a, dtype=torch.long, device=device)
            masked_graph_indicies_b = torch.tensor(masked_graph_indicies_b, dtype=torch.long, device=device)
        return masked_item_sequence, pos_items, (masked_graph_indicies_a, masked_graph_indicies_b)

    def reconstruct_test_data(self, item_seq, item_seq_len, generate_label = False):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        device = item_seq.device
        batch_size = item_seq.size(0)
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        seq_lens = torch.count_nonzero(item_seq, dim=1)
        length_labels = []
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token

            if generate_label:
                label = seq_lens[batch_id] // ( self.max_seq_length // 5)
                length_labels.append(label)
        
        if generate_label:
            length_labels = torch.tensor(length_labels, dtype=torch.long, device=device)
            return item_seq, length_labels

        return item_seq

    def meta_forward(self, item_seq, local_parameters=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        if local_parameters is not None:
            position_embedding = local_parameters["position_embedding.weight"][position_ids]
            item_emb = local_parameters["item_embedding.weight"][item_seq]
        else:
            position_embedding = self.position_embedding(position_ids)
            item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B L H]

    def gcn_forward(self, g = None):
        item_emb = self.item_embedding.weight
        item_emb = self.dropout(item_emb)
        if g is not None:
            light_out = self.gcn(g, item_emb)
        else:
            light_out = self.gcn(self.item_adjgraph, item_emb)
        return self.layernorm(light_out+item_emb)

    def gat_forward(self):
        item_emb = self.item_embedding.weight
        gat_out = self.gat(self.item_adjgraph, item_emb)
        # TODO: 这里需不需要LN??
        return self.layernorm(gat_out+item_emb)

    def multitask_forward(self, item_seq, task_labels):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        # b, 1, h
        task_embeddings = self.task_length_label(task_labels).unsqueeze(1)
        input_emb = torch.cat((input_emb, task_embeddings), dim=1)
        extended_attention_mask = self.get_attention_mask(item_seq, task_label=True)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output[:, 1:, :]  # [B L H]


    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        extended_attention_mask = self.get_attention_mask(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        if self.config["graphcl_enable"]:
            return self.calculate_loss_graphcl(interaction)
        elif self.config["him_enable"]:
            return self.calculate_loss_him(interaction)

        item_seq = interaction[self.ITEM_SEQ]
        last_item = interaction[self.ITEM_ID]
        self.max_seq_length = item_seq.shape[1] + 1
        if self.config["multitask_enable"]:
            masked_item_seq, pos_items, task_labels = self.reconstruct_train_data(item_seq, last_item=last_item, generate_multitask_label=True)
            seq_output = self.multitask_forward(masked_item_seq, task_labels)
        else:
            masked_item_seq, pos_items = self.reconstruct_train_data(item_seq, last_item=last_item)
            seq_output = self.forward(masked_item_seq)

        masked_index = (masked_item_seq==self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss

    def calculate_loss_meta(self, item_seq, local_parameters=None):
        item_seq = item_seq.to(self.device)
        self.max_seq_length = self.config["meta_task_lengths"][-1]
        masked_item_seq, pos_items = self.reconstruct_train_data(item_seq, max_mask_num=1)

        seq_output = self.meta_forward(masked_item_seq, local_parameters)
        masked_index = (masked_item_seq==self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(loss)
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            print(logits)
            print(pos_items)
            input()
        return loss
    
    # def calculate_loss_him(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     last_item = interaction[self.ITEM_ID]
    #     self.max_seq_length = item_seq.shape[1] + 1
    #     masked_item_seq, pos_items, graph_mask_indices = self.reconstruct_train_data(item_seq, last_item=last_item, generate_graph_mask=True)
    #     seq_output = self.forward(masked_item_seq)
    #     valid_item_seq = (masked_item_seq != self.mask_token) & (masked_item_seq != 0)

    #     # graph view
    #     masked_g = graph_mask(self.item_adjgraph, graph_mask_indices)
    #     iadj_graph_output = self.gcn_forward(masked_g)[masked_item_seq]
    #     isim_graph_output = self.gcn_forward(self.item_simgraph)[masked_item_seq]

    #     # First-stage CL, providing CL weights
    #     # CL weights from augmentation
    #     mainstream_weights = self._subgraph_agreement(masked_item_seq, iadj_graph_output, valid_item_seq)
    #     personlization_weights = mainstream_weights.max() - mainstream_weights

    #     # Hierarchical Infomax with weights
    #     local_loss, global_loss = hierarchical_infomax(seq_output[valid_item_seq], isim_graph_output[valid_item_seq], iadj_graph_output[valid_item_seq])
    #     # ssl_loss = self.config['him_coefficient'] * (personlization_weights * local_loss).sum()
    #     ssl_loss = self.config['him_coefficient'] * (mainstream_weights * global_loss + personlization_weights * local_loss).sum()
    #     # selecting masked index only for training
    #     masked_index = (masked_item_seq==self.mask_token)
    #     # [mask_num, H]
    #     # Fusion After CL
    #     seq_output = seq_output[masked_index]

    #     test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
    #     logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

    #     loss = self.loss_fct(logits, pos_items)

    #     if torch.isnan(loss):
    #         print(masked_item_seq.tolist())
    #         print(masked_index.tolist())
    #         input()
    #     return loss + ssl_loss

    def calculate_loss_graphcl(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        last_item = interaction[self.ITEM_ID]
        self.max_seq_length = item_seq.shape[1] + 1

        masked_item_seq, pos_items, graph_mask_indices = self.reconstruct_train_data(item_seq, last_item=last_item, generate_graph_mask=True)
        seq_output = self.forward(masked_item_seq)
        valid_items_indicies = (masked_item_seq != self.mask_token) & (masked_item_seq != 0)
        valid_items_flatten = masked_item_seq[valid_items_indicies]
        # graph view
        masked_g = graph_mask(self.item_adjgraph, graph_mask_indices)
        aug_g = graph_augment(self.item_adjgraph, masked_item_seq, mask_token=self.mask_token)
        iadj_graph_output_raw = self.gcn_forward(masked_g)
        iadj_graph_output_seq = iadj_graph_output_raw[valid_items_flatten]
        isim_graph_output_seq = self.gcn_forward(self.item_simgraph)[valid_items_flatten]

        # First-stage CL, providing CL weights
        # CL weights from augmentation
        mainstream_weights = self._subgraph_agreement(aug_g, iadj_graph_output_raw, iadj_graph_output_seq, valid_items_flatten)
        # free memory
        del iadj_graph_output_raw
        
        expected_weights_distribution = torch.normal(self.config["weight_mean"], 0.1, size=mainstream_weights.size()).to(self.device)
        kl_loss = self.config["kl_weight"] * F.kl_div(F.log_softmax(mainstream_weights, dim=0), expected_weights_distribution, reduction= "batchmean")
        personlization_weights = mainstream_weights.max() - mainstream_weights

        # save weights for visualization
        # user_lengths = torch.count_nonzero(valid_item_seq, dim=1).tolist()
        # user_mainstream_weights = mainstream_weights.split(user_lengths, dim=0)
        # users = interaction["session_id"]
        # file_name = "user_weights_data.pkl"
        # if path.exists(file_name):
        #     with open(file_name, "rb") as f:
        #         user_weights_data = pickle.load(f)
        # else:
        #     user_weights_data = dict()
        # for user, user_weights in zip(users, user_mainstream_weights):
        #     user_weights_data[user.tolist()] = user_weights.tolist()
        # with open(file_name, "wb") as f:
        #     pickle.dump(user_weights_data, f)

        # contrastive learning
        if self.config["cl_ablation"] == "adj":
            cl_loss_adj = self.contrastive_learning_layer.semi_loss(seq_output[valid_items_indicies], iadj_graph_output_seq)
            cl_loss = (self.config["graphcl_coefficient"] * (mainstream_weights * cl_loss_adj)).sum()
        elif self.config["cl_ablation"] == "a2s":
            cl_loss_a2s = self.contrastive_learning_layer.semi_loss(iadj_graph_output_seq, isim_graph_output_seq)
            cl_loss = (self.config["graphcl_coefficient"] * (personlization_weights * cl_loss_a2s)).sum()
        elif self.config["cl_ablation"] == "none":
            cl_loss = 0
        elif self.config["cl_ablation"] == "full":
            cl_loss_adj = self.contrastive_learning_layer.semi_loss(seq_output[valid_items_indicies], iadj_graph_output_seq)
            cl_loss_a2s = self.contrastive_learning_layer.semi_loss(iadj_graph_output_seq, isim_graph_output_seq)
            cl_loss = (self.config["graphcl_coefficient"] * (mainstream_weights * cl_loss_adj + personlization_weights * cl_loss_a2s)).sum()
        # selecting masked index only for training
        masked_index = (masked_item_seq==self.mask_token)
        # [mask_num, H]
        # Fusion After CL
        seq_output = seq_output[masked_index]
        if self.config["view_fusion"]:
            graph_output = graph_output[masked_index]
            # 2, N_mask, dim
            mixed_x = torch.stack((seq_output, graph_output), dim=0)
            weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
            # 2, N_mask, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            mixed_x = (mixed_x*score).sum(0)
        # [item_num, H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss + cl_loss + kl_loss

    
    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        if self.config["multitask_enable"]:
            item_seq, task_labels = self.reconstruct_test_data(item_seq, item_seq_len, generate_label=True)
            seq_output = self.multitask_forward(item_seq, task_labels)
        else:
            item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
            seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]

        # # view fusion if applicable
        # if self.config["graphcl_enable"]:
        #     # graph view
        #     graph_output = self.gather_indexes(self.gcn_forward(self.item_simgraph)[item_seq], item_seq_len)
        #     # 2, N_mask, dim
        #     seq_output = torch.stack((seq_output, graph_output), dim=0)
        #     weights = (torch.matmul(seq_output, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        #     # 2, N_mask, 1
        #     score = F.softmax(weights, dim=0).unsqueeze(-1)
        #     seq_output = (seq_output*score).sum(0)
        test_item_emb = self.item_embedding(test_item) # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(1,2)).squeeze()
        return scores

