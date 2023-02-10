
import os
from ast import arg
from recbole import data
import setproctitle
setproctitle.setproctitle("EXP@DCRec")
from collections import defaultdict, Counter
import argparse
import logging
from logging import getLogger
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from recbole.config import Config
from sklearn.metrics.pairwise import cosine_similarity
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from recbole.data.interaction import Interaction

# torch.autograd.set_detect_anomaly(True)

def build_adj_graph(dataset):
    import dgl
    graph_file = dataset.config['data_path']+"/adj_graph.bin"
    user_edges_file = dataset.config['data_path']+"/user_edges.pkl.zip"
    try:
        g = dgl.load_graphs(graph_file, [0])
        user_edges = pd.read_pickle(user_edges_file)
        print("loading graph from DGL binary file...")
        return g[0][0], user_edges
    except:
        print("constructing DGL graph...")
        item_adj_dict = defaultdict(list)
        item_edges_of_user = dict()
        inter_feat = dataset.inter_feat
        for line in range(len(inter_feat)):
            item_edges_a, item_edges_b = [], []
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            last_item = inter_feat[dataset.iid_field][line].item()
            for i in range(seq_len):
                if i > 0:
                    item_adj_dict[item_seq[i]].append(item_seq[i-1])
                    item_adj_dict[item_seq[i-1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i-1])
                if i+1 < seq_len:
                    item_adj_dict[item_seq[i]].append(item_seq[i+1])
                    item_adj_dict[item_seq[i+1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i+1])

            item_adj_dict[item_seq[-1]].append(last_item)
            item_adj_dict[last_item].append(item_seq[-1])
            item_edges_a.append(item_seq[-1])
            item_edges_b.append(last_item)

            item_edges_of_user[uid] = (np.asarray(item_edges_a), np.asarray(item_edges_b))
        item_edges_of_user = pd.DataFrame.from_dict(item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
        item_edges_of_user.to_pickle(user_edges_file)
        cols = []
        rows = []
        values = []
        for item in item_adj_dict:
            adj = item_adj_dict[item]
            adj_count = Counter(adj)

            rows.extend([item]*len(adj_count))
            cols.extend(adj_count.keys())
            values.extend(adj_count.values())

        adj_mat = csr_matrix((values, (rows, cols)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(np.ones((dataset.item_num + 1,)))
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        g = dgl.from_scipy(norm_adj, 'w')
        g.edata['w'] = g.edata['w'].float()
        print("saving DGL graph to binary file...")
        dgl.save_graphs(graph_file, [g])
        return g, item_edges_of_user

def build_sim_graph(dataset, k):
    import dgl
    graph_file = dataset.config['data_path']+f"/sim_graph_g{k}.bin"
    try:
        g = dgl.load_graphs(graph_file, [0])
        print("loading isim graph from DGL binary file...")
        return g[0][0]
    except:
        print("building isim graph...")
        row = []
        col = []
        inter_feat = dataset.inter_feat
        for line in range(len(dataset.inter_feat)):
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            col.extend(item_seq)
            row.extend([uid]*seq_len)
        row = np.array(row)
        col = np.array(col)
        # n_users, n_items
        cf_graph = csr_matrix(([1]*len(row), (row, col)), shape=(
            dataset.user_num+1, dataset.item_num+1), dtype=np.float32)
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
        adj_mat = csr_matrix((values, (row, col)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        g = dgl.from_scipy(adj_mat, 'w')
        g.edata['w'] = g.edata['w'].float()
        print("saving isim graph to binary file...")
        dgl.save_graphs(graph_file, [g])
        return g

def sequential_augmentation(dataset):
    import torch
    max_item_list_len = dataset.config['MAX_ITEM_LIST_LENGTH']
    old_data = dataset.inter_feat
    new_data = {dataset.uid_field:[],
                dataset.iid_field:[],
                dataset.item_list_length_field:[],
                dataset.item_id_list_field:[],}
    for i in range(len(old_data)):
        seq = old_data[dataset.item_id_list_field][i]
        uid = old_data[dataset.uid_field][i].item()
        seq_len = old_data[dataset.item_list_length_field][i]
        new_data[dataset.uid_field].append(uid)
        new_data[dataset.iid_field].append(old_data[dataset.iid_field][i].item())
        new_data[dataset.item_list_length_field].append(seq_len)
        new_data[dataset.item_id_list_field].append(seq)
        seq = seq[:seq_len]
        for end_point in range(1, seq_len):
            new_seq = seq[:end_point]
            new_truth = seq[end_point].item()
            new_seq_len = len(new_seq)
            new_seq = torch.cat((new_seq, torch.zeros(max_item_list_len - new_seq_len, dtype=torch.long)), dim=0)

            new_data[dataset.uid_field].append(uid)
            new_data[dataset.iid_field].append(new_truth)
            new_data[dataset.item_list_length_field].append(new_seq_len)
            new_data[dataset.item_id_list_field].append(new_seq)
    
    new_data[dataset.item_id_list_field] = torch.stack(new_data[dataset.item_id_list_field], dim=0)
    new_data[dataset.item_list_length_field] = torch.tensor(new_data[dataset.item_list_length_field], dtype=torch.long)
    new_data[dataset.uid_field] = torch.tensor(new_data[dataset.uid_field], dtype=torch.long)
    new_data[dataset.iid_field] = torch.tensor(new_data[dataset.iid_field], dtype=torch.long)
    dataset.inter_feat = (Interaction(new_data))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='DCRec', help='Experiment Description.')
    parser.add_argument('--model', '-m', type=str, default='DCRec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='reddit', help='Benchmarks for session-based rec.')
    parser.add_argument('--log', '-l', type=int, default=0, help='record logs or not')
    parser.add_argument('--log_name', '-ln', type=str, default=None)
    parser.add_argument('--save', '-s', type=int, default=0, help='save models or not')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--graphcl_enable', type=int, default=1)
    parser.add_argument('--ablation', type=str, default="full")
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        "seed":2020,
        "reproducibility":0,
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        'neg_sampling': None,
        'benchmark_filename': ['train', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [1, 5, 10],
        'metrics': ['Recall', 'NDCG'],
        'valid_metric': 'Recall@1',
        'eval_args':{
            'mode':'pop100',
            'order':'TO'
            },
        'gpu_id':args.gpu_id,
        "MAX_ITEM_LIST_LENGTH":50,
        "train_batch_size":args.batch_size,
        "eval_batch_size":256,
        "stopping_step":20,
        "fast_sample_eval":1,

        "hidden_dropout_prob": 0.3,
        "attn_dropout_prob": 0.3,
        
        # Graph Args:
        "graph_dropout_prob":0.3,
        "graphcl_enable": args.graphcl_enable,
        "graphcl_coefficient":1e-4,
        "cl_ablation":args.ablation,
        "graph_view_fusion":1,
        "cl_temp":1
    }
    # BEST SETTINGS
    if args.dataset == "reddit":
        config_dict["train_batch_size"] = 256

        config_dict["graphcl_coefficient"] = 1

        config_dict["weight_mean"] = 0.6
        config_dict["sim_group"] = 4
        config_dict["kl_weight"] = 1

    elif args.dataset == "beauty" or args.dataset == "sports":
        config_dict["graphcl_coefficient"] = 1e-1
        config_dict["graph_dropout_prob"] = 0.5
        config_dict["hidden_dropout_prob"]= 0.5
        config_dict["attn_dropout_prob"]  = 0.5
        config_dict["kl_weight"] = 1e-2

        if args.dataset == "beauty":
            config_dict['train_batch_size'] = 256
            config_dict["sim_group"] = 4
            config_dict["weight_mean"] = 0.6
            config_dict["cl_temp"] = 0.6
        elif args.dataset == "sports":
            config_dict['train_batch_size'] = 64
            config_dict["sim_group"] = 4
            config_dict["weight_mean"] = 0.6
            config_dict["cl_temp"] = 0.2
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, args.log, logfilename=args.log_name)
    logger = getLogger()
    logger.info(f"PID: {os.getpid()}")
    logger.info(args.desc)
    logger.info("\n")

    prior_args = dict()
    keywords = ["graph", "weight_mean", "kl", "sim_group", "dup"]
    for c in config_dict:
        for k in keywords:
            if c.startswith(k):
                prior_args[c] = config_dict[c]
        else:
            if c=="eval_args":
                prior_args[c] = config_dict[c]["mode"]
    prior_args = "\n".join([k+": "+str(v) for k,v in prior_args.items()])+"\n"
    logger.info(prior_args)

    logger.info(config)

    try:
        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_dataset, test_dataset = dataset.build()
        adj_graph, user_edges = build_adj_graph(train_dataset)
        sim_graph = build_sim_graph(train_dataset, config_dict["sim_group"])
        sequential_augmentation(train_dataset)
        train_sampler, _, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
        if args.validation:
            train_dataset.shuffle()
            new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
            train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
            test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
        else:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
            test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

        # model loading and initialization
        external_data = {
            "adj_graph": adj_graph,
            "sim_graph": sim_graph,
            "user_edges": user_edges
        }
        model = get_model(config['model'])(config, train_data.dataset, external_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config["model"])(config, model)

        # model training and evaluation
        test_score, test_result = trainer.fit(
            train_data, test_data, saved=args.save, show_progress=config['show_progress']
        )
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    except Exception as e:
        logger.exception(e)
        raise e