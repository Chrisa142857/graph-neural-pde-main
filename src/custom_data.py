import os, random
import os.path as osp
import sys
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import index_to_mask, to_dense_batch 

class custom_BOLD(InMemoryDataset):

    def __init__(self, root: str, name: str):
        root = root.replace('custom', '')
        self.name = name
        super().__init__(root)
        x, y, edges = read_BOLD_signal(root)
        self.N = len(x)
        self.index = [i for i in range(self.N)]
        random.shuffle(self.index)
        self.index = torch.LongTensor(self.index)
        self.node_nums = [len(x[i]) for i in range(len(x))]
        self.data = collate(x, y, edges)
        self.cross_val_fold_n = 5
        self.current_fold = 0
        # self.num_classes = 1
        self.next_fold()

    @property
    def num_classes(self) -> int:
        return 1

    def next_fold(self):
        split_pt1 = int(self.current_fold * self.N * (1/self.cross_val_fold_n))
        split_pt2 = int((self.current_fold+1) * self.N * (1/self.cross_val_fold_n))
        train_index = torch.cat([self.index[:split_pt1], self.index[split_pt2:]])
        val_index = self.index[split_pt1:split_pt2]
        train_mask = index_to_mask(train_index, size=self.N)
        train_mask = torch.cat([train_mask[i].repeat(self.node_nums[i]) for i in range(self.N)])
        val_mask = index_to_mask(val_index, size=self.N)
        val_mask = torch.cat([val_mask[i].repeat(self.node_nums[i]) for i in range(self.N)])
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.current_fold += 1


def adj_to_edge_index(adj_mat):
    edge_index = np.where(adj_mat > 0)
    edge_attr = adj_mat[edge_index]
    return np.array(edge_index), edge_attr

def collate(x, y, edges):
    all_edge = edges[0][0]
    all_edge_attr = edges[0][1]
    for i in range(1, len(x)):
        edge = edges[i][0]
        edge_attr = edges[i][1]
        all_edge = np.concatenate([all_edge, edge+(x[i].shape[0])*i], axis=1)
        all_edge_attr = np.concatenate([all_edge_attr, edge_attr])
    x = np.concatenate(x)[..., np.newaxis]
    y = np.concatenate(y)[..., np.newaxis]
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(all_edge), edge_attr=torch.from_numpy(all_edge_attr), y=torch.from_numpy(y))
    data.num_features = 1
    data.num_nodes = x.shape[0]
    return data

def read_BOLD_signal(folder):
    r = ''
    for i in folder.split('/')[:-1]: r += i+'/'
    x = [read_file(folder, name) for name in os.listdir(folder)]
    y = [read_file(r+'label_prediction', name) for name in os.listdir(folder)]
    edges = [adj_to_edge_index(read_file(r+'structure_prediction', name)) for name in os.listdir(folder)]
    return x, y, edges

def read_file(csv_r, csv_n):
    return np.loadtxt(osp.join(csv_r, csv_n), delimiter=',').astype(np.float32)