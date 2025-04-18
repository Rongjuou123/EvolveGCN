import utils as u
import os
import torch
import pandas as pd
from datetime import datetime


class Reddit_NC_Dataset():
    def __init__(self, args):
        args.reddit_args = u.Namespace(args.reddit_args)

        self.folder = args.reddit_args.folder
        self.title_file = os.path.join(self.folder, args.reddit_args.title_edges_file)
        self.body_file = os.path.join(self.folder, args.reddit_args.body_edges_file)
        self.node_file = os.path.join(self.folder, args.reddit_args.nodes_file)
        self.label_file = os.path.join(self.folder, args.reddit_args.node_labels_file)
        self.aggr_time = args.reddit_args.aggr_time

        self.nodes_feats, self.node2id = self.load_node_feats()
        self.edges = self.load_edges()
        self.nodes_labels_times = self.load_node_labels()
        

        self.num_nodes = self.nodes_feats.size(0)
        self.feats_per_node = self.nodes_feats.size(1)
        self.prepare_node_feats = self._prepare_node_feats

        self.max_time = self.edges['idx'][:, 2].max()
        self.min_time = self.edges['idx'][:, 2].min()

    def load_node_feats(self):
        df = pd.read_csv(self.node_file)
        if 'id' not in df.columns:
            df.columns = ['id'] + [f'f{i}' for i in range(1, df.shape[1])]
        node2id = {name: i for i, name in enumerate(df['id'])}
        feats = torch.tensor(df.drop(columns=['id']).values, dtype=torch.float)
        return feats, node2id

    def load_node_labels(self):
        df = pd.read_csv(self.label_file)
        label_data = []

        for _, row in df.iterrows():
            name = row['id']
            if name in self.node2id:
                nid = self.node2id[name]
                label = int(row['label'])
                time = int(row['time'])  # 原始时间
                label_data.append([nid, label, time])

        
        label_data_ext = []
        max_t = self.edges['idx'][:, 2].max().item()  
        for nid, label, _ in label_data:
            for t in range(max_t + 1):
                label_data_ext.append([nid, label, t])
        label_data_ext = torch.tensor(label_data_ext, dtype=torch.long)
        labels = label_data_ext[:, 1]
        print("🧪 Label sanity check:")
        print(" - unique labels:", torch.unique(labels))
        print(" - min:", labels.min().item(), "max:", labels.max().item())
        print(" - dtype:", labels.dtype)

        return torch.tensor(label_data_ext, dtype=torch.long)

    def load_edges(self):
        base_time = datetime.strptime("19800101", "%Y%m%d")
        all_edges = []

        for file_path in [self.title_file, self.body_file]:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 4:
                        continue
                    src, tgt, timestamp = parts[0], parts[1], parts[3].split(' ')[0]
                    if src in self.node2id and tgt in self.node2id:
                        t = datetime.strptime(timestamp, "%Y-%m-%d")
                        ts = (t - base_time).days
                        u_id, v_id = self.node2id[src], self.node2id[tgt]
                        all_edges.append([u_id, v_id, ts])
                        all_edges.append([v_id, u_id, ts])

        edge_tensor = torch.tensor(all_edges, dtype=torch.long)
        edge_tensor[:, 2] = u.aggregate_by_time(edge_tensor[:, 2], self.aggr_time)

        return {'idx': edge_tensor, 'vals': torch.ones(edge_tensor.size(0))}

    def _prepare_node_feats(self, x):
        return x[0]
