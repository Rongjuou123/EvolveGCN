import utils as u
import os
import torch
import pandas as pd
from datetime import datetime


class Reddit_LP_Dataset():
    def __init__(self, args):
        args.reddit_args = u.Namespace(args.reddit_args)

        folder = args.reddit_args.folder
        self.title_file = os.path.join(folder, args.reddit_args.title_edges_file)
        self.body_file = os.path.join(folder, args.reddit_args.body_edges_file)
        self.node_file = os.path.join(folder, args.reddit_args.nodes_file)
        self.aggr_time = args.reddit_args.aggr_time

        self.nodes_feats, self.id_map = self.load_node_feats()
        self.edges = self.load_edges()

        self.num_nodes = self.nodes_feats.size(0)
        self.feats_per_node = self.nodes_feats.size(1)
        self.prepare_node_feats = self._prepare_node_feats

        self.max_time = self.edges['idx'][:, 2].max()
        self.min_time = self.edges['idx'][:, 2].min()

    def load_node_feats(self):
        df = pd.read_csv(self.node_file)

        if 'id' not in df.columns:
            df.columns = ['id'] + [f'f{i}' for i in range(1, df.shape[1])]

        id_map = {name: idx for idx, name in enumerate(df['id'])}
        feats = torch.tensor(df.drop(columns=['id']).values, dtype=torch.float)
        return feats, id_map

    def load_edges(self):
        base_time = datetime.strptime("19800101", "%Y%m%d")
        edge_list = []
        max_edges = 2_000_000  # Limit the number of edges to avoid memory issues
        count = 0

        for file_path in [self.title_file, self.body_file]:
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 4:
                        continue
                    src, tgt, timestamp = parts[0], parts[1], parts[3].split(' ')[0]
                    if src in self.id_map and tgt in self.id_map:
                        try:
                            t = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            try:
                                t = datetime.strptime(timestamp, "%Y-%m-%d")
                            except Exception:
                                continue
                        days = (t - base_time).days
                        u_id = self.id_map[src]
                        v_id = self.id_map[tgt]
                        edge_list.append([u_id, v_id, days])
                        edge_list.append([v_id, u_id, days])
                        count += 2
                        if count >= max_edges:
                            break
            if count >= max_edges:
                break

        if len(edge_list) == 0:
            raise ValueError("Edge list is empty! Please check if edge files are correctly formatted and IDs match nodes file.")

        edge_tensor = torch.tensor(edge_list, dtype=torch.int64)
        edge_tensor[:, 2] = u.aggregate_by_time(edge_tensor[:, 2], self.aggr_time)

        return {
            'idx': edge_tensor,
            'vals': torch.ones(edge_tensor.size(0))
        }

    def _prepare_node_feats(self, x):
        return x[0]