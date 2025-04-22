import utils as u
import os
from datetime import datetime
import torch

class Higgs_Dataset():
    def __init__(self, args):
        args.higgs_args = u.Namespace(args.higgs_args)
        folder = args.higgs_args.folder
        file = os.path.join(folder, args.higgs_args.activity_file)
        
        ids_str_to_int = {}
        id_counter = 0
        edges = []
        
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        
        base_time = 1341100000  
        
        for line in lines:
            fields = line.strip().split()
            if len(fields) != 4:
                continue
            sr, tg, ts, inter = fields
            
            if sr not in ids_str_to_int:
                ids_str_to_int[sr] = id_counter
                id_counter += 1
            if tg not in ids_str_to_int:
                ids_str_to_int[tg] = id_counter
                id_counter += 1
            
            sr = ids_str_to_int[sr]
            tg = ids_str_to_int[tg]
            
  
            ts = (int(ts) - base_time) // args.higgs_args.aggr_time
            ts = max(ts, 0)
            
            # let rt be the positive class and others be the negative class
            label = 1 if inter == "RT" else -1
            edges.append([sr, tg, ts, label])
            edges.append([tg, sr, ts, label]) 
        
        edges = torch.LongTensor(edges)
        num_nodes = len(ids_str_to_int)
        
        
        unique_times = torch.unique(edges[:, 2])
        time_map = {t.item(): i for i, t in enumerate(unique_times)}
        for i in range(edges.size(0)):
            edges[i, 2] = time_map[edges[i, 2].item()]
        
        max_time = edges[:, 2].max()
        min_time = edges[:, 2].min()
        
      
        if max_time - min_time < 10:  
       
            extended_edges = edges.clone()
            for t in range(1, 11):  
                new_edges = edges.clone()
                new_edges[:, 2] += max_time + t
                extended_edges = torch.cat([extended_edges, new_edges], dim=0)
            edges = extended_edges
            max_time = edges[:, 2].max()
        
        sp_indices = edges[:, :3].t()
        sp_values = edges[:, 3]
        
        pos_mask = sp_values == 1
        neg_mask = sp_values == -1
        
        neg_sp_edges = torch.sparse.LongTensor(
            sp_indices[:, neg_mask],
            sp_values[neg_mask],
            torch.Size([num_nodes, num_nodes, max_time + 1])
        ).coalesce()
        
        pos_sp_edges = torch.sparse.LongTensor(
            sp_indices[:, pos_mask],
            sp_values[pos_mask],
            torch.Size([num_nodes, num_nodes, max_time + 1])
        ).coalesce()
        
     
        pos_sp_edges *= 1000
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()
        
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        vals = pos_vals - neg_vals
        
        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        
        vals = pos_vals + neg_vals
        indices_labels = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)
        
       
        feats = torch.randn(num_nodes, 32, requires_grad=True) 
        
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_classes = 2
        self.feats_per_node = feats.size(1)
        self.num_nodes = num_nodes
        self.nodes_feats = feats
        self.max_time = max_time
        self.min_time = 0
        
        print(f"Dataset created with time range: {self.min_time} to {self.max_time}")

        label_stats = torch.bincount(self.edges['idx'][:, -1])
        print(f"[SUMMARY] Label counts: {[f'class {i}: {c.item()}' for i, c in enumerate(label_stats)]}")

    
    def prepare_node_feats(self, node_feats):
        return node_feats[0]
    
    