from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, data, adj_mtx):
        self.data = data
        edge_index = adj_mtx.nonzero(as_tuple=False).t()
        self.edge_index = edge_index
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.adj_mtx[idx]