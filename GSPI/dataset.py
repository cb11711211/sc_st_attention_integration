from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

# General rules:
# 1. message passing edge: used for GNN message passing
# 2. supervision edge: used in loss function for backpropagation

## design for transductive learning
transductive_tsf = RandomNodeSplit(
    num_splits=5, # 5-fold cross validation
    num_val=0.2, # 10% of training data for validation
    num_test=0.2, # 10% of training data for testing
)

## design for inductive learning
inductive_tsf = RandomLinkSplit(
    is_undirected=True, # undirected graph
    num_val=0.2, # 10% of training data for validation
    num_test=0.2, # 10% of training data for testing
)