import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import APPNP, SGConv, SplineConv

class appnp(torch.nn.Module):
    def __init__(self, dataset, hidden, K, alpha):
        super().__init__()
        self.layer1 = Linear(dataset[0].num_features, hidden)
        self.layer2 = Linear(hidden, dataset.num_classes)
        self.prop = APPNP(K=K, alpha=alpha)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, d):
        x, edge_index = d.x, d.edge_index
        x = F.dropout(x, training=self.training)
        x = self.layer1(x).relu()
        x = F.dropout(x, training=self.training)
        x = self.layer2(x).relu()
        x = self.prop(x, edge_index)
        return x

class sgconv(torch.nn.Module):
    def __init__(self, dataset, K):
        super().__init__()
        self.layer1 = SGConv(dataset[0].num_node_features, dataset.num_classes, K=K)

    def reset_parameters(self):
        self.layer1.reset_parameters()

    def forward(self, d):
        x, edge_index = d.x, d.edge_index
        x = F.dropout(x, training=self.training)
        x = self.layer1(x, edge_index).relu()
        return F.log_softmax(x, dim=1)
    
class splineconv(torch.nn.Module):
    def __init__(self, dataset, hidden):
        super().__init__()
        self.layer1 = SplineConv(dataset[0].num_node_features, hidden, dim=1, kernel_size=2)
        self.layer2 = SplineConv(hidden, dataset.num_classes, dim=1, kernel_size=2)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = self.layer1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, training=self.training)
        x = self.layer2(x, edge_index, edge_attr).relu()
        return F.log_softmax(x, dim=1)