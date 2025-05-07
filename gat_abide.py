import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.nn import AttentionalAggregation

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean


def load_abide_data(path):

    data = np.load(path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    return final_timeseires, final_pearson, labels, site

class GraphDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data_list = self.generate_data_list()
        self.data, self.slices = self.collate(self.data_list)

    def generate_data_list(self):
        data_list = []

        # Load the npy data
        final_timeseires, final_pearson, labels, _ = load_abide_data('abide.npy')
        length = final_timeseires.shape[0]
        for ind in range(length):
            #node_feature = final_timeseires[ind]
            node_feature = final_pearson[ind]
            x = torch.tensor(node_feature, dtype=torch.float32)
            adj = final_pearson[ind] - np.eye(final_pearson[ind].shape[0])

            # Create edge connections based on threshold
            edge_index = np.stack(np.where(adj.abs() > 0.3), axis=1)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            # edge_index = torch.tensor([[i, j] for i in range(adj.shape[0]) for j in range(adj.shape[1]) if (i != j and adj[i][j].abs() > 0.5)], dtype=torch.long).t().contiguous()

            y = torch.tensor(labels[ind], dtype=torch.long)


            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list

from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, PowerMeanAggregation

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=8, hidden_channels=64, num_layers=2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.att_heads = num_heads
        self.hidden_channels = hidden_channels

        self.convs.append(GATConv(num_node_features, hidden_channels, heads=num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))

        self.convs.append(GATConv(hidden_channels * num_heads, num_classes, heads=1))

        # Choose one of the pooling methods
        self.pool = PowerMeanAggregation()
        #self.pool = AttentionalAggregation(gate_nn=Linear(num_classes, 1))
        #self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
        
        x = self.convs[-1](x, edge_index)
        x = self.pool(x, batch)
        # x = global_mean_pool(x, batch)

        return x

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    correct = 0
    correct_pred = []
    correct_prob = []
    labels = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        prob = out.max(dim=1)[0]
        correct += (pred == data.y).sum().item()
        correct_pred += pred.tolist()
        correct_prob += prob.tolist()
        labels += data.y.tolist()
    return correct / len(loader.dataset), precision_score(labels, correct_pred), recall_score(labels, correct_pred), f1_score(labels, correct_pred)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GraphDataset()
train_dataset = dataset[:int(0.7 * len(dataset))]
test_dataset = dataset[int(0.7 * len(dataset)):]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

weight = train_dataset.data.y.bincount().float()
print(weight)

model = GAT(num_node_features=200, num_classes=2, num_heads=8, hidden_channels=128, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

best_f1 = 0
best_acc = 0
best_recall = 0
best_precision = 0
best_epoch = 0
for epoch in range(1, 101):
    loss = train(model, train_loader, optimizer, criterion)
    acc, precision, recall, f1 = test(model, test_loader)

    # Choose the best results based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_recall = recall
        best_precision = precision
        best_epoch = epoch
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}')
print(f'Best Epoch {best_epoch:02d}, Best Test Acc: {best_acc:.4f}, Best Test Precision: {best_precision:.4f}, Best Test Recall: {best_recall:.4f}, Best Test F1: {best_f1:.4f}')
