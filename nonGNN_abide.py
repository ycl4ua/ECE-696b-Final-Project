import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score

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

from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        _, self.data_list, self.label_list, _ = load_abide_data('abide.npy')
        length = self.data_list.shape[0]
        self.data_list = self.data_list[:int(0.7 * length)]
        self.label_list = self.label_list[:int(0.7 * length)]

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        return {'input': self.data_list[idx], 'label': self.label_list[idx]}
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        _, self.data_list, self.label_list, _ = load_abide_data('/data/yichengleng/GraphSSL/abide.npy')
        length = self.data_list.shape[0]
        self.data_list = self.data_list[int(0.7 * length):]
        self.label_list = self.label_list[int(0.7 * length):]

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        return {'input': self.data_list[idx], 'label': self.label_list[idx]}

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoding = nn.Linear(input_dim, hidden_dim)

        # Choose transformer or linear layers
        
        # self.transformer_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
        #     num_layers=num_layers
        # )
        self.transformer_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_dim, num_classes) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length)
        b, s, i = x.shape
        x = self.encoding(x)  # (batch_size, seq_length, hidden_dim)
        x = x.transpose(0, 1) # (seq_length, batch_size, hidden_dim)
        transformed = self.transformer_encoder(x)  # (seq_length, batch_size, hidden_dim)
        pooled = torch.sum(transformed, dim=0) / s # (batch_size, hidden_dim)
        pooled = self.dropout(pooled)
        output = self.linear(pooled)  # (batch_size, num_classes)
        return output

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data['input'].to(device))
        loss = criterion(out, data['label'].to(int).to(device))
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
        out = model(data['input'].to(device))
        pred = out.argmax(dim=1)
        prob = out.max(dim=1)[0]
        correct += (pred == data['label'].to(int).to(device)).sum().item()
        correct_pred += pred.tolist()
        correct_prob += prob.tolist()
        labels += data['label'].to(int).tolist()
    return correct / len(loader.dataset), precision_score(labels, correct_pred), recall_score(labels, correct_pred), f1_score(labels, correct_pred)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TrainDataset()
test_dataset = TestDataset()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


input_dim = 200
hidden_dim = 256
num_layers = 4
num_heads = 8
num_classes = 2
dropout = 0.1

model = TransformerClassifier(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout).to(device)
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
    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_recall = recall
        best_precision = precision
        best_epoch = epoch
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}')
print(f'Best Epoch {best_epoch:02d}, Best Test Acc: {best_acc:.4f}, Best Test Precision: {best_precision:.4f}, Best Test Recall: {best_recall:.4f}, Best Test F1: {best_f1:.4f}')
