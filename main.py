from load_data import load_data
from train import train
from train import test
from model import create_model

import torch

adj, features, labels, idx_train, idx_val, idx_test = load_data()
n_feat = features.shape[1]
n_hid = 8
n_class = labels.max().item() + 1

model = create_model('GAT', n_feat, n_hid, n_class)

lr = 0.005
weight_decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)

for epoch in range(100):
    loss = train(model=model, optimizer=optimizer, features=features, adj=adj, labels=labels, idx_train=idx_train)
    train_acc, valid_acc, test_acc = test(model=model, features=features, adj=adj, labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train_acc: {100 * train_acc:.3f}%, '
                              f'Valid_acc: {100 * valid_acc:.3f}% '
                              f'Test_acc: {100 * test_acc:.3f}%')