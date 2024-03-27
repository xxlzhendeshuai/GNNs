import torch
import torch.nn.functional as F


def train(model, optimizer, features, adj, labels, idx_train):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = model(features,adj)
    loss = F.nll_loss(output[idx_train],labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, features, adj, labels, idx_train, idx_val, idx_test):
    model.eval()
    out = model(features,adj)
    y_pred = out.argmax(axis=-1)
    correct = y_pred == labels
    train_acc = correct[idx_train].sum().float()/len(idx_train)
    valid_acc = correct[idx_val].sum().float()/len(idx_val)
    test_acc = correct[idx_test].sum().float()/len(idx_test)
    
    return train_acc, valid_acc, test_acc