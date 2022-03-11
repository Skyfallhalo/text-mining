import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class QuestionDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.int32), self.y[idx]


# Calls external model (as specified by config) with data, receives returned data, saves results.
def trainModel(train_dl, val_dl, model, numEpochs=10, lr=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimiser = torch.optim.SGD(filter(lambda e: e.requires_grad, model.parameters()), lr=lr)
    #Softmax ignored
    loss_fn = F.cross_entropy
    for epoch in range(numEpochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_total = 0
        total_train_loss = 0.0
        for X_batch, y_batch in train_dl:
            # Forward pass and get prediction
            X_batch = X_batch.type(torch.LongTensor).to(device)
            y_batch = y_batch.to(device)
            y_out = model(X_batch)

            # Compute the loss, gradients, and update parameters
            optimiser.zero_grad()
            loss = loss_fn(y_out, y_batch)
            loss.backward()
            optimiser.step()

            # Accumulate loss
            train_total += y_batch.shape[0]
            total_train_loss += loss.item() * y_batch.shape[0]
        train_loss = total_train_loss / train_total

        # Validation phase
        model.eval()
        correct = 0
        val_total = 0
        total_val_loss = 0.0
        for X_batch, y_batch in val_dl:
            X_batch = X_batch.type(torch.LongTensor).to(device)
            y_batch = y_batch.to(device)
            y_out = model(X_batch)
            loss = loss_fn(y_out, y_batch)
            val_total += y_batch.shape[0]
            total_val_loss += loss.item() * y_batch.shape[0]
            y_pred = torch.max(y_out, 1)[1]
            correct += (y_pred == y_batch).sum().item()
        val_loss = total_val_loss / val_total
        val_acc = correct / val_total

        elapsed_time = time.time() - start_time

        print('Epoch [{:02}/{:02}] \t loss={:.4f} \t val_loss={:.4f} \t val_acc={:.4f} \t time={:.2f}s'.format(
            epoch + 1, numEpochs, train_loss, val_loss, val_acc, elapsed_time))

    return model

# Attempts to run BOW or BiLSTM with data, receives returned data, and saves results.
def testModel(X, y, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = X.to(device)
    y = y.to(device)
    y_out = model(X)
    y_pred = torch.max(y_out, 1)[1]

    return y_pred


