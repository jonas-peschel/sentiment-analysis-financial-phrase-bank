import wandb 
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm 
from torchmetrics import Accuracy
from pathlib import Path 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

import gc

def get_dataloaders(config, X_train, y_train, X_test, y_test):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, test_dataloader

class TextClassificationModel(nn.Module):

    def __init__(self, d_in, d_out, h):

        super().__init__()

        self.fc_layer_1 = nn.Linear(d_in, h)
        self.fc_layer_2 = nn.Linear(h, d_out)
        self.relu = nn.ReLU()
        self.input_dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):

        x = self.input_dropout(x)
        x = self.fc_layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_layer_2(x)

        return x
    
def l1_regularization(model, l1_reg):

    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))

    return l1_reg * l1_loss

def train_step(model, dataloader, loss_fn, acc_fn, optimizer, l1_reg, device):

    model = model.to(device)
    model.train()
    loss, acc = 0., 0.

    for X, y in dataloader:

        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)

        # Loss
        loss_batch = loss_fn(y_logits, y) + l1_regularization(model, l1_reg)    # add L1 regulatization
        acc_batch = acc_fn(y_logits, y)

        # reset optimizer
        optimizer.zero_grad()

        # error backpropagation
        loss_batch.backward()

        # optimization step
        optimizer.step()

        # accumulate loss and accuracy over the batches of on epoch
        loss += loss_batch 
        acc += acc_batch

    loss /= len(dataloader) 
    acc /= len(dataloader)

    return loss.item(), acc 

def test_step(model, dataloader, loss_fn, acc_fn, device):

    model = model.to(device)
    model.eval()
    loss, acc = 0., 0.

    with torch.inference_mode():
        for X, y in dataloader:

            # Forward pass
            y_logits = model(X)

            # Loss
            loss_batch = loss_fn(y_logits, y)
            acc_batch = acc_fn(y_logits, y)

            # accumulate loss and accuracy over the batches of on epoch
            loss += loss_batch 
            acc += acc_batch

        loss /= len(dataloader) 
        acc /= len(dataloader)

    return loss.item(), acc 

def train_and_test_loop(model, train_dataloader, test_dataloader, loss_fn, acc_fn, optimizer, l1_reg, device, n_epochs, LABELS):

    for epoch in tqdm(range(n_epochs)):

        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, acc_fn, optimizer, l1_reg, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, acc_fn, device)

        # log the results to weights and biases
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }, step=epoch)

        # inference on the complete train and test set each (log classification results to wandb)
        evaluation(model, train_dataloader, LABELS, device, "train", epoch, verbose=False) 
        evaluation(model, test_dataloader, LABELS, device, "test", epoch, verbose=False) 

    return model

def evaluation(model, dataloader, LABELS, device, mode, epoch, verbose=False):

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)

            # Prediction
            y_p = torch.argmax(y_logits, axis=1)

            # append results
            y_true.extend(y.cpu().tolist())
            y_pred.extend(y_p.cpu().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # classification performance results
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=np.nan, output_dict=True)

    # log results
    if verbose:
        wandb.log({f"report_{mode}": report})
    else:
        wandb.log({
            f"{mode}_macro_f1": report['macro avg']['f1-score'],
            f"{mode}_weighted_f1": report['weighted avg']['f1-score']
        }, step=epoch)
            
            
def train(config, X, y, LABELS):

    n_classes = len(LABELS)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    train_dataloader, test_dataloader = get_dataloaders(config, X_train, y_train, X_test, y_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextClassificationModel(d_in=X_train.shape[1], d_out=n_classes, h=config.hidden_layer_dim).to(device)

    class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # use weights due to imbalanced data
    acc_fn = Accuracy(task="multiclass", num_classes=n_classes)
    l1_reg = config.l1_reg
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    n_epochs = config.n_epochs

    trained_model = train_and_test_loop(model, train_dataloader, test_dataloader, loss_fn, acc_fn, optimizer, l1_reg, device, n_epochs, LABELS)

    # inference on the complete train and test set each (log classification results to wandb)
    evaluation(trained_model, train_dataloader, LABELS, device, "train", epoch=None, verbose=True) 
    evaluation(trained_model, test_dataloader, LABELS, device, "test", epoch=None, verbose=True) 

    # save model
    save_dir = Path("trained-models")
    save_path = save_dir / f"model_{wandb.run.name}.pt"
    save_dir.mkdir(exist_ok=True)
    torch.save(obj=trained_model, f=save_path)

    # garbage removal
    del model, trained_model, class_weights, loss_fn, acc_fn, optimizer, train_dataloader, test_dataloader, X_train, X_test, y_train, y_test
    gc.collect()    

