from model import MLP
from data import load_data, save_result, TitanicDataset
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
import datetime
import numpy as np

device = 0
epoch = 50
run_name = datetime.datetime.now().strftime("%m-%d-%H-%M")

x, y, x_test, passenger_ids,y_test = load_data()
dataset = TitanicDataset(x, y)
dataset_test = TitanicDataset(x_test,y_test)

# Split dataset into training and validation sets (8:2 split)
train_size = int(1 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=16, shuffle=False)

# Model, optimizer, and loss function
model = MLP([dataset.feature_dim, 60, 50, 2]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=0.02)
criterion = torch.nn.CrossEntropyLoss()

def train():
    best_acc = 0
    for ep in range(epoch):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{epoch} - Training")
        for b_idx, d in enumerate(train_bar):
            feature, label = d
            feature = feature.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logit = model(feature)
            loss = criterion(logit, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'loss': total_loss / (b_idx + 1)})

        # Validation phase
        model.eval()
        val_total = 0
        val_correct = 0
        val_bar = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{epoch} - Validation")
        with torch.no_grad():
            for feature, label in val_bar:
                feature = feature.to(device)
                label = label.to(device)
                logit = model(feature)
                pred = torch.argmax(logit, dim=1)
                val_correct += torch.sum(label == pred).item()
                val_total += len(label)
                val_bar.set_postfix({'val_acc': val_correct / val_total})

        # Save the best model
        val_acc = val_correct / val_total
        if val_acc > best_acc:
            print(f"NEW BEST ACC: {val_acc}")
            torch.save(model.state_dict(), f'./checkpoints/{run_name}')
            best_acc = val_acc

def inference(run_name):
    model.load_state_dict(torch.load(f'./checkpoints/{run_name}'))
    model.eval()

    predictions = []
    for b in test_dataloader:
        feature, _ = b
        feature = feature.to(device)
        logit = model(feature)
        label_val = torch.argmax(logit, dim=1)
        predictions.extend(label_val.cpu().numpy())

    save_result('mlp', np.array(predictions), passenger_ids)

if __name__ == '__main__':
    train()
    inference(run_name)
