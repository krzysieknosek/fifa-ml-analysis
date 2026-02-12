from pathlib import Path

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

# Loading data
DATA_PATH = Path(__file__).resolve().parent.parent / "players_22.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

df = df.dropna(subset=['club_position', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'])

X = df[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']].values.astype(np.float32)
y = df['club_position']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Data split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Tensors
X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).long()
y_val = torch.from_numpy(y_val).long()
y_test = torch.from_numpy(y_test).long()

# Device
device = torch.device("cpu")

# Dataset & DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(6, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes)
)
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training time measurement
start_time = time.time()

# Training
for epoch in range(20):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# End of measurement
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")


# Evaluation
def evaluate(X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true = y.numpy()
        acc = accuracy_score(true, preds)
        f1_macro = f1_score(true, preds, average='macro')
        f1_weighted = f1_score(true, preds, average='weighted')
        return acc, f1_macro, f1_weighted


acc_train, f1m_train, f1w_train = evaluate(X_train, y_train)
acc_val, f1m_val, f1w_val = evaluate(X_val, y_val)
acc_test, f1m_test, f1w_test = evaluate(X_test, y_test)

print(f"Train:    acc={acc_train:.4f}, macro_f1={f1m_train:.4f}, weighted_f1={f1w_train:.4f}")
print(f"Val:      acc={acc_val:.4f}, macro_f1={f1m_val:.4f}, weighted_f1={f1w_val:.4f}")
print(f"Test:     acc={acc_test:.4f}, macro_f1={f1m_test:.4f}, weighted_f1={f1w_test:.4f}")
