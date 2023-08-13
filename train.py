import torch
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from modules.gaze_predictor import GazePredictor
from modules.dataset import Dataset


X, y = Dataset.read_dataset()
num_landmarks = X.shape[1]
print('shapes', X.shape, y.shape)
print('mean xy', y.mean(axis=0))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize the data
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Initialize the model
input_size = num_landmarks
output_size = y.shape[1]
model = GazePredictor(input_size, output_size)
if len(sys.argv) >= 3:
    model = GazePredictor.load_from_file(sys.argv[2])


class CustomLoss(nn.Module):
    def __init__(self, penalty_factor):
        super(CustomLoss, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        weighted_errors = squared_errors * (1 + y_true ** 2) * self.penalty_factor
        loss = torch.mean(weighted_errors)
        return loss


# Define loss function and optimizer
# criterion = nn.MSELoss()
criterion = CustomLoss(1.5)
optimizer = optim.Adam(model.parameters(), lr=float(sys.argv[1]))


def evaluate():
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
    model.train()


# Training loop
num_epochs = int(100e3)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if (epoch + 1) % 5000 == 0:
        evaluate()
        model.save_to_file(f'./data/model-{epoch+1}.pickle')

evaluate()


model_filepath = './data/model.pickle'
model.save_to_file(model_filepath)
