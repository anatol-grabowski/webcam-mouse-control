import torch
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from modules.eye_position_predictor import EyePositionPredictor


dataset_filepath = './data/prepared.pickle'


def read_dataset(filepath):
    with open(filepath, 'rb') as file:
        dataset_list = pickle.load(file)
    X = np.array([dp['landmarks'].ravel() for dp in dataset_list])
    y = np.array([dp['cursor'] for dp in dataset_list])
    monsize = np.array([2560, 1440])
    y = y / monsize * 2 - 1
    return X, y


# Generate random data for demonstration (replace with your data)
# num_samples = 1000
# num_landmarks = 68
# X = np.random.randn(num_samples, num_landmarks)
# y = np.random.rand(num_samples, 2) * 2 - 1  # Scaling to -1 to 1 range
X, y = read_dataset(dataset_filepath)
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
hidden_size = 128
output_size = 2
model = EyePositionPredictor(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')


model_filepath = './data/model.pickle'
with open(model_filepath, 'wb') as model_file:
    pickle.dump(model, model_file)
print('saved model to file', model_filepath)
