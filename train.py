import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import time
import pickle
from modules.dataset import Dataset
from modules.gaze_predictor import train_indices, GazePredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 678
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


dataset_filepath = './data/datasets/intg-static'
X, y = Dataset.load(dataset_filepath).get_Xy()
X = X.reshape(len(X), -1, 2)[:, train_indices].reshape(len(X), len(train_indices) * 2)
num_landmarks = X.shape[1]


input_size = num_landmarks
output_size = y.shape[1]
model = GazePredictor([input_size, 256, 64, output_size])
# model = EyePositionPredictor.load_from_file('/kaggle/working/model-168-512-256-128-32-2 0.0063 #1400k [mse].pickle')
if len(sys.argv) >= 2:
    model = GazePredictor.load_from_file(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42) # throw out part of the training data

num_tile = 1
X_train = np.tile(X_train, (num_tile, 1))
y_train = np.tile(y_train, (num_tile, 1))

# # Normalize the data
# scaler = MinMaxScaler(feature_range=(-1, 1))
model.scaler.fit(X_train)
X_train = model.scaler.transform(X_train)
X_test = model.scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


class Exp_diff_abs(nn.Module):
    def __init__(self, penalty_factor):
        super(Exp_diff_abs, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        penalty = torch.exp(torch.abs(y_true) - torch.abs(y_pred))
        squared_errors *= (1 + self.penalty_factor * penalty)
        loss = torch.mean(squared_errors)
        return loss

    def loss_name(self):
        return f'exp_diff_abs*{self.penalty_factor:.3f}'


class Sqr_y_true(nn.Module):
    def __init__(self, penalty_factor):
        super(Sqr_y_true, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        penalty = y_true ** 2
        squared_errors *= (1 + self.penalty_factor * penalty)
        loss = torch.mean(squared_errors)
        return loss

    def loss_name(self):
        return f'sqr_y*{self.penalty_factor:.1f}'


class Mse(nn.MSELoss):
    def loss_name(self):
        return f'mse'


# Define loss function and optimizer
criterion = Mse()
# criterion = Sqr_y_true(2.0)
# criterion = Exp_diff_abs(1.5)

model = model.to(device)
criterion = criterion.to(device)

history = []

epoch = 0


def evaluate():
    model.eval()
    eval_criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = eval_criterion(test_outputs, y_test_tensor)
        return test_loss


model_dirpath = './data/models'


best_mse = np.inf   # init to infinity
best_weights = None


def train(num_epochs):
    global epoch, best_mse

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    test_loss = evaluate()
    print(f'Test loss: {test_loss.item():.5f}')

    e0 = epoch
    t0 = time.time()
    for i in range(num_epochs):
        epoch += 1
        model.train()

#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()

        model.optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        model.optimizer.step()
        dt = time.time() - t0
        if dt > 10:
            t0 = time.time()
            test_loss = evaluate()
            print(f'Epoch {epoch}/{e0 + num_epochs}, {dt:.3f}s, Loss: {loss.item():.6f}, Test loss: {test_loss.item():.5f}')
            mse = float(test_loss)
            history.append([epoch, mse])
#             plt.plot(*list(zip(*history)))
#             plt.show()
            if mse < best_mse-0.0002:
                best_mse = mse
                model.save_to_file(f'{model_dirpath}/model-{model.model_name()} best [{criterion.loss_name()}].pickle')
        if epoch % 100000 == 0:
            model.save_to_file(
                f'{model_dirpath}/model-{model.model_name()} tr{loss.item():.4f} ts{test_loss.item():.4f} #{epoch // 1000}k {criterion.loss_name()}.pickle')

    evaluate()


print(f'{device=}')
print('train', X_train_tensor.size(), y_train_tensor.size(), 'mean xy', y.mean(axis=0))
print('test ', X_test_tensor.size(), y_test_tensor.size())
print(model.model_name(), criterion.loss_name())

train(int(1000e3))
