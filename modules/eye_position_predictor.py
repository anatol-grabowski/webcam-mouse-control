import torch.nn as nn
import pickle

model_filepath = './data/model.pickle'


class EyePositionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(EyePositionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(128, 16)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

    def save_to_file(self, filepath=model_filepath):
        with open(filepath, 'wb') as model_file:
            pickle.dump(self, model_file)
        print('saved model to file', filepath)

    def load_from_file(filepath=model_filepath):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
