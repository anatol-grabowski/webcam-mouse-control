import torch.nn as nn
import pickle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# get top inputs:
    # inputs = X_train_tensor.clone().detach().requires_grad_(True)
    # optimizer.zero_grad()
    # outputs = model(inputs)
    # loss = criterion(outputs, y_train_tensor)
    # loss.backward()
    # grads = inputs.grad.cpu().detach().numpy()
    # print(grads.shape)
    # scores = grads.mean(axis=0)
    # print(scores.shape)
    # # print(scores)
    # indices = (np.argsort(scores)[::-1] / 2).astype(dtype=np.int)
    # print(indices.tolist())


train_indices = [
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,  # forehead
    108, 151, 337,  # forehead lower
    143, 156, 70, 63, 105, 66, 107,  # brow right outer
    336, 296, 334, 293, 300, 383, 372,  # brow left outer
    124, 46, 53, 52, 65, 55, 193,  # brow right middle
    285, 295, 282, 283, 276, 353, 417,  # brow left middle
    226, 247, 246, 221,  # around right eye
    446, 467, 466, 441,  # around left eye
    189, 190, 173, 133, 243, 244, 245, 233,  # right z
    413, 414, 398, 362, 463, 464, 465, 153,  # left z
    58, 172, 136, 150,  # right cheek
    288, 397, 365, 379,  # left cheek
    468, 469, 470, 471, 472,  # right iris
    473, 474, 475, 476, 477,  # left iris
]


model_arch = [512, 128, 32]


class EyePositionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(EyePositionPredictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, model_arch[0])
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(model_arch[0], model_arch[1])
        self.relu2 = nn.ReLU()
        self.hidden2 = nn.Linear(model_arch[1], model_arch[2])
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(model_arch[-1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu2(x)
        x = self.hidden2(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def save_to_file(self, filepath=model_filepath):
        self.to(torch.device('cpu'))
        with open(filepath, 'wb') as model_file:
            pickle.dump(self, model_file)
        print('saved model to file', filepath)
        self.to(device)

    def load_from_file(filepath=model_filepath):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model

    def model_name(self):
        layers = [self.input_size, *model_arch, self.output_size]
        return "-".join([str(l) for l in layers])
