import torch.optim as optim
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_filepath = './data/model.pickle'


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


class GazePredictor(nn.Module):
    def __init__(self, arch):
        super(GazePredictor, self).__init__()
        self.scaler = StandardScaler()
        self.arch = arch
        self.fc1 = nn.Linear(*arch[0:1+1])
        self.relu = nn.LeakyReLU()
        if len(arch) > 3:
            self.hidden1 = nn.Linear(*arch[1:2+1])
            self.relu2 = nn.LeakyReLU()
        if len(arch) > 4:
            self.hidden2 = nn.Linear(*arch[2:3+1])
            self.relu3 = nn.LeakyReLU()
        if len(arch) > 5:
            self.hidden3 = nn.Linear(*arch[3:4+1])
            self.relu4 = nn.LeakyReLU()
        self.fc2 = nn.Linear(*arch[-2:])
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if len(self.arch) > 3:
            x = self.hidden1(x)
            x = self.relu2(x)
        if len(self.arch) > 4:
            x = self.hidden2(x)
            x = self.relu3(x)
        if len(self.arch) > 5:
            x = self.hidden3(x)
            x = self.relu4(x)
        x = self.fc2(x)
        return x

    def save_to_file(self, filepath=model_filepath):
        self.to(torch.device('cpu'))
        with open(filepath, 'wb') as model_file:
            pickle.dump(self, model_file)
        print('saved model to file', filepath)
        self.to(device)

    @staticmethod
    def load_from_file(filepath=model_filepath):

        with open(filepath, 'rb') as file:
            state_dict, scaler = pickle.load(file)
            model = GazePredictor([168, 256, 64, 2])
            model.load_state_dict(state_dict)
            model.scaler = scaler
        return model

    def model_name(self):
        return f'{"-".join([str(l) for l in self.arch])}'


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
