import pickle
import numpy as np

dataset_filepath = './data/prepared.pickle'


class Dataset():
    def read_dataset(filepath=dataset_filepath):
        with open(filepath, 'rb') as file:
            dataset_list = pickle.load(file)
        X = np.array([dp['landmarks'].ravel() for dp in dataset_list])
        y = np.array([dp['cursor'] for dp in dataset_list])
        monsize = np.array([2560, 1440])
        y = y / monsize * 2 - 1
        return X, y

    def save_dataset(dataset, filepath=dataset_filepath):
        with open(dataset_filepath, 'wb') as file:
            pickle.dump(dataset, file)
        print('saved to file', dataset_filepath)
