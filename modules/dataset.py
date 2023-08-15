import pickle
import numpy as np
import os

dataset_filepath = './data/datasets'

# should not have local imports for unpickling in kaggle


class Dataset():
    def __init__(self):
        self.datapoints = []

    def add_datapoint(self, label, face, position):
        datapoint = {
            'label': label,
            'face': face,
            'position': position,
        }
        self.datapoints.append(datapoint)
        return True

    def load(filepath=dataset_filepath):
        ds = Dataset()
        if os.path.isdir(filepath):
            for fname in os.listdir(filepath):
                with open(f'{filepath}/{fname}', 'rb') as file:
                    datapoints = pickle.load(file)
                    ds.datapoints.extend(datapoints)
        else:
            with open(filepath, 'rb') as file:
                ds.datapoints = pickle.load(file)
        return ds

    def store(self, filepath=dataset_filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.datapoints, file)
        print('saved to file', filepath)

    def get_Xy(self):
        X = np.array([dp['face'].ravel() for dp in self.datapoints])
        y = np.array([dp['position'] for dp in self.datapoints])
        return X, y
