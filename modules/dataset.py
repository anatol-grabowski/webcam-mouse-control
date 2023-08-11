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


landmark_indices = [
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284,  # forehead
    108, 151, 337,  # forehead lower
    143, 156, 70, 63, 105, 66, 107,  # brow right outer
    336, 296, 334, 293, 300, 383, 345,  # brow left outer
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
