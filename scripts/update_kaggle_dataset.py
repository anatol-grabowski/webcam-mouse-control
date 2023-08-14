import kaggle
import sys


dataset_id = 'grabantot/webcam-mouse'
updated_dataset_file = sys.argv[1]
kaggle.api.dataset_create_version(dataset_id, folder=updated_dataset_file)
