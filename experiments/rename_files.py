import os
import re
import sys
import numpy as np


monxy = np.array([287, 1440])


def rename_files(directory):
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            new_filename = re.sub(
                r'\[(\d+) (-?\d+)\]', lambda match: f"[{int(match.group(1)) - 0} {int(match.group(2)) - monxy[0] + monxy[1]}]", filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")


path_to_directory = sys.argv[1]
rename_files(path_to_directory)
