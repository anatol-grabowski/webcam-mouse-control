import numpy as np
import re
import glob


def get_paths(globs):
    glpaths = []
    for gl in globs:
        glpaths.extend(glob.glob(gl))
    return [str(p) for p in glpaths]


def get_xy_from_filename(filename):
    pattern = r'\[(\d+) (\d+)\]'
    match = re.search(pattern, filename)
    xy = np.array([*map(int, match.groups())])
    return xy
