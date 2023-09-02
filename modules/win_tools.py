import subprocess

import re


def wmctrl_l():
    list_windows = subprocess.run('wmctrl -l', shell=True, stdout=subprocess.PIPE, text=True).stdout
    matches = re.findall(r'(0x\w+)\s+(\d+)\s+([-\w]+)\s+(.*)', list_windows)
    return matches


def wmctrl_r(winname, x, y, w=800, h=600):
    subprocess.run(f'wmctrl -r {winname} -e 0,{x},{y},{w},{h}', shell=True, stdout=subprocess.PIPE, text=True)


if __name__ == '__main__':
    wins = wmctrl_l()
    print(wins)
