import os
import argparse
from os.path import isfile, join

"""
    Removes lower resolution repetitive pictures from folder located at --dirpath
"""

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath", type=str)

args = parser.parse_args()
mypath = args.dirpath

onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

d = {}

for str in onlyfiles:
    splity = str.split("-")
    dim = int(splity[1].split("x")[0])
    d[splity[0]] = max(d[splity[0]], dim) if splity[0] in d.keys() else dim


for str in onlyfiles:
    splity = str.split("-")
    dim = int(splity[1].split("x")[0])
    if d[splity[0]] != dim:
        os.remove(join(mypath,str))

