import zipfile
import io
import requests
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from elliot.run import run_experiment

def runner():
    batchPath = "input"
    onlyfiles = [f for f in listdir(batchPath) if isfile(join(batchPath, f)) and f != 'zzz.yml']

    for batchFileI in onlyfiles:
        print(batchFileI)
        # os.remove(batchPath + os.sep + batchFileI)
        os.renames(batchPath + os.sep + batchFileI, batchPath + os.sep + 'zzz.yml')
        run_experiment(batchPath + os.sep + 'zzz.yml')
        break

if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    runner()