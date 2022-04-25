import zipfile
import io
import requests
import os
import sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from elliot.run import run_experiment
from eliotExtension.runner import runner  # fnc
from eliotExtension.batchesGenerator import generateBatches  # fnc
from eliotExtension.division import generateDatasets # fnc

if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    if len(sys.argv) == 2 and sys.argv[1] == "-generateBatches":
       generateBatches()

    if len(sys.argv) == 2 and sys.argv[1] == "-generateDatasets":
       generateDatasets()

    if len(sys.argv) == 1:
       runner()