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
    onlyfiles = [f for f in listdir(batchPath) if isfile(join(batchPath, f))]

    for batchFileI in onlyfiles:
        print(batchFileI)

        batchName:str = batchFileI.replace(".yml", "")
        print("Running: " + batchFileI)
        if os.path.isdir("results" + os.sep + batchName):
            print("Results are rexist - removing batch")
            print(batchPath + os.sep + batchFileI)
            os.remove(batchPath + os.sep + batchFileI)
            continue

        # os.remove(batchPath + os.sep + batchFileI)
        #os.renames(batchPath + os.sep + batchFileI, batchPath + os.sep + 'zzz.yml')
        #run_experiment(batchPath + os.sep + 'zzz.yml')
        run_experiment(batchPath + os.sep + batchFileI)
        break

if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    runner()
