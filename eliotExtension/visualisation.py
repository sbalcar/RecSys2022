import zipfile
import io
import requests
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from elliot.run import run_experiment
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualisation(df, title:str, xticklabels:List[int], yticklabels:List[int]):
    batchPath = "input"

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(df, interpolation='nearest')
    fig.colorbar(cax)

    plt.title(title, fontsize=16)

    ax.set_xticklabels([0] + xticklabels)
    ax.set_yticklabels([0] + yticklabels)
    #plt.show()
    plt.savefig("visualisation" + os.sep + title + '.png')


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    #datasetID:List[str] = ["libraryThing", "ml1m", "ml25mSel2016"]
    #datasetID:List[str] = ["libraryThing"]
    #datasetID:List[str] = ["ml25mSel2016"]
    datasetID:List[str] = ["libraryThingSel20"]
    datasetFolds:List[int] = [0, 1, 2, 3, 4]
    #datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    datasetParts:List[int] = [20, 50, 70, 80, 90, 95, 100]
    #datasetStarts:List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    datasetStarts:List[int] = [1, 2, 3, 5, 7, 10]
    #algID:str = "EASER"
    #algID:str = "LightGCN"
    #algID:str = "UserKNN"
    #algID: str = "HTUserKNN"
    #algID:str = "ItemKNN"
    algID:str = "HTItemKNN"
    #algID: str = "IALS"
    #algID:str = "PMF"
    #algID:str = "PMF"
    #metric:str = "nDCG"     # nDCG, ItemCoverage, HR, RMSE, EPC, APLT
    metric:str = "HR"

    for datasetIdI in datasetID:
        rows = []
        for startsI in datasetStarts:
            columsI = []
            for datasetPartI in datasetParts:
                valuesI = []
                for datasetFoldI in datasetFolds:
                    datasetPartStrI = str(datasetPartI)
                    if datasetPartI < 100:
                        datasetPartStrI = "0" + str(datasetPartI)
                    startsStrI = "0" + str(startsI)
                    if startsI == 10:
                        startsStrI = "10"
                    batchID:str = datasetIdI + "-Alg" + algID + "-Part" + datasetPartStrI + "-Stars" + startsStrI + "-Fold" + str(datasetFoldI)
                    print(batchID)
                    path = "results" + os.sep + batchID + os.sep + "performance"
                    performanceFiles:str = os.listdir("results" + os.sep + batchID + os.sep + "performance")
                    resultFiles = [x for x in performanceFiles if x.endswith('.tsv')]
                    if resultFiles == []:
                        print(path)
                        continue
                    resultFile = resultFiles[0]
                    #print(resultFile)
                    results_df = pd.read_csv(path + os.sep + resultFile, header=0, delim_whitespace=True)
                    metricI = results_df.iloc[0][metric]
                    valuesI.append(metricI)
                avr = sum(valuesI) / len(valuesI)
                columsI.append(avr)
            rows.append(columsI)

        df = pd.DataFrame(rows, columns=datasetParts)
        print(df)

        title:str = datasetIdI + "-" + algID + "-" + metric
        visualisation(df, title, datasetParts, datasetStarts)
