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

def visualisation(df):
    batchPath = "input"

#    rs = np.random.RandomState(0)
#    df = pd.DataFrame(rs.rand(10, 10))
#    print(df.head(20))

    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


    plt.matshow(df)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    #cb = plt.colorbar(datasetParts)
    #cb.ax.tick_params(labelsize=14)
    plt.title('Matrix', fontsize=16)
    plt.show()


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    datasetID:List[str] = ["libraryThing", "ml1m"]
    #datasetID:List[str] = ["libraryThing"]
    datasetID:List[str] = ["ml1m"]
    datasetFolds:List[int] = [0, 1, 2, 3, 4]
    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    datasetStarts:List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    r = [[[0 for _ in range(len(datasetParts))] for _ in range(len(datasetStarts))] for _ in range(len(datasetFolds))]

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
                    batchID:str = datasetIdI + "-Part" + datasetPartStrI + "-Stars" + startsStrI + "-Fold" + str(datasetFoldI)

                    path = "results" + os.sep + batchID + os.sep + "performance"
                    performanceFiles:str = os.listdir("results" + os.sep + batchID + os.sep + "performance")
                    resultFiles = [x for x in performanceFiles if x.endswith('.tsv')]
                    if resultFiles == []:
                        print(path)
                        continue
                    resultFile = resultFiles[0]
                    #print(resultFile)
                    results_df = pd.read_csv(path + os.sep + resultFile, header=0, delim_whitespace=True)
                    metric = results_df.iloc[0]["nDCG"]
                    metric = results_df.iloc[0]["HR"]
                    valuesI.append(metric)
                avr = sum(valuesI) / len(valuesI)
                columsI.append(avr)
            rows.append(columsI)
    print(r)

    df = pd.DataFrame(rows, columns=datasetParts)
    print(df)


    visualisation(df)
