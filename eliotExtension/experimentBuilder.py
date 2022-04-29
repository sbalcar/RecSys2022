import zipfile
import io
import os
import numpy as np
import pandas as pd
from typing import List

from elliot.run import run_experiment

class ExperimentBuilder:

    experimentDef:str = """experiment:
  dataset: <datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>
  path_output_rec_result: results/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>/recs
  path_output_rec_weight: results/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>/weight
  path_output_rec_performance: results/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>/performance
  path_log_folder: results/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>/log
  data_config:
     strategy: fixed
     train_path: ../data/<datasetID>/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>.tsv
     test_path: ../data/<datasetID>/<datasetID>-Fold<fold>-test.tsv
  top_k: 10
  evaluation:
    simple_metrics: [nDCG, ItemCoverage, HR, RMSE, EPC, APLT]
#  gpu: -1
  models:
"""

    algItemKNN = """    ItemKNN:    # knn, item_knn, item_knn
      meta:
        save_recs: True
      neighbors: 50
      similarity: cosine
"""
    algSVDpp = """    SVDpp:
      meta:
        save_recs: True
"""
    algiALS = """    iALS:       # latent_factor_models, iALS, iALS
      meta:
        save_recs: True
"""
    algMultiVAE = """    MultiVAE:   # autoencoders, multi_vae
      meta:
        save_recs: True
"""
    algLightGCN = """    LightGCN:   # graph_based, lightgcn, LightGCN
      meta:
        save_recs: True
"""

    @staticmethod
    def getExperimentStr(datasetID:str, datasetPart:str, granularity:str, fold:str, algorithms:List[str]):

        experimentStr = ExperimentBuilder.experimentDef.replace("<datasetID>", datasetID)
        experimentStr = experimentStr.replace("<datasetPart>", datasetPart)
        experimentStr = experimentStr.replace("<stars>", granularity)
        experimentStr = experimentStr.replace("<fold>", fold)

        for algorithmI in algorithms:
            experimentStr = experimentStr + algorithmI

        return experimentStr


def generateBatches():

    datasetID:List[str] = ["libraryThing", "ml1m"]
    datasetFolds:List[int] = [0, 1, 2, 3, 4]
    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    datasetStarts:List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for datasetIdI in datasetID:
        for datasetFoldI in datasetFolds:
            for datasetPartI in datasetParts:
                for startsI in datasetStarts:
                    datasetPartStrI = str(datasetPartI)
                    if datasetPartI < 100:
                        datasetPartStrI = "0" + str(datasetPartI)
                    startsStrI = "0" + str(startsI)
                    if startsI == 10:
                        startsStrI = "10"
                    batchID:str = datasetIdI + "-Part" + datasetPartStrI + "-Stars" + startsStrI + "-Fold" + str(datasetFoldI)
                    experimentStr:str = ExperimentBuilder.getExperimentStr(
                                    datasetIdI, datasetPartStrI, startsStrI, str(datasetFoldI),
                                    #[ExperimentBuilder.algiALS])
                                    [ExperimentBuilder.algItemKNN])

                    f = open("./batches" + os.sep + batchID + ".yml", "wt")
                    f.write(experimentStr)
                    f.close()


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    generateBatches()
