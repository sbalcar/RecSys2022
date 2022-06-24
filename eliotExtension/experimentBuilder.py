import zipfile
import io
import os
import numpy as np
import pandas as pd
from typing import List

from elliot.run import run_experiment

class ExperimentBuilder:

    agsIds:List[str] = ["IALS", "HTIALS", "ItemKNN", "HTItemKNN", "UserKNN", "EASER", "LightGCN"]

    experimentDef:str = """experiment:
  dataset: <datasetID>-Part<datasetPart>-Alg<alg>-Stars<stars>-Fold<fold>
  path_output_rec_result: results/<datasetID>-Alg<alg>-Part<datasetPart>-Stars<stars>-Fold<fold>/recs
  path_output_rec_weight: results/<datasetID>-Alg<alg>-Part<datasetPart>-Stars<stars>-Fold<fold>/weight
  path_output_rec_performance: results/<datasetID>-Alg<alg>-Part<datasetPart>-Stars<stars>-Fold<fold>/performance
  path_log_folder: results/<datasetID>-Alg<alg>-Part<datasetPart>-Stars<stars>-Fold<fold>/log
  data_config:
     strategy: fixed
     train_path: ../data/<datasetID>/<datasetID>-Part<datasetPart>-Stars<stars>-Fold<fold>.tsv
     test_path: ../data/<datasetID>/<datasetID>-Fold<fold>-test.tsv
  top_k: 10
  evaluation:
    simple_metrics: [nDCG, ItemCoverage, HR, EPC, APLT]
#  gpu: -1
  models:
"""

    algItemKNN = """    ItemKNN:    # knn, item_knn, item_knn
      meta:
        verbose: True
        save_recs: False
      neighbors: 50
      similarity: cosine
"""
    algHTItemKNN = """    ItemKNN:    # knn, item_knn, item_knn
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 20
        hyper_opt_alg: tpe
        save_recs: False
      neighbors: [uniform, 1, 75]
      similarity: [cosine, jaccard, dice, euclidean]
      implementation: classical
"""
    algHTGridItemKNN = """    ItemKNN:    # knn, item_knn, item_knn
          meta:
            verbose: True
            save_recs: False
            save_recs: False
          neighbors: [1, 3, 5, 8, 15, 20, 30, 45, 60, 75]
          similarity: [cosine]
          implementation: classical
    """
    algSVDpp = """    SVDpp:
      meta:
        verbose: True
        save_recs: False
"""
    algiALS = """    iALS:       # latent_factor_models, iALS, iALS
      meta:
        verbose: True
        save_recs: False
"""
    algHTIALS = """    iALS:    # latent_factor_models, iALS, iALS
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 20
        hyper_opt_alg: tpe
      factors: [uniform, 10, 50]
      alpha: [uniform, 1, 5]
      reg: [uniform, 10e-4, 10e-1]
"""
    algUserKNN = """    UserKNN:   # autoencoders, EASE_R, ease_r
      meta:
        save_recs: False
      neighbors: 50
      similarity: cosine
      implementation: classical
"""
    algHTUserKNN = """    UserKNN:    # knn, item_knn, item_knn
          meta:
            verbose: True
            save_recs: False
            hyper_max_evals: 20
            hyper_opt_alg: tpe
            save_recs: False
          neighbors: [uniform, 1, 75]
          similarity: [cosine, jaccard, dice, euclidean]
          implementation: classical
"""
    algHTGridUserKNN = """    UserKNN:    # knn, item_knn, item_knn
          meta:
            verbose: True
            save_recs: False
            save_recs: False
          neighbors: [1, 3, 5, 8, 15, 20, 30, 45, 60, 75]
          similarity: [cosine]
          implementation: classical
"""
    algEASER = """    EASER:   # autoencoders, EASE_R, ease_r
      meta:
        verbose: True
        save_recs: False
"""
    algLightGCN = """    LightGCN:   # graph_based, lightgcn, LightGCN
      meta:
        verbose: True
        save_recs: False
"""
    algSVDpp = """    SVDpp:   # latent_factor_models, SVDpp
      meta:
        verbose: True
        save_recs: False
"""
    algPMF = """    PMF:   # latent_factor_models, PMF
      meta:
        verbose: True
        save_recs: False
    """
    algDMF = """    DMF:   # neural, DMF
      meta:
        verbose: True
        save_recs: False
    """
    algNAIS = """    NAIS:   # neural, NAIS
      meta:
        verbose: True
        save_recs: False
    """

    @staticmethod
    def getAlgorithmByAlgID(algID:str):
        if algID == "IALS":
            return ExperimentBuilder.algiALS
        elif algID == "HTIALS":
            return ExperimentBuilder.algHTIALS
        elif algID == "ItemKNN":
            return ExperimentBuilder.algItemKNN
        elif algID == "HTItemKNN":
            return ExperimentBuilder.algHTItemKNN
        elif algID == "HTGridItemKNN":
            return ExperimentBuilder.algHTGridItemKNN
        elif algID == "UserKNN":
            return ExperimentBuilder.algUserKNN
        elif algID == "HTUserKNN":
            return ExperimentBuilder.algHTUserKNN
        elif algID == "HTGridUserKNN":
            return ExperimentBuilder.algHTGridUserKNN
        elif algID == "EASER":
            return ExperimentBuilder.algEASER
        elif algID == "LightGCN":
            return ExperimentBuilder.algLightGCN
        elif algID == "SVDpp":
            return ExperimentBuilder.algSVDpp
        elif algID == "PMF":
            return ExperimentBuilder.algPMF
        elif algID == "DMF":
            return ExperimentBuilder.algDMF
        elif algID == "NAIS":
            return ExperimentBuilder.algNAIS
        return None

    @staticmethod
    def getExperimentStr(datasetID:str, datasetPart:str, granularity:str, fold:str, algorithms:List[str], algID:List[str]):

        experimentStr = ExperimentBuilder.experimentDef.replace("<datasetID>", datasetID)
        experimentStr = experimentStr.replace("<datasetPart>", datasetPart)
        experimentStr = experimentStr.replace("<stars>", granularity)
        experimentStr = experimentStr.replace("<fold>", fold)
        experimentStr = experimentStr.replace("<alg>", algID[0])

        for algorithmI in algorithms:
            experimentStr = experimentStr + algorithmI

        return experimentStr


def generateBatches():

    #datasetID:List[str] = ["libraryThing", "ml1m", "ml25mSel2016"]
    datasetID:List[str] = ["ml25mSel2016"]
    datasetID:List[str] = ["libraryThingSel20"]
    #agsIds:List[str] = ["IALS", "HTIALS", "ItemKNN", "HTItemKNN", "UserKNN", "EASER", "LightGCN", "SVDpp", "PMF", "DMF", "NAIS"]
    #agsIds:List[str] = ["HTItemKNN", "HTUserKNN"]
    agsIds:List[str] = ["HTGridItemKNN", "HTGridUserKNN"]
    #agsIds:List[str] = ["DMF", "NAIS"]
    #agsIds: List[str] = ["PMF"]
    datasetFolds:List[int] = [0, 1, 2, 3, 4]
    datasetParts:List[int] = [20, 50, 70, 80, 90, 95, 100]
    datasetStarts:List[int] = [1, 2, 3, 5, 7, 10]

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
                    for algIdI in agsIds:
                        batchID:str = datasetIdI + "-Alg" + algIdI + "-Part" + datasetPartStrI + "-Stars" + startsStrI + "-Fold" + str(datasetFoldI)
                        algStrI:str = ExperimentBuilder.getAlgorithmByAlgID(algIdI)
                        experimentStr:str = ExperimentBuilder.getExperimentStr(
                                        datasetIdI, datasetPartStrI, startsStrI, str(datasetFoldI), [algStrI], [algIdI])
                        #getAlgorithmByAlgID
                        f = open("./batches" + os.sep + batchID + ".yml", "wt")
                        f.write(experimentStr)
                        f.close()


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    generateBatches()
