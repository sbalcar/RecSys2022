import zipfile
import io
import os
import numpy as np
import pandas as pd
from typing import List

from elliot.run import run_experiment

def getExperimentStr(datasetID:str, datasetPart:str, granularity:str):

    experiment:str = """experiment:
  dataset: <datasetID>-Part<datasetPart>-Gran<granularity>
#  path_output_rec_result: ../results/<datasetID>/<granularity>/recs
#  path_output_rec_weight: ../results/<datasetID>/<granularity>/weight
#  path_output_rec_performance: ../results/<datasetID>/<granularity>/performance
  path_log_folder: ../results/<datasetID>-Part<datasetPart>-Gran<granularity>/log
  data_config:
    strategy: dataset
    dataset_path: ../data/<datasetID>/<datasetID>-Part<datasetPart>-Gran<granularity>.tsv
  splitting:
    save_on_disk: True
    save_folder: ../data/<datasetID>-Part<datasetPart>/split/<granularity>
    test_splitting:
      strategy: random_cross_validation
      timestamp: best
      test_ratio: 0.2
      leave_n_out: 1
      folds: 5
      validation_splitting:
        strategy: random_cross_validation
        timestamp: best
        test_ratio: 0.2
        leave_n_out: 1
        folds: 5
  top_k: 10
  evaluation:
    simple_metrics: [nDCG, ItemCoverage, HR, RMSE]
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

    experiment = experiment.replace("<datasetID>", datasetID)
    experiment = experiment.replace("<granularity>", granularity)
    experiment = experiment.replace("<datasetPart>", datasetPart)
    #experiment = experiment + algItemKNN
    experiment = experiment + algiALS

    return experiment

def generateBatches():

    datasetID:List[str] = ["libraryThing", "ml1m"]
    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    granularity:List[str] = ["02", "03", "05", "08", "10"]

    for datasetIdI in datasetID:
        for datasetPartI in datasetParts:
            for granularityI in granularity:
                datasetPartStrI = str(datasetPartI)
                if datasetPartI < 100:
                    datasetPartStrI = "0" + str(datasetPartI)
                batchID:str = datasetIdI + "-Part" + datasetPartStrI + "-Gran" + granularityI
                experimentStr:str = getExperimentStr(datasetIdI, datasetPartStrI, granularityI)

                f = open("./batches/" + batchID + ".yml", "wt")
                f.write(experimentStr)
                f.close()

#    run_experiment("./config/pokus.yml")


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    generateBatches()
