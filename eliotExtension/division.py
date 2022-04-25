import zipfile
import io
import requests
import os
import numpy as np
import pandas as pd


def divideMl1mDataset():
    print("Divide Ml1m Dataset")

    ratings_df = pd.read_csv('data/ml1m/ratings.csv',
                 dtype= {'userId':np.int32,
                         'movieId':np.int32,
                         'rating':np.float64,
                         'timestamp':np.int64},
                 header=0, #skiprows=1
                 names= ['userId','movieId','rating','timestamp'])

    ratingsStars10_df = ratings_df.copy()
    ratingsStars10_df["rating"] = 2 * ratingsStars10_df["rating"]

    ratingsStars02_df, ratingsStars03_df, ratingsStars05_df, ratingsStars08_df = divideGeneralDataset(ratingsStars10_df)

    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for datasetPartI in datasetParts:
        if datasetPartI < 100:
            partI = "Part0" + str(datasetPartI)
        else:
            partI = "Part" + str(datasetPartI)

        ratingsStars10Sel_df = ratingsStars10_df.copy().sample(frac=datasetPartI/100)
        ratingsStars08Sel_df = ratingsStars08_df.copy().sample(frac=datasetPartI/100)
        ratingsStars05Sel_df = ratingsStars05_df.copy().sample(frac=datasetPartI/100)
        ratingsStars03Sel_df = ratingsStars03_df.copy().sample(frac=datasetPartI/100)
        ratingsStars02Sel_df = ratingsStars02_df.copy().sample(frac=datasetPartI/100)

        ratingsStars10Sel_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars10.tsv", sep = "\t", index=False, header=False)
        ratingsStars08Sel_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars08.tsv", sep = "\t", index=False, header=False)
        ratingsStars05Sel_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars05.tsv", sep = "\t", index=False, header=False)
        ratingsStars03Sel_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars03.tsv", sep = "\t", index=False, header=False)
        ratingsStars02Sel_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars02.tsv", sep = "\t", index=False, header=False)



def divideLTDataset():
    print("Divide LT Dataset")

    ratingsStars10_df = pd.read_csv('data/libraryThing/testset.tsv',
                 dtype= {'userId':np.int32,
                         'itemId':np.int32,
                         'rating':np.float64},
                 header=0, #skiprows=1
                 names= ['userId','itemId','rating'],
                 delim_whitespace=True)

    ratingsStars02_df, ratingsStars03_df, ratingsStars05_df, ratingsStars08_df = divideGeneralDataset(ratingsStars10_df)

    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for datasetPartI in datasetParts:
        if datasetPartI < 100:
            partI = "Part0" + str(datasetPartI)
        else:
            partI = "Part" + str(datasetPartI)

        ratingsStars10Sel_df = ratingsStars10_df.copy().sample(frac=datasetPartI/100)
        ratingsStars08Sel_df = ratingsStars08_df.copy().sample(frac=datasetPartI/100)
        ratingsStars05Sel_df = ratingsStars05_df.copy().sample(frac=datasetPartI/100)
        ratingsStars03Sel_df = ratingsStars03_df.copy().sample(frac=datasetPartI/100)
        ratingsStars02Sel_df = ratingsStars02_df.copy().sample(frac=datasetPartI/100)

        ratingsStars10Sel_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars10.tsv", sep = "\t", index=False, header=False)
        ratingsStars08Sel_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars08.tsv", sep = "\t", index=False, header=False)
        ratingsStars05Sel_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars05.tsv", sep = "\t", index=False, header=False)
        ratingsStars03Sel_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars03.tsv", sep = "\t", index=False, header=False)
        ratingsStars02Sel_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars02.tsv", sep = "\t", index=False, header=False)


    #print(ratingsStars10_df.head(20))


def divideGeneralDataset(ratings10_df):
    ratings08_df = ratings10_df.copy()  # 3U4 and 6U7    [3,4]->3, 5->4, 6->5, [7,8]->6, 9->7, 10->8
    ratings08_df["rating"] = ratings08_df["rating"].apply(lambda x: x - 1 if x in [4, 5, 6, 7] else x)
    ratings08_df["rating"] = ratings08_df["rating"].apply(lambda x: x - 2 if x in [8, 9, 10] else x)

    ratings05_df = ratings10_df.copy()  # 1U2 and 2U3 and 3U4 and 5U6 and 7U8 and 9U10
    ratings05_df["rating"] = round(ratings05_df["rating"] / 2)

    ratings03_df = ratings10_df.copy()  # 1U2U3U4 and 5U6 and 7U8U9U10
    ratings03_df.loc[ratings03_df["rating"] <= 4, "rating"] = 1
    ratings03_df.loc[(ratings03_df["rating"] > 4) & (ratings03_df["rating"] <= 6), "rating"] = 2
    ratings03_df.loc[ratings03_df["rating"] > 6, "rating"] = 3

    ratings02_df = ratings10_df.copy()  # 1U2U3U4U5 and 6U7U8U9U10
    ratings02_df.loc[ratings02_df["rating"] <= 5, "rating"] = 1
    ratings02_df.loc[ratings02_df["rating"] > 5, "rating"] = 2

    return (ratings02_df, ratings03_df, ratings05_df, ratings08_df)


def generateDatasets():
    divideMl1mDataset()
    divideLTDataset()


if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    generateDatasets()
