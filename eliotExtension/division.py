import zipfile
import io
import requests
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

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

    ratingsStars_dic:Dict[DataFrame] = divideGeneralDataset(ratingsStars10_df)

    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for datasetPartI in datasetParts:
        partI = "Part0" + str(datasetPartI)
        if datasetPartI == 100:
            partI = "Part" + str(datasetPartI)
        for starsI, ratingsStarI_df in ratingsStars_dic.items():
            starsStrI:str = "0" + str(starsI)
            if starsI == 10:
                starsStrI: str = str(starsI)
            ratingsStarSelI_df = ratingsStarI_df.copy().sample(frac=datasetPartI/100)
            ratingsStarSelI_df.to_csv("./data/ml1m/ml1m-" + partI + "-Stars" + starsStrI + ".tsv", sep = "\t", index=False, header=False)



def divideLTDataset():
    print("Divide LT Dataset")

    ratingsStars10_df = pd.read_csv('data/libraryThing/testset.tsv',
                 dtype= {'userId':np.int32,
                         'itemId':np.int32,
                         'rating':np.float64},
                 header=0, #skiprows=1
                 names= ['userId','itemId','rating'],
                 delim_whitespace=True)

    ratingsStars_dic:Dict[DataFrame] = divideGeneralDataset(ratingsStars10_df)

    datasetParts:List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for datasetPartI in datasetParts:
        if datasetPartI < 100:
            partI = "Part0" + str(datasetPartI)
        else:
            partI = "Part" + str(datasetPartI)
        for starsI, ratingsStarI_df in ratingsStars_dic.items():
            starsStrI:str = "0" + str(starsI)
            if starsI == 10:
                starsStrI: str = str(starsI)
            ratingsStarSelI_df = ratingsStarI_df.copy().sample(frac=datasetPartI/100)
            ratingsStarSelI_df.to_csv("./data/libraryThing/libraryThing-" + partI + "-Stars" + starsStrI + ".tsv", sep = "\t", index=False, header=False)



def divideGeneralDataset_(ratings10_df):
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


def divideGeneralDataset(ratings10_df):
    print(ratings10_df.head(20))

    ratingsNorm_df = ratings10_df.copy()
    # normalisation fnc    norm_rating = (rating - np.min(rating)) / (np.max(rating)-np.min(rating))
    ratingsNorm_df["rating"] = ratingsNorm_df["rating"].apply(lambda rating: (rating -1) / (10-1))

    # normalisation fnc np.round(norm_rating * (k_max - k_min) + k_min)
    r_dict:Dict = {}
    for maxStarI in [2,3,4,5,6,7,8,9,10]:
        ratingsI_df = ratingsNorm_df.copy()
        ratingsI_df["rating"] = ratingsI_df["rating"].apply(lambda nRating: np.round(nRating * (maxStarI - 1) +1))
        r_dict[maxStarI] = ratingsI_df
    return r_dict


def generateDatasets():
    divideMl1mDataset()
    divideLTDataset()


def mappingStarsInReduction():
    rating = [0,1,2,3,4,5,6,7,8,9]
    norm_rating = (rating - np.min(rating)) / (np.max(rating)-np.min(rating))
    print(norm_rating)

    k_max = 7
    k_min = 1
    reduced_rating = np.round(norm_rating * (k_max - k_min) + k_min)
    print(reduced_rating)



if __name__ == "__main__":
    os.chdir('../')
    print(os.getcwd())

    generateDatasets()

    #mappingStarsInReduction()