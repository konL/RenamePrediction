from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
def load_attributes(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    # cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=",", header=0)

    # determine (1) the unique zip codes and (2) the number of data
    # points with each zip code
    # zipcodes = df["zipcode"].value_counts().keys().tolist()
    # counts = df["zipcode"].value_counts().tolist()
    #
    # # loop over each of the unique zip codes and their corresponding
    # # count
    # for (zipcode, count) in zip(zipcodes, counts):
    #     # the zip code counts for our housing dataset is *extremely*
    #     # unbalanced (some only having 1 or 2 houses per zip code)
    #     # so let's sanitize our data by removing any houses with less
    #     # than 25 houses per zip code
    #     if count < 25:
    #         idxs = df[df["zipcode"] == zipcode].index
    #         df.drop(idxs, inplace=True)

    # return the data frame
    return df


def process_attributes(df, x):
    #   数据标准归一化
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X = ss.fit_transform(x)

    return X
#然后将连续的和分类的特性连接起来并返回(第53-57行)。