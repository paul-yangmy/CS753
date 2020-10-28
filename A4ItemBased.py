#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 16:25
# @Author  : Paul1998
# @File    : A4ItemBased.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from operator import itemgetter
import time
import itertools

np.seterr(divide='ignore', invalid='ignore')

file = Path("./Yelp data/review.json")
iterations = sum([1 for i in open(file, "r")])
frame = []
with open(file) as file:
    for idx, line in tqdm(enumerate(file), total=iterations):
        temp = json.loads(line.strip())  # load data by row
        frame.append(temp)
df = pd.DataFrame(frame).sort_values(by=['date'])  # sort by date
splitNum = math.ceil(len(df) * 0.8)  # get split number
trainData = df[0:splitNum]
testData = df[splitNum:]


def mapReal2Index(setList):
    mapDict = dict()
    mapReverse = dict()
    step = 0
    for i in setList:
        mapDict[i] = step  # map real to index
        mapReverse[step] = i  # map real to index
        step += 1
    return mapDict, mapReverse


# map for all to create dictionary
mapBusiness, mapReverseBusiness = mapReal2Index(set(df.business_id.tolist()))
mapUser, mapReverseUser = mapReal2Index(set(df.user_id.tolist()))

# Item-based collaborative filtering
## cosine similarity

# Build mapping to rename
trainBusiness = np.array(itemgetter(*trainData.business_id.tolist())(mapBusiness))
trainUser = np.array(itemgetter(*trainData.user_id.tolist())(mapUser))
trainStar = np.array(trainData.stars.tolist(), dtype=float)
matrix = sparse.csr_matrix((trainStar, (trainBusiness, trainUser)),
                           shape=(len(set(df.business_id.tolist())),
                                  len(set(df.user_id.tolist()))))

start1 = time.time()
similarity = cosine_similarity(matrix)
end1 = time.time()
print("The time for computing all pairwise similarities is %.4fs" % (end1 - start1))
q1aBusiness1 = ["rjZ0L-P1SRLJfiK5epnjYg",
                "6H8xfhoZ2IGa3eNiY5FqLA",
                "rfwJFFzW6xW2qYfJh14OTA",
                "0QSnurP5Ibor2zepJmEIlw"]
q1aBusiness2 = ["cL2mykvopw-kodSyc-z5jA",
                "XZbuPXdyA0ZtTu3AzqtQhg",
                "G58YATMKnn-M-RUDWg3lxw",
                "6-lmL3sC-axuh8y1SPSiqg"]
q1aBusiness1Map = np.array(itemgetter(*q1aBusiness1)(mapBusiness))
q1aBusiness2Map = np.array(itemgetter(*q1aBusiness2)(mapBusiness))
for i in range(len(q1aBusiness1Map)):
    a = q1aBusiness1Map[i]
    b = q1aBusiness2Map[i]
    cosSim = similarity[q1aBusiness1Map[i], q1aBusiness2Map[i]]
    print("The cosine similarity for Business pair ( %s, %s ) is %f" %
          (q1aBusiness1[i], q1aBusiness2[i], cosSim))

## matrix multiplication

# R be the m * n rating matrix,
# where m and n are the number of users and number of businesses, respectively
start2 = time.time()
R = matrix.transpose()
RTR = R.T * R  # numerator
# L2-norm of each business
eachDenom = np.array([np.sqrt(RTR.diagonal())])
denominator = np.matmul(eachDenom.T, eachDenom)
similarity2 = RTR / denominator
end2 = time.time()
print("The time for computing all pairwise similarities is %.4fs by matrix multiplication" % (end2 - start2))
# for i in range(len(q1aBusiness1Map)):
#     cosSim = similarity2[q1aBusiness1Map[i], q1aBusiness2Map[i]]
#     print("The cosine similarity for Business pair ( %s, %s ) is %f" %
#           (q1aBusiness1[i], q1aBusiness2[i], cosSim))

## RMSE

# get test data
testBusiness = np.array(itemgetter(*testData.business_id.tolist())(mapBusiness))
testUser = np.array(itemgetter(*testData.user_id.tolist())(mapUser))
testStar = np.array(testData.stars.tolist(), dtype=float)
# predict
testTuple = np.vstack((testUser, testBusiness)).T
similarity = similarity - np.diag(np.diag(similarity))

# def predStars(testTuple, R, similarity):
#     pred = []
#     K = 20  # topK
#     for user, business in tqdm(testTuple):  # for each (user, business) tuple
#         cosineScores = similarity[business]  # target row
#         ratingsScores = R[user].toarray().squeeze()  # true star for user
#         nonNULL = np.nonzero(ratingsScores)  # rated by the user
#         cosineScores = cosineScores[nonNULL]
#         topK = np.argsort(-cosineScores)[0: K]
#         ratingsScores = ratingsScores[nonNULL][topK] # rated by the user and Top 20
#         cosineScores = cosineScores[topK]
#         predRate = np.dot(ratingsScores, cosineScores) / cosineScores.sum()
#         pred.append(predRate)
#     return np.array(pred)


# Batch process computing increases speed
def predStars(testTuple, R, similarity):
    pred = []
    K = 20
    batch_size = 1024
    for start in tqdm(range(0, len(testTuple), batch_size)):  # for each batch with 1024 (user, business) tuples
        end = min(start + batch_size, len(testTuple))
        batch_user, batch_business = testTuple[start: end, 0], testTuple[start: end, 1]
        cosineScores = similarity[batch_business].copy()  # avoid changing original data
        ratingsScores = R[batch_user].toarray()
        cosineScores[ratingsScores == 0] = 0  # set 0 for new
        topK = np.argsort(cosineScores)[:, -K:]
        idx = np.arange(end - start).reshape(-1, 1)
        ratingsScores = ratingsScores[idx, topK].reshape(-1, 1, K)  # reshape
        cosineScores = cosineScores[idx, topK].reshape(-1, K, 1)
        predRate = np.matmul(ratingsScores, cosineScores).squeeze() / cosineScores.squeeze().sum(axis=1)
        pred.append(predRate)
    pred = np.concatenate(pred)
    return pred


predicted = predStars(testTuple, R, similarity)
predicted[np.isnan(predicted)] = 0  # set 0 for new business or user
MSE = mean_squared_error(predicted, testStar)
print("RMSE is ", math.sqrt(MSE))

## incorporating the bias
bg = trainStar.mean()  # average over all transactions
biTotal = np.array(np.sum(R, axis=0)).squeeze()
buTotal = np.array(np.sum(R, axis=1)).squeeze()
temp = R.T.getnnz(axis=0)
buTotal[np.where(temp == 0)] = bg  # set for new user
temp[np.where(temp == 0)] = 1  # avoid denominator is 0
buTotal = buTotal / temp - bg  # bias on a user u over all businesses

temp = R.T.getnnz(axis=1)
biTotal[np.where(temp == 0)] = bg  # set for new user
temp[np.where(temp == 0)] = 1  # avoid denominator is 0
biTotal = biTotal / temp - bg  # bias on a business b over all users

# def predStarsWithBias(testTuple, R, similarity, buTotal, biTotal):
#     pred = []
#     K = 20
#     for user, business in tqdm(testTuple):
#         ratingsScoresByUser = R[user].toarray().squeeze()
#         nonNULLByUser = np.nonzero(ratingsScoresByUser)
#         bi = biTotal[business]
#         bu = buTotal[user]
#         bui = bg + bu + bi
#
#         cosineScores = similarity[business][nonNULLByUser]
#         topK = np.argsort(-cosineScores)[0: K]
#         ratingsScores = ratingsScoresByUser[nonNULLByUser][topK]
#         cosineScores = cosineScores[topK]
#         bki = bg + bu + biTotal[nonNULLByUser][topK]
#         temp = np.dot(ratingsScores - bki, cosineScores) / cosineScores.sum()
#         if np.isnan(temp):
#             temp = 0
#         predRate = temp + bui
#         pred.append(predRate)
#     return np.array(pred)


# Batch process computing increases speed
def predStarsWithBias(testTuple, R, similarity, buTotal, biTotal):
    pred = []
    K = 20
    batch_size = 1024
    for start in tqdm(range(0, len(testTuple), batch_size)):
        end = min(start + batch_size, len(testTuple))
        batch_user, batch_business = testTuple[start: end, 0], testTuple[start: end, 1]
        length = len(batch_user)  # to reshape bias
        cosineScores = similarity[batch_business].copy()
        ratingsScores = R[batch_user].toarray()
        cosineScores[ratingsScores == 0] = 0

        # repeat to calculate
        bi = np.tile(biTotal[batch_business].reshape(length, 1), K)
        bu = np.tile(buTotal[batch_user].reshape(length, 1), K)
        bui = (bg + bu + bi).reshape(-1, 1, K)

        topK = np.argsort(cosineScores)[:, -K:]
        idx = np.arange(end - start).reshape(-1, 1)
        ratingsScores = ratingsScores[idx, topK].reshape(-1, 1, K)
        cosineScores = cosineScores[idx, topK].reshape(-1, K, 1)
        bkiRepeat = np.tile(biTotal, length).reshape(length, len(biTotal))
        # for user with topK business, get these business' bi from bkiRepeat
        bki = (bg + bu).reshape(-1, 1, K) + bkiRepeat[idx, topK].reshape(-1, 1, K)
        temp = np.matmul(ratingsScores - bki, cosineScores).squeeze() / cosineScores.squeeze().sum(axis=1)
        newDetection = (np.isnan(temp).reshape(1, length)).squeeze()  # if denominator is 0 we would get nan
        # nan means new user or business, add bui for them
        temp[newDetection] = 0  #
        predRate = temp + bui[:, 0, 0]
        pred.append(predRate)
    pred = np.concatenate(pred)
    return pred


predicted2 = predStarsWithBias(testTuple, R, similarity, buTotal, biTotal)
MSE2 = mean_squared_error(predicted2, testStar)
print("RMSE is %f by incorporating the bias" % math.sqrt(MSE2))
