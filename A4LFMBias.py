#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/25 2:21
# @Author  : Paul1998
# @File    : A4LFM.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import json
import math
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from operator import itemgetter
import gc

np.seterr(divide='ignore', invalid='ignore')

file = Path("./Yelp data/review.json")
iterations = sum([1 for i in open(file, "r")])
frame = []
with open(file) as file:
    for idx, line in tqdm(enumerate(file), total=iterations):
        temp = json.loads(line.strip())
        frame.append(temp)
df = pd.DataFrame(frame).sort_values(by=['date'])
splitNum = math.ceil(len(df) * 0.8)
testData = df[splitNum:]
splitNum2 = math.ceil(len(df) * 0.7)
trainData = df[0:splitNum2]
validData = df[splitNum2: splitNum]


def mapReal2Index(setList):
    mapDict = dict()
    mapReverse = dict()
    step = 0
    for i in setList:
        mapDict[i] = step  # map real to index
        mapReverse[step] = i  # map real to index
        step += 1
    return mapDict, mapReverse


mapBusiness, mapReverseBusiness = mapReal2Index(set(df.business_id.tolist()))
mapUser, mapReverseUser = mapReal2Index(set(df.user_id.tolist()))
# Build mapping to rename
trainBusiness = np.array(itemgetter(*trainData.business_id.tolist())(mapBusiness))
trainUser = np.array(itemgetter(*trainData.user_id.tolist())(mapUser))
trainStar = np.array(trainData.stars.tolist(), dtype=float)
testBusiness = np.array(itemgetter(*testData.business_id.tolist())(mapBusiness))
testUser = np.array(itemgetter(*testData.user_id.tolist())(mapUser))
testStar = np.array(testData.stars.tolist(), dtype=float)
validBusiness = np.array(itemgetter(*validData.business_id.tolist())(mapBusiness))
validUser = np.array(itemgetter(*validData.user_id.tolist())(mapUser))
validStar = np.array(validData.stars.tolist(), dtype=float)
userSize = len(set(df.user_id.tolist()))
businessSize = len(set(df.business_id.tolist()))
matrix = sparse.csr_matrix((trainStar, (trainUser, trainBusiness)),
                           shape=(userSize, businessSize))

# Latent factor model
## latent factor model with number of latent factors k
kList = [8, 16, 32, 64]


def initialPQ(userSize, k, businessSize):
    q = np.random.random([userSize, k])
    p = np.random.random([businessSize, k])
    return q, p


def SGD(matrix, trainTuple, q, p, bi, bj, lambd1=0.3, lambd2=0.3, lambd3=0.3, lambd4=0.3, elta=0.01):
    p[p < 0] = 0
    q[q < 0] = 0
    bi[bi < 0] = 0
    bj[bj < 0] = 0

    for user, business in trainTuple:
        temp = matrix[user, business] - np.dot(p[business, ], q[user, ]) - bg - bi[user] - bj[business]
        temp_p = p[business, ] - elta * (-2 * temp * q[user, ] + 2 * lambd1 * p[business, ])
        temp_q = q[user, ] - elta * (-2 * temp * p[business, ] + 2 * lambd2 * q[user, ])
        temp_bi = bi[user] - elta * (-2 * temp + 2 * lambd3 * bi[user])
        temp_bj = bj[business] - elta * (-2 * temp + 2 * lambd4 * bj[business])

        p[business, ] = temp_p
        q[user, ] = temp_q
        bi[user] = temp_bi
        bj[business] = temp_bj
    gc.collect()
    return p, q, bi, bj


def RMSE(tupleData, trueStar, q, p, bi, bj):
    pred = []
    p[p < 0] = 0
    q[q < 0] = 0
    
    for i in range(len(trueStar)):
        user, business = tupleData.T[0][i], tupleData.T[1][i]
        pred.append(np.dot(p[business, ], q[user, ]) + bg + bi[user] + bj[business])
    predicted = np.array(pred)
    MSE = mean_squared_error(predicted, trueStar)
    return math.sqrt(MSE)


def checkNew(data, size):
    tempDict = dict()
    step = 0
    for i in data:
        tempDict[i] = step  # use dictionary to find new
    new = []
    for i in range(size):
        key = tempDict.get(i)
        if key is None:
            new.append(i)
    return new


def LFMtrain(kList, matrix, userSize, businessSize, epochs=20):
    newUser = checkNew(set(trainUser), userSize)
    newBusiness = checkNew(set(trainBusiness), businessSize)
    for k in kList:
        # random init p, q
        q, p = initialPQ(userSize, k, businessSize)
        bi = np.random.random(userSize)
        bj = np.random.random(businessSize)
        # run SGD for 20 epochs
        for epoch in tqdm(range(epochs)):
            p, q, bi, bj = SGD(matrix, trainTuple, q, p, bi, bj)
        q[newUser, ] = 0
        p[newBusiness, ] = 0
        bi[newUser] = 0
        bj[newBusiness] = 0
        validRMSE = RMSE(validTuple, validStar, q, p, bi, bj)
        trainRMSE = RMSE(trainTuple, trainStar, q, p, bi, bj)
        testRMSE = RMSE(testTuple, testStar, q, p, bi, bj)

        print('When k = %d: RMSE valid is %f, RMSE train is %f, RMSE test is %f'
              % (k, validRMSE, trainRMSE, testRMSE))


bg = np.mean(trainStar)
trainTuple = np.vstack((trainUser, trainBusiness)).T
testTuple = np.vstack((testUser, testBusiness)).T
validTuple = np.vstack((validUser, validBusiness)).T
LFMtrain(kList, matrix, userSize, businessSize)
