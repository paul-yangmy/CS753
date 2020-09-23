import sys
import pandas as pd
import time
import random
import numpy as np

random.seed(123)
summary = pd.read_csv('docword.kos.txt', sep=' ', nrows=3, header=None)
[doc, word, length] = summary[0]
df = pd.read_csv('docword.kos.txt', sep=' ', skiprows=[0, 1, 2], header=None)
columns_name = ["docID", "wordID", "Frequency"]
df.columns = columns_name
df.Frequency = 1  # count = 1 for each pair
dictionary = {}
for i in set(df.docID):
    dictionary[i] = df[df.docID == i].wordID.values.tolist()

df2 = pd.read_csv('similarity.txt', sep=' ', header=None)
columns_name2 = ["docID1", "docID2", "Similarity"]
df2.columns = columns_name2
actualJaccard = {}
for i in set(df2.docID1):
    actualJaccard[i] = df2[df2.docID1 == i].Similarity.values.tolist()


def minHash(d):
    def universalHash(d):
        n = word
        p = 1000003  # large prime
        permutations = [[] for row in range(d)]
        for i in range(d):
            a = random.randint(0, p)
            b = random.randint(0, p)
            temp = []
            for w in range(word):
                bucket = (((a * w) + b) % p) % n
                temp.append(bucket)
            permutations[i] = temp
        return permutations

    def minusOne(x):
        x -= 1
        return x

    hashFunction = universalHash(d)
    maxNum = sys.maxsize
    signature = [[maxNum for i in range(doc)] for j in range(d)]
    for s in range(doc):
        oneRow = list(map(minusOne, dictionary[s + 1]))
        for index in oneRow:
            for j in range(d):
                if hashFunction[j][index] <= signature[j][s]:
                    signature[j][s] = hashFunction[j][index]
    return signature


d = 100
t = 0.6
p = 0.1
start = time.time()
signatureMatrix = np.array(minHash(d))
bList = []
rList = []
for i in range(1, d + 1):
    if d % i == 0:
        bList.append(i)
        rList.append(d / i)
# false negative at most 10% means it's a candidate probability at least 90%
candidate = []
for i in range(len(bList)):
    falseNegative = (1 - t ** rList[i]) ** bList[i]
    if falseNegative < p:
        candidate.append([bList[i], rList[i], falseNegative])

def hashFamily(p, b, r):
    # create a r * b matrix to store all hash parameters for ecah band
    an = []
    for i in range(r):
        an.append([random.randint(1, p) for k in range(b)])
    bn = [random.randint(1, p) for k in range(b)]
    return np.array(an).T, bn

def randomHash(M, b, r):
    p = 1000003
    n = 500000
    bucket = []
    # using hash ((bn + a1*x1 + a2*x2 + ... till ar*xr) mod p ) mod n
    [An, Bn] = hashFamily(p, b, r)
    for col in range(len(M[0])):  # calculate for each col for all bands
        generalHash = []
        docSignature = np.array(signatureMatrix[:, col]).reshape(b, r)
        for i in range(b):
            generalHash.append(((np.dot(docSignature[i], An[i]) + Bn[i]) % p) % n)  # matrix multiplication
        bucket.append(generalHash)
    return np.array(bucket).T


# using the first b, r
[b, r, prob] = candidate[0]
# hash the signature divide by bands into one number
hashBucket = randomHash(signatureMatrix, int(b), int(r))

pairsExactNum = 0
# calculate the exact similarity in brute force result
for key in actualJaccard:
    # use lambda expression to avoid using loop
    pairsExactNum += len(list(filter(lambda x: x <= 0.3, actualJaccard[key])))

pairs = []
pairsNum = 0
candidatePairsExactNum30 = 0
candidatePairsExactNum60 = 0
for col in range(len(hashBucket[0]) - 1):
    # select column from doc1 ..  to doc3429, using all remaining columns to minus this column
    minusCols = (hashBucket[:, col + 1:].T - hashBucket[:, col]).T
    # if exist 0 means for this two cols they can be hash into at least one bucket
    judgeZero = np.any(minusCols == 0, axis=0)
    # find 0 index
    colIndex = np.where(judgeZero == True)[0]
    if len(colIndex) == 0:
        continue
    else:
        # because I only store half of comparing e.g only (1,2) not store (2,1)
        # my result for Task1 is doc1 compared to doc2 ... doc3430
        #                        doc2 compared to doc3 ... doc3430
        # so, the index is of 0 is what I stored in dictionary
        for col2 in colIndex:
            # add candidate pairs
            pairs.append([col, col2])
            pairsNum += 1
            # for this candidate pairs, find the exact similarity from brute force result
            probability = actualJaccard[col + 1][col2]
            if probability < 0.6:
                candidatePairsExactNum60 += 1
            if probability <= 0.3:
                candidatePairsExactNum30 += 1


end = time.time()
print("Running time is ", end - start)
falseCandidateRatio = candidatePairsExactNum60 / pairsNum
dissimilarPair = candidatePairsExactNum30 / pairsExactNum
print("False candidate ratio is ", falseCandidateRatio)
print("The probability that a dissimilar pair with Jaccard <= 0:3 is a candidate pair is ",dissimilarPair)
