import sys
import pandas as pd
import time
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(123)  # set seed
summary = pd.read_csv('docword.kos.txt', sep=' ', nrows=3, header=None)
[doc, word, length] = summary[0]
df = pd.read_csv('docword.kos.txt', sep=' ', skiprows=[0, 1, 2], header=None)
columns_name = ["docID", "wordID", "Frequency"]
df.columns = columns_name
df.Frequency = 1  # count = 1 for each pair
dictionary = {}
for i in set(df.docID):
    dictionary[i] = df[df.docID == i].wordID.values.tolist()

# load the brute force result
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
            # create random a & b belongs to [0,p]
            a = random.randint(0, p)
            b = random.randint(0, p)
            temp = []
            for w in range(word):
                bucket = (((a * w) + b) % p) % n
                temp.append(bucket)
            permutations[i] = temp
        return permutations

    def minusOne(x):
        x -= 1  # for every wordID, I minus 1 to get its index
        return x

    start = time.time()
    hashFunction = universalHash(d)
    maxNum = sys.maxsize  # positive infinity
    signature = [[maxNum for i in range(doc)] for j in range(d)]  # init
    for s in range(doc):
        # all minus 1 to get the row index, map is quicker than for loop
        oneRow = list(map(minusOne, dictionary[s + 1]))
        for index in oneRow:
            for j in range(d):
                if hashFunction[j][index] <= signature[j][s]:
                    signature[j][s] = hashFunction[j][index]  # replace by the min one
    end = time.time()
    print("MinHash running time is %fs for %d hash functions" % ((end - start), d))
    return signature


MAE = []
for num in range(10, 101, 10):
    start2 = time.time()
    signatureMatrix = minHash(num)
    # reverse it for getting all doc for one hash function
    reverseSignature = np.array(signatureMatrix).T
    similarity = []
    for i in range(doc - 1):
        temp = []
        for j in range(i + 1, doc):
            # for doc1 & doc2, if they are same for one hash, the count plus 1
            k = np.sum(reverseSignature[i][:] == reverseSignature[j][:])  # same count
            temp.append(float(k) / float(num))  # similarity is equal to same count / total hash function numbers
        similarity.append(temp)
    end2 = time.time()
    print("Calculate similarity time for %d hash functions is %fs" % (num, end2 - start2))

    n = 0  # count the total compare times
    absSum = 0
    for i in range(len(similarity)):
        for j in range(len(similarity[i])):
            absSum += abs(actualJaccard[i + 1][j] - similarity[i][j])
            n += 1
    # because I only store half of the total matrix (only store e.g. (1,2) without (2,1))
    # thus, for MAE, I should only divide the compare times for this half part.
    MAE.append(float(absSum) / float(n))
plt.plot(range(10, 101, 10), MAE, 'g-s')
plt.xlabel('d')
plt.ylabel('MAE')
plt.title('The figure of MAEs with different values of d and MAE')
plt.show()
