import random
from collections import Counter
import pandas as pd
import time
import numpy as np
import math

random.seed(123)

# read first three rows to get information of the txt
summary = pd.read_csv('docword.kos.txt', sep=' ', nrows=3, header=None)
[doc, word, length] = summary[0]  # set as doc, word and length
# import data
df = pd.read_csv('docword.kos.txt', sep=' ', skiprows=[0, 1, 2], header=None)
columns_name = ["docID", "wordID", "Frequency"]
df.columns = columns_name  # set col names to help use this dataframe
df.Frequency = 1  # we do not use the information of count

# Get the stream tuple
stream = np.vstack((df.docID, df.wordID, df.Frequency)).T.tolist()

# init
S = np.zeros(10000, dtype=int)
totalWordID = range(1, word + 1)  # Avoid some wordID not being counted (not appears in this stream)
for m in range(length):
    if m < len(S):  # store all the first s elements of the stream to S
        S[m] = stream[m][1]
    else:  # m element arrives
        probNum = random.randint(1, m)
        if probNum > len(S):  # discard it
            continue
        else:  # with prob. s/m
            # Uniformly replace
            replaceNum = random.randint(0, len(S) - 1)
            S[replaceNum] = stream[m][1]

# assign a big space to store frequency
dic = Counter(S)
frequency = np.zeros([word, 2])  # init to save memory allocation time (compare than append())
frequency[:, 0] = totalWordID
for key in dic:
    frequency[key - 1, 1] = dic[key]
# Sort frequency in descending (arrange their opposite number in ascending order)
sortFreq = frequency[(-frequency[:, 1]).argsort()]
# Theorem: If k = Rounded up of 1 / frequent item, all frequent items will be detected by the Misra-Gries algorithm
# to find the most frequent words whose frequency is larger than 1,000.
w = 1000
k = int(math.ceil(1 / (w / length)))
counter = dict()
decreaseNum = 0

for m in range(length):
    a = stream[m][1]
    if a in counter.keys():  # already have a counter for a
        counter[a] += 1
    elif len(counter) <= k:  # no counter for a and fewer k counter
        counter[a] = 1
    else:
        for key in list(counter.keys()):
            counter[key] -= 1
            decreaseNum += 1
            if counter[key] == 0:
                counter.pop(key)

temp = []
for i in counter:
    if counter[i] >= 1000:
        temp.append(i)
print("the most frequent words whose frequency is larger than 1000 is:", temp)
print("the number of decrement steps with k = %d is %d" % (k, decreaseNum))
