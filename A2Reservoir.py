import random
from collections import Counter

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import math

random.seed(123)
S = 10000

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


def reservoirSampling(n):
    # init
    S = np.zeros((n, 10000), dtype=int)
    count = np.zeros(n, dtype=int)
    totalWordID = range(1, word + 1)  # Avoid some wordID not being counted (not appears in this stream)
    for i in range(n):
        for m in range(length):
            if m < len(S[i]):  # store all the first s elements of the stream to S
                S[i][m] = stream[m][1]
            else:  # m element arrives
                probNum = random.randint(1, m)
                if probNum > len(S[i]):  # discard it
                    continue
                else:  # with prob. s/m
                    # Uniformly replace
                    replaceNum = random.randint(0, len(S[i]) - 1)
                    S[i][replaceNum] = stream[m][1]
                    count[i] += 1  # count update

        # assign a big space to store frequency
        dic = Counter(S[i])
        frequency = np.zeros([word, 2])  # init to save memory allocation time (compare than append())
        frequency[:, 0] = totalWordID
        for key in dic:
            frequency[key - 1, 1] = dic[key]
        # Sort frequency in descending (arrange their opposite number in ascending order)
        sortFreq = frequency[(-frequency[:, 1]).argsort()]

        # plot
        plt.plot(range(word), sortFreq[:, 1])
        plt.xlabel('Words in descending sort')
        plt.ylabel('Frequency')
        plt.title('The figure of descending sorted words by their frequencies')
        plt.show()
    return np.mean(count)


averageCount = reservoirSampling(5)
print("The average number of times the summary has been updated over these 5 runs is :", averageCount)

