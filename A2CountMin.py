import random
import pandas as pd
import matplotlib.pyplot as plt
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

# εm= 100, I choose w = 2 / ε, this prob. is at most 1/2
w = int(2 / (100 / length))
# Count-Min Sketch choose d = log(1 /δ), have Pr [ F1 ≥ f1 + εm] ≤ δ
d = int(math.ceil(math.log(1 / 0.1)))

# The hash function elements
p = 1000003
an = np.array(random.sample(range(0, p), d))
bn = np.array(random.sample(range(0, p), d))
# initial the counter
counter = np.zeros((d, w), dtype=int)
totalHash = {}
for i in range(length):
    m = stream[i][1]
    hashValue = ((((m * an) + bn) % p) % w)
    totalHash[m] = hashValue
    for j in range(len(hashValue)):
        counter[j, hashValue[j]] += 1

totalWordID = range(1, word + 1)
frequency = np.zeros([word, 2])  # init to save memory allocation time (compare than append())
frequency[:, 0] = totalWordID
for i in totalWordID:
    temp = totalHash[i]
    frequency[i - 1, 1] = min(counter[j, temp[j]] for j in range(d))
# Sort frequency in descending (arrange their opposite number in ascending order)
sortFreq = frequency[(-frequency[:, 1]).argsort()]
# plot
plt.plot(range(word), sortFreq[:, 1])
plt.xlabel('Words in descending sort')
plt.ylabel('Frequency')
plt.title('The figure of descending sorted words by their frequencies')
plt.show()

# load brute force result
brute = pd.read_csv('sortFreq.txt', sep=' ',  header=None)
columns_name = ["wordID", "Frequency"]
brute.columns = columns_name
# judge if it's frequency is larger than 1000, return its index
indexList = np.where(brute.Frequency.values > 1000)
larger1000 = {}
# only store >1000 data and change to list
larger1000WordIDList = brute.wordID.values[indexList].tolist()
larger1000FrequencyList = brute.Frequency.values[indexList].tolist()
ans = []
for i in range(len(larger1000WordIDList)):
    index = larger1000WordIDList[i]
    # compare count-Min sketch result with data which it's larger than 1000 in brute force result
    if frequency[index - 1, 1] > 1000:
        ans.append(frequency[index - 1, 0])
print(ans)
