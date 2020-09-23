import numpy as np
import pandas as pd
import time

# read first three rows to get information of the txt
summary = pd.read_csv('docword.kos.txt', sep=' ', nrows=3, header=None)
[doc, word, length] = summary[0]  # set as doc, word and length
# skip first three rows to load data
df = pd.read_csv('docword.kos.txt', sep=' ', skiprows=[0, 1, 2], header=None)
columns_name = ["docID", "wordID", "Frequency"]
df.columns = columns_name  # set col names to help use this dataframe
df.Frequency = 1  # count = 1 for each pair
# shingles = np.zeros([word, doc], dtype=int)  # init 0
# for i in range(length):
#     shingles[df.wordID[i]-1, df.docID[i]-1] = 1  # if exist, change to 1
dictionary = {}
for i in set(df.docID):
    # for all docID, select all wordID and change to list
    dictionary[i] = df[df.docID == i].wordID.values.tolist()
res = []
start = time.time()
for i in range(doc - 1):
    temp = []
    for j in range(i + 1, doc):  # avoid double counting, e.g. (1,2) & (2,1)
        # using set to calculate union and intersection
        docSet1 = set(dictionary[i + 1])
        docSet2 = set(dictionary[j + 1])
        # bit operation
        union = len(docSet1 | docSet2)
        intersection = len(docSet1 & docSet2)
        temp.append(float(intersection) / float(union))
    res.append(temp)
end = time.time()
print('Running time %fs' % (end - start))

n = 0
totalSum = 0
average = []
with open("similarity.txt", "w+") as f:
    for i in range(len(res)):
        for j in range(len(res[i])):
            f.write("%-i %-i %-f\r\n" % (i+1, i+j+2, res[i][j]))
            totalSum += res[i][j]  # total similarity
            n += 1  # count
f.close()
print("The average Jaccard similarity of all pairs except identical pairs is :", totalSum / n)
