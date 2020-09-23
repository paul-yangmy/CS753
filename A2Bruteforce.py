import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
# assign a big space to store frequency
dic = Counter(df.wordID.values)
totalWordID = range(1, word + 1)  # Avoid some wordID not being counted (not appears in this stream)
frequency = np.zeros([word, 2])  # init to save memory allocation time (compare than append())
frequency[:, 0] = totalWordID  # first column stores wordID, second column stores its frequency which initialized to 0
for key in dic:
    frequency[key - 1, 1] = dic[key]
# Average frequency
averageFreq = np.mean(frequency[:, 1])
print("The average frequency of the words in stream is :", averageFreq)

# Sort frequency in descending (arrange their opposite number in ascending order)
sortFreq = frequency[(-frequency[:, 1]).argsort()]

# Save the result into file
with open("sortFreq.txt", "w+") as f:
    for i in range(len(sortFreq)):
        f.write("%-i %-i\n" % (sortFreq[i, 0], sortFreq[i, 1]))
f.close()

# plot
plt.plot(range(word), sortFreq[:, 1])
plt.xlabel('Words in descending sort')
plt.ylabel('Frequency')
plt.title('The figure of descending sorted words by their frequencies')
plt.show()
