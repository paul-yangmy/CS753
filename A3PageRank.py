from collections import Counter
import numpy as np
import pandas as pd
import time
from scipy import sparse
from operator import itemgetter

# read the third row to get information of the txt
summary = pd.read_table('web-Google.txt', sep=' ', skiprows=2, nrows=1, header=None)
[[nodes, edges]] = summary[[2, 4]].values
# import data
df = pd.read_table("web-Google.txt", sep='\t', skiprows=4, header=None)
columns_name = ["FromNodeId", "ToNodeId"]
df.columns = columns_name  # set col names to help use this dataframe
df = df.sort_values(axis=0, ascending=True, by="FromNodeId")  # sort
# dictionary = {}
# for i in set(df.FromNodeId):
#     dictionary[i] = df[df.FromNodeId == i].ToNodeId.values.tolist()

fromNodeId = df.FromNodeId.tolist()
toNodeId = df.ToNodeId.tolist()
d_value = []
for key, value in Counter(fromNodeId).items():
    d = 1 / value
    dForKey = np.repeat(d, value).tolist()
    d_value.extend(dForKey)
# the information in txt isn't the real size. Like 4th row's data (0,891835)
# which is quite larger than the information in txt(Nodes: 875713)
nodeList = list(set(set(fromNodeId).union(set(toNodeId))))

# Build mapping to rename nodes
map = dict()
mapreverse = dict()
index = 0
for node in nodeList:
    map[node] = index  # map real to index
    mapreverse[index] = node  # map real to index
    index += 1

# def real2Index(realList):
#     indexList = []
#     for i in realList:
#         indexList.append(nodeList.index(i))
#     return indexList

# For each node i, if we have link i -> j
# then Mji = 1 / di  otherwise Mji = 0 (the adjacency matrix is Mji not Mij)
# so choose csc_matrix which can help to find if each column sums up to 1
matrix = sparse.csc_matrix((d_value, (np.array(itemgetter(*toNodeId)(map)),
                                      np.array(itemgetter(*fromNodeId)(map)))), shape=(nodes, nodes))

# initialize r0 = [1/N, 1/N, ... , 1/N]‘s transpose
vec = np.full(nodes, 1 / nodes)


# def remove_zero(X):
#     # X is a scipy sparse matrix. We want to remove all zero rows from it
#     nonzero_row_indice, nonzero_col_indice = X.nonzero()
#     unique_nonzero_row_indice = np.unique(nonzero_row_indice)
#     X = X[unique_nonzero_row_indice]
#     unique_nonzero_col_indice = np.unique(nonzero_col_indice)
#     X = X[:, unique_nonzero_col_indice]
#     return X


# Q1:
def PageRank(mat, vector):
    num = 1
    while True:
        vecNew = mat * vector
        if sum(abs(vecNew - vector)) < 0.02:  # smaller than threshold
            break
        vector = vecNew
        num += 1
    return vecNew, num


start1 = time.time()
rankVec1, stopNum1 = PageRank(matrix, vec)
end1 = time.time()
print('The running time of PageRank in Equation 1 is %fs' % (end1 - start1))
print("The number of iterations needed to stop is", stopNum1)
print("")

# map index to real nodes
result1 = np.c_[np.array(itemgetter(*range(len(rankVec1)))(mapreverse)), rankVec1]
result1 = result1[np.argsort(result1[:, 1])[::-1]]
print("The IDs and scores of the top-10 ranked nodes are:")
for i in range(10):
    print("NodeID:%-6i, PageRank:%f" % (result1[i, 0], result1[i, 1]))
print("")


# Q2: handle spider traps
def teleportPageRank(mat, vector, beta, n):
    # beta = 0.9
    addConst = (1 - beta) / n
    num = 1
    while True:
        vecNew = beta * (mat * vector) + addConst
        if sum(abs(vecNew - vector)) < 0.02:
            break
        vector = vecNew
        num += 1
    return vecNew, num


start2 = time.time()
beta = np.linspace(1, 0.5, num=6)
rankVec2, stopNum2 = teleportPageRank(matrix, vec, beta[1], nodes)
end2 = time.time()
print('The running time of PageRank in Equation 2 is %fs' % (end2 - start2))
print("The number of iterations needed to stop is", stopNum2)
print("")

# map index to real nodes
result2 = np.c_[np.array(itemgetter(*range(len(rankVec2)))(mapreverse)), rankVec2]
result2 = result2[np.argsort(result2[:, 1])[::-1]]
print("The IDs and scores of the top-10 ranked nodes are:")
for i in range(10):
    print("NodeID:%-6i, PageRank:%f" % (result2[i, 0], result2[i, 1]))
print("")

for b in beta:
    rankVec2c, stopNum2c = teleportPageRank(matrix, vec, b, nodes)
    print("The number of iterations needed to stop is %d for beta = %.1f" % (stopNum2c, b))
    # Q3: Exploit dead-ends
    leakedScore = 1 - sum(rankVec2c)
    print("The leaked PageRank score is %f for beta = %.1f" % (leakedScore, b))


def iterPageRank(mat, vector, n):
    beta = 0.9
    addConst = (1 - beta) / n
    num = 1
    while True:
        vecNew = beta * (mat * vector) + addConst
        leakedScore = 1 - sum(vecNew)
        print("For iteration %d, the leaked PageRank score is %f" % (num, leakedScore))
        if sum(abs(vecNew - vector)) < 0.02:
            break
        vector = vecNew
        num += 1
    return vecNew, num


print("")
print("With beta = 0.9")
rankVec3, stopNum3 = iterPageRank(matrix, vec, nodes)


# Q4: support dead-ends
# To address the dead-end problem
# we replace the all-zero columns in the original transition matrix M with [1/N] N*1.
def redistributePageRank(mat, vector, n):
    beta = 0.9
    num = 1
    while True:
        vecNew = beta * (mat * vector)
        leakedScore = 1 - sum(vecNew)  # add L back to each iterations
        vecNew += leakedScore / n
        if sum(abs(vecNew - vector)) < 0.02:
            break
        vector = vecNew
        num += 1
    return vecNew, num


start4 = time.time()
rankVec4, stopNum4 = redistributePageRank(matrix, vec, nodes)
end4 = time.time()
print("")
print('The running time of redistributing the leaked PageRank: is %fs' % (end4 - start4))
print("The number of iterations needed to stop is", stopNum4)
print("")

# map index to real nodes
result4 = np.c_[np.array(itemgetter(*range(len(rankVec4)))(mapreverse)), rankVec4]
result4 = result4[np.argsort(result4[:, 1])[::-1]]
print("The IDs and scores of the top-10 ranked nodes are:")
for i in range(10):
    print("NodeID:%-6i, PageRank:%f" % (result4[i, 0], result4[i, 1]))
print("")
# justify by adding L back to the pagerank equation
print(sum(rankVec4))
