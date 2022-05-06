import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def distance(a,b):
    summ = 0
    for i in range(len(a)):
        summ += (a[i]-b[i])**2
    return np.sqrt(summ)


def updateCenter(data, assignments, K):
    newCenters = []
    for cluster in range(K):
        newCenter = np.array([0.0]*9)
        numberOfCluster = 0
        for i in range(len(data)):
            if assignments[i] == cluster:
                numberOfCluster += 1
                newCenter += data[i]
        newCenters.append(newCenter/numberOfCluster)
    return np.array(newCenters)


def updateAssignment(data, centers):
    assignments = []
    for i in range(len(data)):
        distances = []
        for center in centers:
            distances.append(distance(data[i], center))
        assignments.append(distances.index(min(distances)))
    return np.array(assignments)


def checkAssignment(assignments, K):
    missing = []
    for i in range(K):
        if i not in assignments:
            missing.append(i)
    for missingvalue in missing:
        replacement = np.random.choice(len(assignments), int(len(assignments)/10))
        for j in replacement:
            assignments[j] = missingvalue
    return assignments


def loss(centers, assignments, data):
    lossSum = 0
    for i in range(len(data)):
        sample = data[i]
        lossSum += (LA.norm(sample-centers[assignments[i]]) ** 2)
    return lossSum


data = []
iterations = 200

with open('breast-cancer-wisconsin.data', 'r', encoding='utf-8') as fin:
    for i in fin:
        j = i.strip().split(',')
        try:
            j = [float(x) for x in j]
        except:
            continue
        data.append(j[1:-1])

data = np.array(data)
potential_lst = list()

for K in range(2, 9):
    # set random seed
    random.seed(42)
    np.random.seed(42)

    # assignments = np.random.randint(0, K, data.shape[0])
    center_idx = np.random.choice(data.shape[0], size=K, replace=False)
    centers = data[center_idx, :]
    assignments = updateAssignment(data, centers)

    for iteration in range(iterations):
        new_centers = updateCenter(data, assignments, K)
        new_assignments = updateAssignment(data, new_centers)

        if (new_assignments == assignments).all():
            potential = loss(new_centers, new_assignments, data)
            potential_lst.append(potential)
            break
        else:
            centers = new_centers
            assignments = checkAssignment(new_assignments, K)


# plot
plt.figure()
plt.plot(list(range(2, 9)), potential_lst)
plt.xlabel('Number of Clusters k')
plt.ylabel('SSE Values')
plt.title('SSE Value vs. Number of Clusters k')
plt.savefig('kmeans.png')
