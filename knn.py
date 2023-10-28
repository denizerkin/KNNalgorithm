import numpy as np

class KNN:

  def __init__(self, k=3):
    self.k = k

  def euclidean_distance(self, x, y):
    distance = np.sqrt(np.sum(x-y)**2)
    return distance

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    predictions = []
    for test in X:
      temp = self.__predict(test)
      predictions.append(temp)
    return predictions

  def __predict(self, x):
    distances = []
    for train in self.X_train:
      distance = self.euclidean_distance(x,train)
      distances.append(distance)
    min_distance_indices = sorted(range(len(distances)), key=lambda index: distances[index])[:self.k]
    print(min_distance_indices)
    min_distance_labels = []
    for i in min_distance_indices:
      min_distance_labels.append(self.y_train[i])

    most_common = {}
    for i in min_distance_labels:
      if i not in most_common.keys():
        most_common[i] = 1
      else:
        most_common[i] += 1
    return max(most_common, key = lambda x:most_common[x])

