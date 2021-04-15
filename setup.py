"""
Lizeth Lucero
"""


import csv
import math

examples = []
ys = [] # label

# Load data into python from CSV file
with open("deidentified_data.csv") as fp:
    rows = csv.reader(fp)
    header = next(rows) # I assume your datasets have a row with column-labels!
    for row in rows:
        # print(header) # print the labels
        # print(row) # print the current row
        # entry = dict(zip(header, row)) # glue them into a dict
        # print(entry) # print that dict
    
        # want to remove rows that have NaN for ECG or skin tone
        if (not row[0] == 'NaN') and (not row[8] == 'NaN'):
            features = dict(zip(header, row)) # glue them into a dict
            features.pop('Activity', None)
            features.pop('ID', None)
            for key in features.keys():
                if features[key] == 'NaN':
                    features.update({key: float(0)})
                else:
                    features.update({key: float(features[key])})
            ECG = features.pop('ECG', None)
            entry = {'features':features}
            entry.update({'ECG': ECG})

            
            keep = entry["features"]
            ys.append(entry["ECG"])
            examples.append(keep)
            
            # print("entry:", entry) # entry
            # break # stop after 1 row of data, so you can inspect, choose which columns matter, etc.

'''
I think that my task definition works well for this project
I think that I need to think more about the training error that I will accept. 
Because this project examines heartbeats, I think that I want to do 
some more research with error rates that I can accept. 


Cleaning up data: 
I decided to remove any data points which had a NaN for ECG because
I am using those as the 'true' result. Additionally, I will be removing any data that has
NaN for skin color because I am hoping to use that for my algorithm. 


'''

'''
Training Set Check In

I will need a non-linear model. First I reformatted all the NaN's in my data to equal 0
I also looked into non-numerical data and removed activity and ID for now but I might
put back activity once I transform it into numerical(ordinal) data.


Regarding the training of the data, I think I may not have enough data based on the graphs, 
but I need to check more into this especially since I hope to add activity.

'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from shared import simple_boxplot
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import json
from sklearn.tree import DecisionTreeRegressor

#%% load up the data
# examples = []
# ys = [] # label

# Load our data to list of examples:
# with open(dataset_local_path("poetry_id.jsonl")) as fp:
#     for line in fp:
#         info = json.loads(line)
#         keep = info["features"]
#         ys.append(info["poetry"])
#         examples.append(keep)

## CONVERT TO MATRIX:
feature_numbering = DictVectorizer(sort=True, sparse=False)
X = feature_numbering.fit_transform(examples)
del examples

## SPLIT DATA:
RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
rX_tv, rX_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
rX_train, rX_vali, y_train, y_vali = train_test_split(
    rX_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)


scale = StandardScaler()
X_train = scale.fit_transform(rX_train)
X_vali: np.ndarray = scale.transform(rX_vali)  # type:ignore
X_test: np.ndarray = scale.transform(rX_test)  # type:ignore

#%% Actually compute performance for each % of training data
N = len(y_train)
num_trials = 100
percentages = list(range(5, 100, 5))
percentages.append(100)
scores = {}
acc_mean = []
acc_std = []


# Which subset of data will potentially really matter.
for train_percent in percentages:
    n_samples = int((train_percent / 100) * N)
    print("{}% == {} samples...".format(train_percent, n_samples))
    label = "{}".format(train_percent, n_samples)

    # So we consider num_trials=100 subsamples, and train a model on each.
    scores[label] = []
    for i in range(num_trials):
        X_sample, y_sample = resample(
            X_train, y_train, n_samples=n_samples, replace=False
        )  # type:ignore
        # Note here, I'm using a simple classifier for speed, rather than the best.
        clf = DecisionTreeRegressor(random_state=RANDOM_SEED + train_percent + i)
        clf.fit(X_sample, y_sample)
        # so we get 100 scores per percentage-point.
        scores[label].append(clf.score(X_vali, y_vali))
    # We'll first look at a line-plot of the mean:
    acc_mean.append(np.mean(scores[label]))
    acc_std.append(np.std(scores[label]))

# First, try a line plot, with shaded variance regions:
import matplotlib.pyplot as plt

means = np.array(acc_mean)
std = np.array(acc_std)
plt.plot(percentages, acc_mean, "o-")
plt.fill_between(percentages, means - std, means + std, alpha=0.2)
plt.xlabel("Percent Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, 100])
plt.title("Shaded Accuracy Plot")
plt.savefig("graphs/p09-area-Accuracy.png")
plt.show()


# Second look at the boxplots in-order: (I like this better, IMO)
simple_boxplot(
    scores,
    "Learning Curve",
    xlabel="Percent Training Data",
    ylabel="Accuracy",
    save="graphs/p09-boxplots-Accuracy.png",
)