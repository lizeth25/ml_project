"""
Lizeth Lucero
"""


import csv
import math
import random

# Order of CSV File: ECG  Apple Watch  Empatica  Garmin  Fitbit  Miband  Biovotion  ID  Skin Tone Activity

def load_Apple():
    examples = []
    ys = [] # label
    acts = {}

    # Load data into python from CSV file
    with open("deidentified_data.csv") as fp:
        rows = csv.reader(fp)
        header = next(rows) # I assume your datasets have a row with column-labels!
        for row in rows:

            entry = dict(zip(header, row)) # glue them into a dict
        
            # want to remove rows that have NaN for ECG or skin tone or if apple/fitbit heartbeat not given
            if (not row[0] == 'NaN') and (not row[8] == 'NaN') and (row[1] != 'NaN') and (row[9]!= 'NaN'):
                features = dict(zip(header, row)) # glue them into a dict
                #features.pop('Activity', None)  # given smaller amount of data, for now do not include
                features.pop('ID', None)
                features.pop('Miband', None)
                features.pop('Garmin', None)
                features.pop('Empatica', None)
                features.pop('Biovotion', None)
                features.pop('Fitbit', None)
                
                act = features.pop("Activity", None)
                features.update({"Rest": 0})
                features.update({"Breathe": 0})
                features.update({"Activity": 0})
                features.update({"Type": 0})
                features.update({act: 1})
                        
                for key in features.keys():
                    if features[key] == 'NaN':
                        features.update({key: float(0)})
                    elif key == "Rest" or key == "Breathe" or key == "Activity" or key == "Type":
                        features.update({key: int(features[key])})
                    else:
                        features.update({key: float(features[key])})
                ECG = features.pop('ECG', None)
                entry = {'features':features}
                entry.update({'ECG': ECG})

                keep = entry["features"]
                target = entry["ECG"]
                del entry["ECG"]
                ys.append(target)
                examples.append(keep)
    return examples, ys

def load_FitBit():
    examples = []
    ys = [] # label

    # Load data into python from CSV file
    with open("deidentified_data.csv") as fp:
        rows = csv.reader(fp)
        header = next(rows) # I assume your datasets have a row with column-labels!
        for row in rows:

            entry = dict(zip(header, row)) # glue them into a dict
        
            # want to remove rows that have NaN for ECG or skin tone or if apple/fitbit heartbeat not given
            if (not row[0] == 'NaN') and (not row[8] == 'NaN') and (row[4] != 'NaN') and (row[9]!= 'NaN'):
                features = dict(zip(header, row)) # glue them into a dict
                #features.pop('Activity', None)  # given smaller amount of data, for now do not include
                features.pop('ID', None)
                features.pop('Miband', None)
                features.pop('Garmin', None)
                features.pop('Empatica', None)
                features.pop('Biovotion', None)
                features.pop('Apple Watch', None)

                act = features.pop("Activity", None)
                features.update({"Rest": 0})
                features.update({"Breathe": 0})
                features.update({"Activity": 0})
                features.update({"Type": 0})
                features.update({act: 1})

                for key in features.keys():
                    if features[key] == 'NaN':
                        features.update({key: float(0)})
                    elif key == "Rest" or key == "Breathe" or key == "Activity" or key == "Type":
                        features.update({key: int(features[key])})
                    else:
                        features.update({key: float(features[key])})
                ECG = features.pop('ECG', None)
                entry = {'features':features}
                entry.update({'ECG': ECG})

                keep = entry["features"]
                target = entry["ECG"]
                del entry["ECG"]
                ys.append(target)
                examples.append(keep)
    return examples, ys     


examples, ys = load_FitBit()

from shared import simple_boxplot
from sklearn.utils import resample
import json
from sklearn.feature_extraction import DictVectorizer

## CONVERT TO MATRIX:
feature_numbering = DictVectorizer(sort=True)
X = feature_numbering.fit_transform(examples)


from sklearn.model_selection import train_test_split
import numpy as np

## SPLIT DATA:
RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
X_train, X_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

print(X_train.shape, X_vali.shape, X_test.shape)

from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    vali_acc: float
    params: Dict[str, Any]
    model: RegressorMixin

def consider_DecisionTree():
    performances : List[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["mse", "friedman_mse", "mae", "poisson"]:
            for d in range(1,14):
                params = {"criterion": crit, "max_depth":d, "random_state": rnd}
                f = DecisionTreeRegressor(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return min(performances, key=lambda result: result.vali_acc)

def consider_RandomTree():
    performances : List[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["mse", "mae"]:
            for d in range(1,14):
                params = {"criterion": crit, "max_depth":d, "random_state": rnd}
                f = RandomForestRegressor(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return min(performances, key=lambda result: result.vali_acc)

def consider_SGD():
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        for penal in ["l1","l2","elasticnet"]:
            for los in ["squared_loss","huber"]:
                params = {
                    "random_state": rnd,
                    "penalty": penal,
                    "max_iter": 100,
                    "loss":los
                }
                f = SGDRegressor(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return min(performances, key=lambda result: result.vali_acc)

dtree = consider_DecisionTree()
rforest = consider_RandomTree()
sgd = consider_SGD()


print("Best DTree",dtree)
print("Best RForest",rforest)
print("Best SGD", sgd)


################# Bootstrap Visualization #################
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
import matplotlib.pyplot as plt

# Changing to use mean absolute error instead
def bootstrap_mae(
    f: RegressorMixin,
    X,  # numpy array
    y,  # numpy array
    num_samples: int = 100,
    random_state: int = random.randint(0, 2 ** 32 - 1),
) -> List[float]:
    """
    Take the regressor ``f``, and compute it's bootstrapped mse over the dataset ``X``,``y``.
    Generate ``num_samples`` samples; and seed the resampler with ``random_state``.
    """
    dist: List[float] = []
    y_pred = f.predict(X)  # type:ignore (predict not on ClassifierMixin)
    # do the bootstrap:
    for trial in range(num_samples):
        sample_pred, sample_truth = resample(
            y_pred, y, random_state=trial + random_state
        )  # type:ignore
        score = mean_absolute_error(y_true=sample_truth, y_pred=sample_pred)  # type:ignore
        dist.append(score)
    return dist


boxplot_data: List[List[float]] = [
    bootstrap_mae(dtree.model, X_vali, y_vali), 
    bootstrap_mae(rforest.model,X_vali, y_vali),
    bootstrap_mae(sgd.model, X_vali, y_vali)

]
plt.boxplot(boxplot_data)
plt.xticks(ticks=[1,2], labels = ["DTree","RTree"])
plt.xlabel("Model")
plt.ylabel("MAE")
plt.ylim([0,1.5])
plt.show()





"""


PAST COMMENTS BELOW



"""

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