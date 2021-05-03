'''
Things to do : 
Round ECG
Play with Activity


Notes: 

The more I explore the data, the more I realize testing every single watch against
the ECG might be a lot more work so I might need to focus on only a couple of them
so that I can get to play around with features and the algorithm a lot more.

Moving forward I will only look at Apple Watch and Fitbit given they are
very popular amongst everyone. 
'''

import random	
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import typing as T
import re
import numpy as np
from dataclasses import dataclass

from shared import bootstrap_accuracy, bootstrap_auc, simple_boxplot

RAND = 123456
random.seed(RAND)

df: pd.DataFrame = pd.read_csv(
    "deidentified_data.csv"
)

def extract_features(row):
    """
    Given the ID and body of a Wikipedia article,
    extract features that might be of use to the 'is literary' task.

    Return named features in a dictionary.
    """

    new_features: T.Dict[str, T.Any] = {}
    skinTone = row["Skin Tone"]
    apple = row["Apple Watch"]

    new_features = {
        "st_3-": sum(1 for x in range(1) if skinTone <= 3),
        "st_4+": sum(1 for x in range(1) if skinTone > 4),
        "Skin Tone": row["Skin Tone"],
        "Apple Watch": row["Apple Watch"],
        # "Empatica": row["Empatica"],
        # "Garmin": row["Garmin"],
        # "Fitbit": row["Fitbit"],
        # "Miband": row["Miband"],
        # "Biovotion": row["Biovotion"],
    }
    
    return new_features
    



# right now each entry of the dataframe is a dictionary; json_normalize flattenst hat for us.
designed_f = pd.json_normalize(df.apply(extract_features, axis="columns"))


# Pandas lets us join really easily.
features: pd.DataFrame = designed_f.join(df["ECG"])
# Get names of indexes for which column ECG has value NaN
indexNames = features[ pd.isna(features["ECG"])].index
# Delete these row indexes from features
features.drop(indexNames , inplace=True)

indexNames = features[ pd.isna(features["Apple Watch"])].index
features.drop(indexNames , inplace=True)

features = features.fillna(0.0)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

# split the whole dataframe (including y-values)
tv_f, test_f = train_test_split(features, test_size=0.25, random_state=RAND)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RAND)

# feature numberer
numberer = DictVectorizer(sparse=False)
# feature scaling
scaling = StandardScaler()


def prepare_data(
    df: pd.DataFrame, fit: bool = False
) -> T.Tuple[np.ndarray, np.ndarray]:
    """This function converts a dataframe to an (X, y) tuple. It learns if fit=True."""
    global numeric, scaling
    y = df.pop("ECG").values
    # use fit_transform only on training data:
    if fit:
        return y, scaling.fit_transform(numberer.fit_transform(df.to_dict("records")))
    # use transform on vali & test:
    return y, scaling.transform(
        numberer.transform(df.to_dict("records"))
    )  # type:ignore


# use the 'prepare_data' function right above here:
train_y, train_X = prepare_data(train_f, fit=True)
vali_y, vali_X = prepare_data(vali_f)
test_y, test_X = prepare_data(test_f)


#%%
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Direct feature-importances (can think of them as how many times a feature was used):
rf = RandomForestRegressor(random_state=RAND, n_estimators=100)
rf.fit(train_X, train_y)

# loop over each tree and ask them how important each feature was!
importances = dict((name, []) for name in numberer.feature_names_)
for tree in rf.estimators_:
    for name, weight in zip(numberer.feature_names_, tree.feature_importances_):
        importances[name].append(weight)

# Think: what does 'how many splits' actually measure? Usefulness, or something else?
simple_boxplot(
    importances,
    title="Tree Importances",
    ylabel="Decision Tree Criterion Importances",
    save="graphs/DecisionTree-importances.png",
)


#%%

# graphs: T.Dict[str, T.List[float]] = {}


# @dataclass
# class Model:
#     vali_score: float
#     m: T.Any


# def train_and_eval(name, x, y, vx, vy):
#     """Train and Eval a single model."""
#     options: T.List[Model] = []
#     for i in range(5):
#         m = SGDRegressor(random_state=RAND + i)
#         m.fit(x, y)
#         options.append(Model(m.score(vx, vy), m))

#     for d in range(3, 15):
#         m = DecisionTreeRegressor(
#             random_state=RAND
#         )
#         m.fit(x, y)
#         options.append(Model(m.score(vx, vy), m))

#     # pick the best model:
#     best = max(options, key=lambda m: m.vali_score)
#     # bootstrap its output:
#     graphs[name] = bootstrap_accuracy(best.m, vx, vy)
#     # record our progress:
#     print("{:20}\t{:.3}\t{}".format(name, np.mean(graphs[name]), best.m))

# train_and_eval("Full Model", train_X, train_y, vali_X, vali_y)

# for fid, fname in enumerate(numberer.feature_names_):
#     # one-by-one, delete your features:
#     without_X = train_X.copy()
#     without_X[:, fid] = 0.0
#     # score a model without the feature to see if it __really__ helps or not:
#     train_and_eval("without {}".format(fname), without_X, train_y, vali_X, vali_y)

# # Inline boxplot code here so we can sort by value:
# box_names = []
# box_dists = []
# for (k, v) in sorted(graphs.items(), key=lambda tup: np.mean(tup[1])):
#     box_names.append(k)
#     box_dists.append(v)

# # Matplotlib stuff:
# plt.boxplot(box_dists)
# plt.xticks(
#     rotation=30,
#     horizontalalignment="right",
#     ticks=range(1, len(box_names) + 1),
#     labels=box_names,
# )
# plt.title("Feature Removal Analysis")
# plt.xlabel("Included?")
# plt.ylabel("AUC")
# plt.tight_layout()
# plt.savefig("graphs/feature-removal.png")
# plt.show()






