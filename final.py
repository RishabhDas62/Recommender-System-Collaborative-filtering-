import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import itertools
from collections import Counter

data=fetch_movielens(min_rating=4.0)

train=data["train"]
test=data["test"]

def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)

def learning_curve(model, train, test,iterarray,**fit_params):
    old_epoch = 0
    train_p = []
    test_p = []
    headers = ['Epoch', 'train', 'test']
    print_log(headers, header=True)
    for epoch in iterarray:
        more = epoch - old_epoch
        model.fit_partial(train,epochs=more,**fit_params)
        train_auc = auc_score(model, data["train"], num_threads=2)
        test_auc = auc_score(model, data["test"], train_interactions=data["train"], num_threads=2)
        train_p.append(np.mean(train_auc))
        test_p.append(np.mean(test_auc))
        row = [epoch, train_p[-1], test_p[-1]]
        print_log(row)
    return model, train_p, test_p


model = LightFM(loss='warp', random_state=2016)
model.fit(train, epochs=0);

iterarray = range(10, 110, 10)

model, train_p, test_p = learning_curve(
    model, train, test, iterarray, **{'num_threads': 4}
)

def sample_recommendation(model,data,user_ids):
    n_users,n_items=data["train"].shape

    for user_id in user_ids:
        known_positives=data["item_labels"][data["train"].tocsr()[user_id].indices]

        scores=model.predict(user_id,np.arange(n_items))

        top_items=data["item_labels"][np.argsort(-scores)]
        print("User %s" % user_id)
        print("     Known Positives:")


        for x in known_positives[:3]:
            print("       %s" % x)

        print("     Recommendations:")

        for x in top_items[:3]:
            print("       %s" % x)

sns.set_style('white')

def plot_p(iterarray, p,
              title):
    plt.plot(iterarray, p);
    plt.title(title, fontsize=20);
    plt.xlabel('Epochs', fontsize=24);
    plt.xticks(fontsize=14);

ax = plt.subplot(1, 2, 1)
fig = ax.get_figure();
sns.despine(fig);
plot_p(iterarray, train_p,
         'Train')

ax = plt.subplot(1, 2, 2)
fig = ax.get_figure();
sns.despine(fig);
plot_p(iterarray, test_p,
         'Test')

plt.tight_layout();
