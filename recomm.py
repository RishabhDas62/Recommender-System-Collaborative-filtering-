import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score


data=fetch_movielens(min_rating=4.0)

print(repr(data["train"]))
print(repr(data["test"]))

model=LightFM(loss="warp")
model1=LightFM(loss="bpr")
model2=LightFM(loss="warp-kos")
model3=LightFM(loss="logistic")
#diff bw desired output and models prediction
#ewighted approximate rank pairwise

model.fit(data["train"],epochs=60,num_threads=2)
model1.fit(data["train"],epochs=60,num_threads=2)
model2.fit(data["train"],epochs=60,num_threads=2)
model3.fit(data["train"],epochs=60,num_threads=2)


def sample_recommendation(model,data,user_ids):
    n_users,n_items=data["train"].shape

    for user_id in user_ids:
        known_positives=data["item_labels"][data["train"].tocsr()[user_id].indices]

        scores=model.predict(user_id,np.arange(n_items))

        top_items=data["item_labels"][np.argsort(-scores)]
        print("User %s" % user_id)
        print("     Known Positives:")


        for x in known_positives[:3]:
            print("     %s" % x)

        print("      Recommendations:")

        for x in top_items[:3]:
            print("       %s" % x)

sample_recommendation(model,data,[3,25,450])
#print("WARP")
#print("Train precision {}%".format(precision_at_k(model,data["train"],k=10).mean()))
#print("Test precision {}%".format(precision_at_k(model,data["test"],k=10).mean()))
#print("bpr")
#print("Train precision {}%".format(precision_at_k(model1,data["train"],k=10).mean()))
#print("Test precision {}%".format(precision_at_k(model1,data["test"],k=10).mean()))
#print("warp-kos")
#print("Train precision {}%".format(precision_at_k(model2,data["train"],k=10).mean()))
#print("Test precision {}%".format(precision_at_k(model2,data["test"],k=10).mean()))
#print("logistic")
#print("Train precision {}%".format(precision_at_k(model3,data["train"],k=10).mean()))
#print("Test precision {}%".format(precision_at_k(model3,data["test"],k=10).mean()))


train_auc = auc_score(model1, data["train"], num_threads=2).mean()
print('Collaborative filtering train AUC: %s' % train_auc)


test_auc = auc_score(model1, data["test"], train_interactions=data["train"], num_threads=2).mean()
print('Collaborative filtering test AUC: %s' % test_auc)       

        
