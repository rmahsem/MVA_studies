import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from math import sqrt
from sklearn.model_selection import train_test_split

TTTT=0
TTTJ=1
TTTW=2

samp2Sig = {TTTT:0, TTTJ:1, TTTW:1}

infile = uproot.open("../VVAnalysis/test_tree.root")

tttt = infile["4top2016"]["testTree"]
tttj = infile["tttj"]["testTree"]
tttw = infile["tttw"]["testTree"]


vars = [ "HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"]
tttt_frame = tttt.pandas.df(vars)
tttj_frame = tttj.pandas.df(vars)
tttw_frame = tttw.pandas.df(vars)
tttt_frame["type"] = TTTT
tttj_frame["type"] = TTTJ
tttw_frame["type"] = TTTW
tttt_frame["isSignal"] = 0
tttj_frame["isSignal"] = 1
tttw_frame["isSignal"] = 1

all_frame = pandas.concat([tttt_frame, tttj_frame, tttw_frame], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(all_frame.drop(["isSignal"], axis=1), all_frame["isSignal"], test_size=0.30, random_state=42)
X_train = X_train.drop(["type"], axis=1)

all_test = pandas.concat([X_test, y_test], axis=1)
split_test = [all_test[all_test.type==i].drop(["type", "isSignal"], axis=1) for i in range(3)]


mod = xgb.XGBRegressor(
    gamma=1,                 
    learning_rate=0.01,
    max_depth=3,
    n_estimators=10000,                                                                    
    subsample=0.8,
    random_state=34
)

mod.fit(X_train, y_train)

split_pred = [mod.predict(frame) for frame in split_test]
split_truth = [np.empty(len(arr)) for arr in split_pred]
for i in range(3):
    split_truth[i].fill(samp2Sig[i])


lumi = 35.9
tttt_scale = lumi*9.2/1022962.2
tttj_scale = lumi*0.474/100000
tttw_scale = lumi*0.788/97200



print("iterative test\n")
for x in range(0, 11):
    i = x*0.1
    b = len(split_pred[TTTT][split_pred[TTTT] > i])*tttt_scale
    s = len(split_pred[TTTJ][split_pred[TTTJ] > i])*tttj_scale \
        + len(split_pred[TTTW][split_pred[TTTW] > i])*tttw_scale
    if b <= 0:
        break

    print("%s: s/sqrt(b) = %f     s/sqrt(b+s) = %f " % (i, s/sqrt(b), s/sqrt(s+b)))


xgb.plot_importance(mod)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

plt.hist(split_pred[TTTT],color='r',alpha=0.5,label='4top',density=True)
plt.hist(split_pred[TTTJ],color='g', alpha=0.5,label='3top+J', density=True)
plt.hist(split_pred[TTTW] ,color='b', alpha=0.5,label='3top+W', density=True)
plt.legend()
plt.show()



from sklearn.metrics import roc_auc_score, roc_curve


pred_all = np.concatenate(split_pred)
truth_all = np.concatenate(split_truth)


fpr, tpr,_ = roc_curve(truth_all, pred_all)
auc_test = roc_auc_score(truth_all, pred_all)
plt.plot(fpr,tpr, label="test AUC = {:.3f}".format(auc_test))
plt.legend()
plt.show()
