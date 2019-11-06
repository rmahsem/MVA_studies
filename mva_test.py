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
infile2 = uproot.open("../VVAnalysis/test_tree_2.root")
#["HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"]

def getFrame(infile, dirName, typeName, signal):
    bNames = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"]
    tree = infile[dirName]["testTree"]
    frame = tree.pandas.df(bNames)
    frame["type"] = typeName
    frame["isSignal"] = signal
    return frame

tttt_frame = getFrame(infile, "4top2016", "TTTT", 0)
tttj_frame = getFrame(infile, "tttj", "TTTJ", 1)
tttw_frame = getFrame(infile, "tttw", "TTTW", 1)
dy_frame = getFrame(infile2, "DYm50", "DY", 0)

frameList = [tttt_frame, tttj_frame, tttw_frame]


number = 1000
split_train = [tttt_frame.truncate(after=2*number-1).drop(["type"], axis=1), \
               tttj_frame.truncate(after=number-1).drop(["type"], axis=1), \
               tttw_frame.truncate(after=number-1).drop(["type"], axis=1)]
split_test = [tttt_frame.truncate(before=number).drop(["isSignal"], axis=1), \
              tttj_frame.truncate(before=number).drop(["isSignal"], axis=1), \
              tttw_frame.truncate(before=number).drop(["isSignal"], axis=1)]

# [tttt_frame.drop(["isSignal"], axis=1), \
#               tttj_frame.drop(["isSignal"], axis=1), \
#               tttw_frame.drop(["isSignal"], axis=1)]


X_train = pandas.concat(split_train,ignore_index=True).drop(["isSignal"], axis=1)
y_train = pandas.concat(split_train, ignore_index=True)["isSignal"]

 # all_frame = pandas.concat(frameList).reset_index(drop=True)
# X_train, X_test, y_train, y_test = train_test_split(all_frame.drop(["isSignal"], axis=1), all_frame["isSignal"], test_size=0.30, random_state=42)


# split_train = [X_train[X_train.type=="TTTT"].drop(["type"], axis=1),
#               X_train[X_train.type=="TTTJ"].drop(["type"], axis=1)]
# X_train = X_train.drop(["type"], axis=1)

# split_test = [X_test[X_test.type=="TTTT"],
#               X_test[X_test.type=="TTTJ"]]


mod = xgb.XGBRegressor()

mod.fit(X_train, y_train)




test_pred = [mod.predict(frame.drop("type", axis=1)) for frame in split_test]
train_pred = [mod.predict(frame.drop(["isSignal"], axis=1)) for frame in split_train]

lumi = 140
tttt_scale = lumi*9.2/1022962.2*len(tttt_frame)/len(test_pred[0])
tttj_scale = lumi*0.474/100000*len(tttj_frame)/len(test_pred[1])
tttw_scale = lumi*0.788/97200*len(tttw_frame)/len(test_pred[2])



print("iterative test\n")
for x in range(0, 11):
    i = x*0.1
    b = len(test_pred[0][test_pred[0] > i])*tttt_scale
    s = len(test_pred[1][test_pred[1] > i])*tttj_scale + len(test_pred[2][test_pred[2] > i])*tttw_scale
    if b <= 0:
        break

    print("%s: s: %f   b: %f   s/sqrt(b) = %f     s/sqrt(b+s) = %f " % (i, s, b, s/sqrt(b), s/sqrt(s+b)))


plt.hist(test_pred[0],color='r',alpha=0.5,label='4top',density=True)
plt.hist(test_pred[1] ,color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2] ,color='g', alpha=0.5,label='3top+W', density=True)
plt.legend()
plt.show()

plt.hist(train_pred[0],color='r',alpha=0.5,label='4top',density=True)
plt.hist(train_pred[1] ,color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2] ,color='g', alpha=0.5,label='3top+W', density=True)
plt.legend()
plt.show()


exit(0)

    
split_truth = [np.empty(len(arr)) for arr in test_pred]
for i in range(len):
    split_truth[i].fill(samp2Sig[i])

xgb.plot_importance(mod)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()





from sklearn.metrics import roc_auc_score, roc_curve


pred_all = np.concatenate(test_pred)
truth_all = np.concatenate(split_truth)


fpr, tpr,_ = roc_curve(truth_all, pred_all)
auc_test = roc_auc_score(truth_all, pred_all)
plt.plot(fpr,tpr, label="test AUC = {:.3f}".format(auc_test))
plt.legend()
plt.show()
