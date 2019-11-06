import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from math import sqrt


infile = uproot.open("test_tree.root")

tttt = infile["4top2016"]["testTree"]
tttj = infile["tttj"]["testTree"]
tttw = infile["tttw"]["testTree"]

tttt_frame = tttt.pandas.df(["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"])
tttj_frame = tttj.pandas.df(["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"])
tttw_frame = tttw.pandas.df(["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"])

train_tttt_frame = tttt_frame.truncate(after=1000)
train_tttj_frame = tttj_frame.truncate(after=1000)
train_tttw_frame = tttw_frame.truncate(after=1000)

train_tttt_frame["isSignal"] = 0
train_tttj_frame["isSignal"] = 1
train_tttw_frame["isSignal"] = 1

fulltrain = pandas.concat([train_tttt_frame, train_tttj_frame, train_tttw_frame])

target = fulltrain["isSignal"]
train = fulltrain.drop(["isSignal"], axis=1)

mod = xgb.XGBRegressor(
    gamma=1,                 
    learning_rate=0.01,
    max_depth=3,
    n_estimators=10000,                                                                    
    subsample=0.8,
    random_state=34
)

mod.fit(train, target)

pred_tttt = mod.predict(tttt_frame)
pred_tttj = mod.predict(tttj_frame)
pred_tttw = mod.predict(tttw_frame)

lumi = 150
tttt_scale = lumi*9.2/1022962.2
tttj_scale = lumi*0.474/100000
tttw_scale = lumi*0.788/97200

print("iterative test\n")
for i in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
    b = len(pred_tttt[pred_tttt > i])*tttt_scale
    s = len(pred_tttj[pred_tttj > i])*tttj_scale + len(pred_tttw[pred_tttw > i])*tttw_scale
    
    print("%s: s/sqrt(b) = %f     s/sqrt(b+s) = %f " % (i, s/sqrt(b), s/sqrt(s+b)))


xgb.plot_importance(mod)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

plt.hist(pred_tttt,color='r',alpha=0.5,label='4top',density=True)
plt.hist(pred_tttj ,color='g', alpha=0.5,label='3top+J', density=True)
plt.hist(pred_tttw ,color='b', alpha=0.5,label='3top+W', density=True)
plt.legend()
plt.show()


