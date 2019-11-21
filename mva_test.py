import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


infile4top = uproot.open("../VVAnalysis/test_tree_1108_tttt.root")
infile3topW = uproot.open("../VVAnalysis/test_tree_1108_tttw.root")
infile3topJ = uproot.open("../VVAnalysis/test_tree_1108_tttj.root")
infileAll = uproot.open("../VVAnalysis/full_MC_tree.root")
# infile2 = uproot.open("../VVAnalysis/test_tree_2.root")
#["HT", "MET", "l1Pt", "l2Pt", "lepMass", "sphericity", "centrality", "j1Pt", "b1Pt"]
bNames = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", \
          "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", \
          "b1Pt", "b2Pt", "b3Pt", "b4Pt", "DilepCharge",\
          "weight",]
# "Shape1", "Shape2",
def getFrame(infile, dirName, signal):
    global bNames
    dir = infile[dirName]
    i=1
    valid = list()
    while("testTree;%i"%i in dir):
        valid.append(dir["testTree;%i"%i])
        i += 1
    frames = [tree.pandas.df(bNames) for tree in valid]
    frame = pandas.concat(frames, ignore_index=True)
    frame["isSignal"] = signal
    return frame

tttt_frame = getFrame(infile4top, "tttt", 0)
tttj_frame = getFrame(infile3topJ, "tttj", 1)
tttw_frame = getFrame(infile3topW, "tttw", 1)

ttw_frame = getFrame(infileAll, "ttw", 0)
ttz_frame = getFrame(infileAll, "ttz", 0)
tth_frame = getFrame(infileAll, "tth2nonbb", 0)
ttv = [ttw_frame, ttz_frame, tth_frame]

ttwh_frame = getFrame(infileAll, "ttwh", 0)
ttwz_frame = getFrame(infileAll, "ttwz", 0)
ttww_frame = getFrame(infileAll, "ttww", 0)
tthh_frame = getFrame(infileAll, "tthh", 0)
ttzh_frame = getFrame(infileAll, "ttzh", 0)
ttzz_frame = getFrame(infileAll, "ttzz", 0)
ttvv = [ttwh_frame, ttwz_frame, ttww_frame, tthh_frame, ttzh_frame, ttzz_frame]

www_frame = getFrame(infileAll, "www", 0)
wwz_frame = getFrame(infileAll, "wwz", 0)
wzz_frame = getFrame(infileAll, "wzz", 0)
zzz_frame = getFrame(infileAll, "zzz", 0)
vvv = [www_frame, wwz_frame, wzz_frame, zzz_frame]

zz_frame = getFrame(infileAll, "zz4l_powheg", 0)
wz_frame = getFrame(infileAll, "wz3lnu_mg5amcnlo", 0)
ww_double_frame = getFrame(infileAll, "ww_doubleScatter", 0)
wpwpjj_frame = getFrame(infileAll, "wpwpjj_ewk", 0)
vv = [zz_frame, wz_frame, ww_double_frame, wpwpjj_frame]

print len(tttw_frame)
print len(tttj_frame)
print len(tttt_frame)
print len(ttw_frame), len(ttz_frame)


frameList = [tttt_frame, tttj_frame, tttw_frame, ttw_frame, ttz_frame]
frameFact = [ 4 , 2, 2,    2,   4] 

number = 3000
split_train = [frame.truncate(after=fact*number-1) for frame, fact in zip(frameList, frameFact)]
split_test = [frame.truncate(before=fact*number) for frame, fact in zip(frameList, frameFact)]


X_train = pandas.concat(split_train,ignore_index=True).drop(["isSignal", "weight"], axis=1)
w_train = pandas.concat(split_train,ignore_index=True)["weight"]
y_train = pandas.concat(split_train, ignore_index=True)["isSignal"]

mod = xgb.XGBRegressor(n_estimators=500, \
                       eta = 0.07,\
                       max_depth = 5, \
                       subsample = 0.6, \
                       alpha = 8.0, \
                       gamma = 2.0, \
                       )

fitModel = mod.fit(X_train, y_train, sample_weight=w_train)


test_pred = [mod.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in split_test]
train_pred = [mod.predict(frame.drop(["isSignal", "weight"], axis=1)) for frame in split_train]

ttv_pred = [mod.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in ttv]
ttvv_pred = [mod.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in ttvv]
vvv_pred = [mod.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in vvv]
vv_pred = [mod.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in vv]


lumi = 140
tttt_scale = lumi*9.2/2496900*1/3*len(tttt_frame)/len(test_pred[0])
tttj_scale = lumi*0.474/484000*len(tttj_frame)/len(test_pred[1])
tttw_scale = lumi*0.788/497200*len(tttw_frame)/len(test_pred[2])

print tttt_scale

print("iterative test\n")
for x in range(0, 11):
    i = x*0.1
    b = len(test_pred[0][test_pred[0] > i])*tttt_scale
    s = len(test_pred[1][test_pred[1] > i])*tttj_scale + len(test_pred[2][test_pred[2] > i])*tttw_scale
    if b <= 0:
        break

    print("%s: s: %f   b: %f   s/sqrt(b) = %f     s/sqrt(b+s) = %f " % (i, s, b, s/sqrt(b), s/sqrt(s+b)))


plt.hist(test_pred[0], np.linspace(0,1,21), color='r',alpha=0.5,label='4top',density=True)
plt.hist(test_pred[1], np.linspace(0,1,21), color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2], np.linspace(0,1,21), color='g', alpha=0.5,label='3top+W', density=True)
plt.hist(test_pred[3], np.linspace(0,1,21), color='orange', alpha=0.5,label='ttw', density=True)
plt.hist(test_pred[4], np.linspace(0,1,21), color='purple', alpha=0.5,label='ttz', density=True)
plt.hist(test_pred[4], np.linspace(0,1,21), color='firebrick', alpha=0.5,label='tth', density=True)
plt.legend()
plt.savefig("ttv.png")
plt.show()


plt.hist(test_pred[1], np.linspace(0,1,21), color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2], np.linspace(0,1,21), color='g', alpha=0.5,label='3top+W', density=True)
plt.hist(vvv_pred[0], np.linspace(0,1,21), color='y', alpha=0.5,label='www', density=True)
plt.hist(vvv_pred[1], np.linspace(0,1,21), color='c', alpha=0.5,label='wwz', density=True)
plt.hist(vvv_pred[2], np.linspace(0,1,21), color='m', alpha=0.5,label='wzz', density=True)
plt.hist(vvv_pred[3], np.linspace(0,1,21), color='lightcoral', alpha=0.5,label='zzz', density=True)
plt.legend()
plt.savefig("vvv.png")
plt.show()

plt.hist(test_pred[1], np.linspace(0,1,21), color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2], np.linspace(0,1,21), color='g', alpha=0.5,label='3top+W', density=True)
plt.hist(ttvv_pred[0], np.linspace(0,1,21), color='royalblue', alpha=0.5,label='ttwh', density=True)
plt.hist(ttvv_pred[1], np.linspace(0,1,21), color='midnightblue', alpha=0.5,label='ttwz', density=True)
plt.hist(ttvv_pred[2], np.linspace(0,1,21), color='darkblue', alpha=0.5,label='ttww', density=True)
plt.hist(ttvv_pred[3], np.linspace(0,1,21), color='mediumblue', alpha=0.5,label='tthh', density=True)
plt.hist(ttvv_pred[4], np.linspace(0,1,21), color='blue', alpha=0.5,label='ttzh', density=True)
plt.hist(ttvv_pred[5], np.linspace(0,1,21), color='mediumpurple', alpha=0.5,label='ttzz', density=True)

plt.legend()
plt.savefig("ttvv.png")
plt.show()

plt.hist(test_pred[1], np.linspace(0,1,21), color='b', alpha=0.5,label='3top+J', density=True)
plt.hist(test_pred[2], np.linspace(0,1,21), color='g', alpha=0.5,label='3top+W', density=True)
plt.hist(vv_pred[0], np.linspace(0,1,21), color='royalblue', alpha=0.5,label='zz', density=True)
plt.hist(vv_pred[1], np.linspace(0,1,21), color='midnightblue', alpha=0.5,label='wz', density=True)
plt.hist(vv_pred[2], np.linspace(0,1,21), color='darkblue', alpha=0.5,label='ww', density=True)
plt.hist(vv_pred[3], np.linspace(0,1,21), color='mediumblue', alpha=0.5,label='ww_ewk', density=True)
plt.legend()
plt.savefig("vv.png")
plt.show()
# plt.hist(train_pred[0], np.linspace(0,1,21),color='r',alpha=0.5,label='4top',density=True)
# plt.hist(train_pred[1], np.linspace(0,1,21) ,color='b', alpha=0.5,label='3top+J', density=True)
# plt.hist(train_pred[2], np.linspace(0,1,21) ,color='g', alpha=0.5,label='3top+W', density=True)
# plt.legend()
# plt.show()

# def sortDict(dict):
#     return sorted(dict.items(), key=lambda kv: kv[1], reverse=True)


# pred_all = np.concatenate(test_pred)
# truth_all = pandas.concat(split_test)["isSignal"]


# fpr, tpr,_ = roc_curve(truth_all, pred_all)
# auc_test = roc_auc_score(truth_all, pred_all)

# print "AUC", auc_test

# import pprint

# print
# # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# print "\ngain:"
# pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='gain')))
# print "\nweight:"
# pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='weight')))
# print "\ncoverage:"
# pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='cover')))
 

# def calcChangeAUC(Xtrain, ytrain, Xtest, test_truth, oldauc):
#     mod = xgb.XGBRegressor()
#     fitModel = mod.fit(Xtrain, ytrain)
#     test_pred = mod.predict(Xtest)
#     return oldauc - roc_auc_score(test_truth, test_pred)


# # auc_changes = dict()
# # for var in bNames:
# #     X_testNew = pandas.concat(split_test).drop([var, "isSignal", "weight"], axis=1)
# #     y_testNew = pandas.concat(split_test)["isSignal"]
# #     change = calcChangeAUC(X_train.drop([var],axis=1), y_train, X_testNew, y_testNew, auc_test)
# #     auc_changes[var] = change
# #     print var, change

# # print "\nChange in AUC:"
# # pprint.pprint(sortDict(auc_changes))

    
# plt.plot(fpr,tpr, label="test AUC = {:.3f}".format(auc_test))
# plt.legend()
# plt.show()
