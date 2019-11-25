import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


class treeGroup:
    def __init__(self, infile, label, xsec, isSig=False):
        self.f = [self.getFrame(infile, dirN, isSig) for dirN in label]
        self.xsec = xsec
        self.label = label
        self.sumweight = [infile[dirN]["sumweights"].values.sum() for dirN in label]
        self.extraWeight = [1]*len(self.xsec)

        self.bins = np.linspace(0,1,21)
        self.histOpt = {"stacked":True, "alpha":0.5, "edgecolor":'black', "histtype":'stepfilled'}
        self.madeWeight = False

    def extend(self, other):
        self.f.extend(other.f)
        self.xsec.extend(other.xsec)
        self.label.extend(other.label)
        self.sumweight.extend(other.sumweight)
        self.extraWeight.extend(other.extraWeight)
        
    def makePred(self, func):
        self.pred = [func.predict(frame.drop(["isSignal","weight"], axis=1)) for frame in self.f]

    def makeWgt(self, lumi=140000):
        self.wgt = [[mxsec*wgt*mwgt*lumi for wgt in mframe["weight"]] for mxsec, mframe, mwgt in zip(self.xsec, self.f, self.extraWeight)]
        
    def getFrame(self, infile, dirName, signal):
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

    def makeTest(self, num, idx=0):
        self.extraWeight[idx] *= len(self.f[idx])
        train_return = self.f[idx].truncate(after=num-1)
        self.f[idx] = self.f[idx].truncate(before=num)
        self.extraWeight[idx] /= len(self.f[idx])
        return train_return

    def makePlot(self, cls, isDen=True):
        if not self.madeWeight:
            self.makeWgt()

        return cls.hist(self.pred, self.bins, weights=self.wgt, label=self.label, density=isDen, **self.histOpt)


    
bNames = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", \
          "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", \
          "b1Pt", "b2Pt", "b3Pt", "b4Pt", "DilepCharge", "Shape1", "Shape2", \
          "weight",]
infile4top = uproot.open("test_tree_1108_tttt.root")
infile3topW = uproot.open("test_tree_1108_tttw.root")
infile3topJ = uproot.open("test_tree_1108_tttj.root")
infileAll = uproot.open("full_MC_tree_only2lep.root")


top4_g = treeGroup(infile4top, ["tttt"], [0.0092])
top3_g = treeGroup(infile3topJ, ["tttj"], [0.000474], True)
tmp_g = treeGroup(infile3topW, ["tttw"], [0.000788], True)
top3_g.extend(tmp_g)
ttv_g = treeGroup(infileAll,
                  ["ttw", "ttz", "tth2nonbb"],
                  [0.2043, 0.2529, 0.2151])
ttvv_g = treeGroup(infileAll,
                  ["ttwh", "ttwz", "ttww", "tthh", "ttzh", "ttzz"],
                  [0.001582, 0.003884, 0.01150, 0.000757, 0.001535, 0.001982])
vvv_g = treeGroup(infileAll,
                   ["www", "wwz", "wzz", "zzz"],
                   [0.2086, 0.1651, 0.5565, 0.01398])
vv_g = treeGroup(infileAll,
                 ["zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "vh2nonbb"],
                 [1.256, 4.4297, 0.16975, 0.03711, 0.9561])
xg_g = treeGroup(infileAll,
                 ["ttg_dilep", "wwg", "wzg", "zg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz"],
                 [0.632, 0.2147, 0.04123, 123.9, 0.769, 0.77, 0.01181])
extra_g = treeGroup(infileAll,
                    ["tzq", "st_twll", "DYm50", "ttbar"],
                    [0.0758, 0.01123, 6020.85, 831.762])
# missing tGjets, WGtolnug, wjets, 
fact = 3000

split_train = list()
split_train.append(top4_g.makeTest(fact*6))
split_train.append(top3_g.makeTest(fact*2,0))
split_train.append(top3_g.makeTest(fact*4,1))
split_train.append(ttv_g.makeTest(fact*2,0))
split_train.append(ttv_g.makeTest(fact*2,1))

print [len(i) for i in split_train]
print [len(i) for i in [top4_g.f[0], top3_g.f[0], top3_g.f[1], ttv_g.f[0], ttv_g.f[1]]]

X_train = pandas.concat(split_train,ignore_index=True).drop(["isSignal", "weight"], axis=1)
w_train = pandas.concat(split_train,ignore_index=True)["weight"]
y_train = pandas.concat(split_train, ignore_index=True)["isSignal"]

mod = xgb.XGBRegressor(n_estimators=200, \
                       eta = 0.07,\
                       max_depth = 5, \
                       subsample = 0.6, \
                       alpha = 8.0, \
                       gamma = 2.0, \
                       )
fitModel = mod.fit(X_train, y_train, sample_weight=w_train)

train_pred = [mod.predict(frame.drop(["isSignal", "weight"], axis=1)) for frame in split_train]

top4_g.makePred(fitModel)
top3_g.makePred(fitModel)
ttv_g.makePred(fitModel)
ttvv_g.makePred(fitModel)
vvv_g.makePred(fitModel)
vv_g.makePred(fitModel)
xg_g.makePred(fitModel)
extra_g.makePred(fitModel)

isDen = True
doShow = False

top4_val, _,_ = top4_g.makePlot(plt, isDen)
top3_val, _,_ = top3_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("4vs3.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
ttv_val, _,_ = ttv_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("ttv.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
ttvv_val, _,_ = ttvv_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("ttvv.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
vvv_val, _,_ = vvv_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("vvv.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
vv_val, _,_ = vv_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("vv.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
xg_val, _,_ = xg_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("xg.png")
if doShow: plt.show()
plt.clf()
plt.cla()

top3_val, _,_ = top3_g.makePlot(plt, isDen)
extra_val, _,_ = extra_g.makePlot(plt, isDen)
plt.legend()
plt.savefig("extra.png")
if doShow: plt.show()
plt.clf()
plt.cla()

exit()



print("iterative test\n")
print("3vs4 top")
for x in range(0, 11):
    i = x*0.1
    b = len(test_pred[0][test_pred[0] > i])*tttt_scale
    s = len(test_pred[1][test_pred[1] > i])*tttj_scale + len(test_pred[2][test_pred[2] > i])*tttw_scale
    if b <= 0:
        break

    print("%s: s: %f   b: %f   s/sqrt(b) = %f     s/sqrt(b+s) = %f " % (i, s, b, s/sqrt(b), s/sqrt(s+b)))




    
top3_val, _,_ = plt.hist(top3_pred, bins, weights=top3_wgt, label=top3_label, density=isDen, **histOpt)
top4_val, _,_ = plt.hist(top4_pred, bins, weights=top4_wgt, label=top4_label, density=isDen, **histOpt)
plt.legend()
plt.savefig("4vs3.png")
plt.show()

top3_val, _,_ = plt.hist(top3_pred, bins, weights=top3_wgt, label=top3_label, density=isDen, **histOpt)
ttv_val, _,_ = plt.hist(ttv_pred, bins, weights=ttv_wgt, label=ttv_label, density=isDen, **histOpt)
plt.legend()
plt.savefig("ttv.png")
plt.show()

top3_val, _,_ = plt.hist(top3_pred, bins, weights=top3_wgt, label=top3_label, density=isDen, **histOpt)
ttvv_val, _,_ = plt.hist(ttvv_pred, bins, weights=ttvv_wgt, label=ttvv_label, density=isDen, **histOpt)
plt.legend()
plt.savefig("ttvv.png")
plt.show()

top3_val, _,_ = plt.hist(top3_pred, bins, weights=top3_wgt, label=top3_label, density=isDen, **histOpt)
vvv_val, _,_ = plt.hist(vvv_pred, bins, weights=vvv_wgt, label=vvv_label, density=isDen, **histOpt)
plt.legend()
plt.savefig("vvv.png")
plt.show()

top3_val, _,_ = plt.hist(top3_pred, bins, weights=top3_wgt, label=top3_label, density=isDen, **histOpt)
vv_val, _,_ = plt.hist(vv_pred, bins, weights=vv_wgt, label=vv_label, density=isDen, **histOpt)
plt.legend()
plt.savefig("vv.png")
plt.show()


def sortDict(dict):
    return sorted(dict.items(), key=lambda kv: kv[1], reverse=True)


pred_all = np.concatenate(test_pred[:3])
truth_all = pandas.concat(split_test[:3])["isSignal"]


fpr, tpr,_ = roc_curve(truth_all, pred_all)
auc_test = roc_auc_score(truth_all, pred_all)

print "AUC", auc_test
plt.plot(fpr,tpr, label="test AUC = {:.3f}".format(auc_test))
plt.legend()
plt.show()

import pprint

print
# importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
print "\ngain:"
pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='gain')))
print "\nweight:"
pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='weight')))
print "\ncoverage:"
pprint.pprint(sortDict(fitModel.get_booster().get_score(importance_type='cover')))
 

    
