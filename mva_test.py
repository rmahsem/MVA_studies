import pandas
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import Utilities.helper as helper



#inputs to change
bins = np.linspace(0,1,41)
doShow = False
saveDir = "3top_blah"
saveModelName = None


helper.saveDir = saveDir
helper.doShow = doShow
from Utilities.samples import *


# Create Train and Test groups
split_train = list()
#for sample in all_samples:
for sample in special_samples:
    for i in range(len(sample)):
        nEvt = len(sample[i])
        if nEvt > 1000:
            split_train.append(sample.makeTest(nEvt/3,i))
        elif nEvt > 500:
            split_train.append(sample.makeTest(nEvt/6,i))

for frame in all_samples:
    print([(name, len(i)) for i, name in zip(frame.f, frame.label)])
print()



# Setup XGBoost stuff
X_train = pandas.concat(split_train,ignore_index=True).drop(["isSignal", "weight"], axis=1)
y_train = pandas.concat(split_train, ignore_index=True)["isSignal"]

scalefact = 1.0*np.sum(y_train)/len(y_train)
scale_train = np.where(y_train, 1, scalefact)

sig_wgt, bkg_wgt = list(), list()
for fr in split_train:
    if np.unique(fr["isSignal"]) == 1:
        sig_wgt.append(np.sum(fr["weight"]))
    else: 
        bkg_wgt.append(np.sum(fr["weight"]))


#weight nonsense
# use a "mean xsec" method to keep weights O(1) while including relative weights between xsec
sig_wgt = np.array(sig_wgt)
bkg_wgt = np.array(bkg_wgt)
tot_mean_wgt = np.concatenate([sig_wgt/np.mean(sig_wgt), bkg_wgt/np.mean(bkg_wgt)])
w_train = np.concatenate([wgt*np.ones(len(frame)) for frame, wgt in zip(split_train, tot_mean_wgt)])

print( "In weights:", np.unique(w_train))



# XGBoost training
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)

evallist  = [(dtrain,'train')]
num_round=200
param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.09
param['max_depth'] = 5
param['silent'] = 1
param['nthread'] = 2
param['eval_metric'] = "auc"
param['subsample'] = 0.9
param['colsample_bytree'] = 0.5
fitModel = xgb.train(param.items(), dtrain, num_round, evallist,early_stopping_rounds=100, verbose_eval=100 )

if saveModelName:
    fitModel.save_model(saveModelName)
    
for frame in all_samples:
    frame.makePred(fitModel)

    
# Setting up background
bkg_g = None
from copy import deepcopy
for frame in [other_g, ttv_g, ttvv_g, vvv_g, vv_g, xg_g, extra_g]:
    if bkg_g:
        bkg_g += frame
    else:
        
        bkg_g = deepcopy(frame)

# Make Plots

helper.createPlot(other_g, signal_g, "4vs3", bins)
helper.createPlot(ttv_g, signal_g, "ttv", bins)
helper.createPlot(ttvv_g, signal_g, "ttvv", bins)
helper.createPlot(vvv_g, signal_g, "vvv", bins)
helper.createPlot(vv_g, signal_g, "vv", bins)
helper.createPlot(xg_g, signal_g, "xg", bins)
helper.createPlot(extra_g, signal_g, "extra", bins)
helper.createPlot(bkg_g, signal_g, "AllBkg", bins)

# get Signal/Background ratio

fineBin = np.linspace(0,1,101)
helper.StoB(signal_g.getHist(fineBin), other_g.getHist(fineBin), fineBin, "4top", noSB=True)
helper.StoB(signal_g.getHist(fineBin), bkg_g.getHist(fineBin), fineBin, "All", noSB=True)

# Get approximated Likelihood values

print()
print( "Signal to similar: ", helper.approxLikelihood(signal_g.getHist(bins), other_g.getHist(bins)))
print( "Signal to all Bkg: ", helper.approxLikelihood(signal_g.getHist(bins), bkg_g.getHist(bins)))

helper.makeROC(special_samples, "3vs4")
helper.makeROC(all_samples, "all")

# sorted usefulness of variables

fscoreFunc = fitModel.get_fscore().get
for rank,name in sorted([(fscoreFunc(name) if fscoreFunc(name) else 0, name) for name in fitModel.feature_names], reverse=True):
    print( "{:4s} {:15s}".format(str(rank),str(name)))
print()

 

    
