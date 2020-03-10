import pandas
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import Utilities.helper as helper
from Utilities.treeGroup import TreeGroup
import uproot, os, argparse, logging

parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('-f', '--infile', required=True, type=str, help='Infile/outfile name')
parser.add_argument('--mva', action="store_true", help='Run TMVA along with make plots')
parser.add_argument('--useNeg', action="store_true", help='Use Negative event weights in training')
parser.add_argument('--show', action="store_true", help='Set if one wants to see plots when they are made (default is to run in batch mode)')
parser.add_argument('-l', '--lumi', type=float, default=140., help='Luminosity to use for graphs given in ifb: default 140')
parser.add_argument('--debug', action="store_true", help="Add 'debug' flag to get more information")

args = parser.parse_args()

bNames = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "DilepCharge", "NlooseBJets", "NtightBJets", "NlooseLeps"]

#inputs to change
bins = np.linspace(0,1,41)
saveModelName = None
helper.saveDir = args.infile
helper.doShow = args.show
TreeGroup.bNames = bNames
TreeGroup.bins = bins
logging.basicConfig(level=(logging.INFO if args.debug else logging.WARNING))

if not os.path.isdir(args.infile):
    os.mkdir(args.infile)

infile = uproot.open("inputTrees.root")

top3SF = 1.0
top4_g = TreeGroup(infile, ["4top2016"], [0.0092], False)
top3_g = TreeGroup(infile, ["tttj", "tttw"], [top3SF*0.000474, top3SF*0.000788], True)

signal_g = top3_g
signal_g.histOpt["alpha"] = 0.5
other_g = top4_g

ttv_g = TreeGroup(infile, ["ttw", "ttz", "tth2nonbb"])
ttvv_g = TreeGroup(infile, ["ttwh", "ttwz", "ttww", "tthh", "ttzh", "ttzz"])
vvv_g = TreeGroup(infile, ["www", "wwz", "wzz", "zzz"])
vv_g = TreeGroup(infile, ["zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "vh2nonbb"])
xg_g = TreeGroup(infile, ["ttg_dilep", "wwg", "wzg", "zg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz", "wg"])
extra_g = TreeGroup(infile, ["tzq", "st_twll", "DYm50", "DYm10-50", "wjets"])#"ttbar"

all_samples = [signal_g, other_g, ttv_g, ttvv_g, vvv_g, vv_g, xg_g, extra_g]
all_bkg = [other_g, ttv_g, ttvv_g, vvv_g, vv_g, xg_g, extra_g]
special_samples = [signal_g, other_g]

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
    logging.info([(name, len(i)) for i, name in zip(frame.f, frame.label)])
print 


if args.mva:
    # Setup XGBoost stuff
    X_train = pandas.concat(split_train,ignore_index=True).drop(TreeGroup.exclude, axis=1)
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
    
    fitModel.save_model("{}/model.bin".format(args.infile))



loadModel = xgb.Booster({'nthread': 4})
loadModel.load_model("{}/model.bin".format(args.infile))

for frame in all_samples:
    frame.makePred(loadModel)

# Setting up background
bkg_g = None
from copy import deepcopy
for frame in all_bkg:
    if bkg_g:  bkg_g += frame
    else:      bkg_g = deepcopy(frame)

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
print( "Signal to similar: {}".format(helper.approxLikelihood(signal_g.getHist(bins), other_g.getHist(bins))))
print( "Signal to all Bkg: {}".format(helper.approxLikelihood(signal_g.getHist(bins), bkg_g.getHist(bins))))
print 
helper.makeROC(special_samples, "3vs4")
helper.makeROC(all_samples, "all")

# sorted usefulness of variables

fscoreFunc = loadModel.get_fscore().get
for rank,name in sorted([(fscoreFunc(name) if fscoreFunc(name) else 0, name) for name in loadModel.feature_names], reverse=True):
    print("{:4s} {:15s}".format(str(rank),str(name)))


 

    
