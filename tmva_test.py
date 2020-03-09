#!/usr/bin/env python
import sys
import os
from os import environ, path
import ROOT
import copy
import numpy as np
ROOT.gROOT.SetBatch(True)
import argparse
import Utilities.helper as helper
#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('-f', '--infile', required=True, type=str, help='Infile/outfile name')
parser.add_argument('--mva', action="store_true", help='Run TMVA along with make plots')
parser.add_argument('--useNeg', action="store_true", help='Use Negative event weights in training')
parser.add_argument('--show', action="store_true", help='Set if one wants to see plots when they are made (default is to run in batch mode)')
parser.add_argument('-l', '--lumi', type=float, default=140., help='Luminosity to use for graphs given in ifb: default 140')

args = parser.parse_args()

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps",]
specVar= ["newWeight"]
outname = args.infile
UseMethod = ["BDTCW"] #More methods in the cfg
runTMVA = args.mva
useNeg = args.useNeg
helper.saveDir = outname
helper.doShow = args.show
lumi=args.lumi*1000


if not os.path.isdir(outname):
    os.mkdir(outname)

if runTMVA:
    sigNames = ["tttj", "tttw"]
    bkgNames = ["4top2016",
        #"ttw", "ttz", "tth2nonbb", "ttwh", "ttzz", "ttzh", "tthh", "ttww", "ttwz", "www", "wwz", "wzz", "zzz", "zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "ttg_dilep", "wwg", "wzg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz", "wg", "tzq", "st_twll", "DYm50", "ttbar",
                # Ignored for SS study (near 0 events, code complains)
              #  ["DYm10-50", 18610], ["zg", 123.9], ["vh2nonbb", 0.9561], ["wjets", 61334.9],
    ]
    
    infile = ROOT.TFile("inputTrees.root")
    fout = ROOT.TFile("{}/BDT.root".format(outname),"RECREATE")

    analysistype = 'AnalysisType=Classification'                     
    # analysistype = 'AnalysisType=MultiClass' #For many backgrounds
    factoryOptions = [ "!Silent", "Color", "DrawProgressBar", "Transformations=I", analysistype]
    ROOT.TMVA.Tools.Instance()
    factory = ROOT.TMVA.Factory("TMVAClassification", fout,":".join(factoryOptions))
    dataset = ROOT.TMVA.DataLoader('MVA_weights')

    for var in usevar:
        dataset.AddVariable(var)
    for var in specVar:
        dataset.AddSpectator(var)

    addcut = 'HT>150&&DilepCharge>0&&MET>25'

    bkgSum = dict()
    for name in bkgNames:
        tree = infile.Get(name)
        tree.Draw("(newWeight)>>tmp_{}".format(name), addcut)
        tree.Draw("(genWeight)>>tmp2_{}".format(name), addcut)
        eventRate = ROOT.gDirectory.Get("tmp_{}".format(name)).GetMean()*tree.GetEntries(addcut)
        genVsAll = 1/ROOT.gDirectory.Get("tmp2_{}".format(name)).GetMean()
        bkgSum[name] = (eventRate, genVsAll)
        
    sigSum = dict()
    for name in sigNames:
        tree = infile.Get(name)
        tree.Draw("(newWeight)>>tmp_{}".format(name), addcut)
        tree.Draw("(genWeight)>>tmp2_{}".format(name), addcut)
        eventRate = ROOT.gDirectory.Get("tmp_{}".format(name)).GetMean()*tree.GetEntries(addcut)
        genVsAll = 1/ROOT.gDirectory.Get("tmp2_{}".format(name)).GetMean()
        sigSum[name] = (eventRate, genVsAll)


    for name in sigNames:
        tree = infile.Get(name)
        fac = sigSum[name][0]/sum([i for i, j in sigSum.values()])
        if useNeg:  fac *= sigSum[name][1]
        dataset.AddSignalTree(tree, fac/tree.GetEntries(addcut))

    for name in bkgNames:
        tree = infile.Get(name)
        fac = bkgSum[name][0]/sum([i for i, j in bkgSum.values()])
        if useNeg:  fac *= bkgSum[name][1]
        dataset.AddBackgroundTree(tree, fac/tree.GetEntries(addcut))
    if useNeg:
        dataset.SetBackgroundWeightExpression("genWeight")
        dataset.SetSignalWeightExpression("genWeight")

    
    cut = ROOT.TCut(addcut)

    modelname = 'trained_weights' #For the output name"NormMode=EqualNumEvents"

    dataset.PrepareTrainingAndTestTree(cut, ":".join(["SplitMode=Random:NormMode=EqualNumEvents",]))
    
    # "CrossEntropy"
    # "GiniIndex"
    # "GiniIndexWithLaplace"
    # "MisClassificationError"
    # "SDivSqrtSPlusB"
    sepType="CrossEntropy"

    #methodInput =["!H", "NTrees=500", "nEventsMin=150", "MaxDepth=5", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType={}".format(sepType),"nCuts=20","PruneMethod=NoPruning","IgnoreNegWeightsInTraining",]
    methodInput = ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType=SDivSqrtSPlusB","nCuts=50","Pray"]
    # 
     # ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType={}".format(sepType),"nCuts=50"]

    method = factory.BookMethod(dataset, ROOT.TMVA.Types.kBDT, "BDT",
                           ":".join(methodInput))
    factory.TrainAllMethods() 
    factory.TestAllMethods() 
    factory.EvaluateAllMethods() 

    fout.Close()



import matplotlib.pyplot as plt
import uproot, pandas,math

f_out = uproot.open("{}/BDT.root".format(outname))


test_tree = f_out["MVA_weights"]["TestTree"].pandas.df(["*"])
train_tree = f_out["MVA_weights"]["TrainTree"].pandas.df(["classID"])

sigFac = (len(test_tree[test_tree["classID"]==0])+len(train_tree[train_tree["classID"]==0]))/len(test_tree[test_tree["classID"]==0])
bkgFac = (len(test_tree[test_tree["classID"]==1])+len(train_tree[train_tree["classID"]==1]))/len(test_tree[test_tree["classID"]==1])

signal_df = test_tree[test_tree["classID"] == 0].reset_index(drop=True)
background_df = test_tree[test_tree["classID"] == 1].reset_index(drop=True)
background_df.insert(0, "finalWeight", background_df["newWeight"]*lumi*bkgFac)
signal_df.insert(0, "finalWeight", signal_df["newWeight"]*lumi*sigFac)


sbBins = np.linspace(-1., 1., 101)
b = np.histogram(background_df["BDT"], bins=sbBins, weights=background_df["finalWeight"])[0]
s = np.histogram(signal_df["BDT"], bins=sbBins, weights=signal_df["finalWeight"])[0]


###############################
# OUTPUT -- MAKE GRAPHES HERE #
###############################

helper.StoB(s,b, sbBins, outname, noSB=False)
print( "Signal to Bkg: {}".format(helper.approxLikelihood(s, b)))
helper.plotFuncTMVA(signal_df, background_df, 140000, "BDT", np.linspace(-1,1.,51))
# plotFuncTMVA(signal_df, background_df, 140000, "NlooseBJets", np.linspace(0, 15, 15))
# plotFuncTMVA(signal_df, background_df, 140000, "NtightBJets", np.linspace(0, 10, 10))
# plotFuncTMVA(signal_df, background_df, 140000, "NJets", np.linspace(0, 20, 20))

helper.plotRocTMVA(outname)

