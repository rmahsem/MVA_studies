#!/usr/bin/env python
import sys
import os
from os import environ, path
import ROOT
import copy
import numpy as np
ROOT.gROOT.SetBatch(True)

#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps",]
specVar= ["newWeight"]
outname = 'BDT_testing' #Output file name
UseMethod = ["BDTCW"] #More methods in the cfg
runTMVA = True


if runTMVA:
    

    sigNames = [["tttj", 0.000474], ["tttw", 0.000788]]
    bkgNames = [["4top2016", 0.0092],
                #["ttw", 0.2043], ["ttz", 0.2529], ["tth2nonbb", 0.2151] , ["ttwh", 0.001582], ["ttzz", 0.001982], ["ttzh", 0.001535], ["tthh", 0.000757], ["ttww", 0.01150], ["ttwz", 0.003884], ["www", 0.2086], ["wwz", 0.1651],  ["wzz", 0.5565],  ["zzz", 0.01398], ["zz4l_powheg", 1.256], ["wz3lnu_mg5amcnlo", 4.4297], ["ww_doubleScatter", 0.16975],  ["wpwpjj_ewk", 0.03711],  ["vh2nonbb", 0.9561], ["ttg_dilep", 0.632],  ["wwg", 0.2147],  ["wzg", 0.04123],  ["zg", 123.9], ["ttg_lepfromTbar", 0.769], ["ttg_lepfromT", 0.77], ["ggh2zz", 0.01181], ["wg", 405.271], ["tzq", 0.0758], ["st_twll", 0.01123], ["DYm50", 6020.85], ["ttbar", 831.762], ["DYm10-50", 18610]
    ]
    #, ["wjets", 61334.9],
    infile = ROOT.TFile("inputTrees.root")
    fout = ROOT.TFile(outname+".root","RECREATE")

    analysistype = 'AnalysisType=Classification'                     
    # analysistype = 'AnalysisType=MultiClass' #For many backgrounds
    factoryOptions = [ "!Silent", "Color", "DrawProgressBar", "Transformations=I", analysistype]
    ROOT.TMVA.Tools.Instance()
    factory = ROOT.TMVA.Factory("TMVAClassification", fout,":".join(factoryOptions))
    # Weight folder name to store the output

    dataset = ROOT.TMVA.DataLoader('MVA_weights')

    for var in usevar:
        print var
        dataset.AddVariable(var) #my int variables have a n_* in the name
    for var in specVar:
        dataset.AddSpectator(var)

    st = list()
    bot = list()
    bt = list()

    sigSum = np.sum([pair[1] for pair in sigNames])
    bkgSum = np.sum([pair[1] for pair in bkgNames])

    addcut = 'HT>150&&DilepCharge>0&&MET>25'
    
    for pair in sigNames:
        sumweight = infile.FindObjectAny("sumweight_%s" % pair[0])
        tree = infile.Get(pair[0])
        #dataset.AddSignalTree(infile.Get(pair[0]), pair[1]*sumweight.GetBinContent(2)/sumweight.GetBinContent(1))
        dataset.AddSignalTree(tree, pair[1]/sigSum/tree.GetEntries(addcut))

    for pair in bkgNames:
        sumweight = infile.FindObjectAny("sumweight_%s" % pair[0])
        tree = infile.Get(pair[0])
        #dataset.AddBackgroundTree(infile.Get(pair[0]), pair[1]/sumweight.GetBinContent(1))
        dataset.AddBackgroundTree(tree, pair[1]/bkgSum/tree.GetEntries(addcut))

    # dataset.SetBackgroundWeightExpression("newWeight")
    # dataset.SetSignalWeightExpression("newWeight")

    
    cut = ROOT.TCut(addcut)

    modelname = 'trained_weights' #For the output name"NormMode=EqualNumEvents"

    dataset.PrepareTrainingAndTestTree(cut, ":".join(["SplitMode=Random:NormMode=EqualNumEvents",]))
    # ROOT.TMVA.VariableImportance(dataset)

    # "CrossEntropy"
    # "GiniIndex"
    # "GiniIndexWithLaplace"
    # "MisClassificationError"
    # "SDivSqrtSPlusB"
    sepType="GiniIndex"

    methodInput =["!H", "NTrees=500", "nEventsMin=150", "MaxDepth=5", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType={}".format(sepType),"nCuts=20","PruneMethod=NoPruning","IgnoreNegWeightsInTraining",]
    # ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType={}".format(sepType),"nCuts=50"]
    # 
     # ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType={}".format(sepType),"nCuts=50"]
    # #["!H", "!V", "NTrees=500", "nEventsMin=150", "MaxDepth=5", # orig 4
    #                            "BoostType=AdaBoost", "AdaBoostBeta=0.", # orig 0.5
    #                            "SeparationType={}".format(sepType),
    #                            "nCuts=20",
    #                            "PruneMethod=NoPruning",
    #                            "IgnoreNegWeightsInTraining",
    #                            # "MinNodeSize=5", # in %]

    method = factory.BookMethod(dataset, ROOT.TMVA.Types.kBDT, "BDT",
                           ":".join(methodInput))

    factory.TrainAllMethods() 
    factory.TestAllMethods() 
    factory.EvaluateAllMethods() 

    fout.Close()




lumi=140000

import matplotlib.pyplot as plt
import uproot, pandas,math

f_out = uproot.open(outname+".root")


test_tree = f_out["MVA_weights"]["TestTree"].pandas.df(["*"])
train_tree = f_out["MVA_weights"]["TrainTree"].pandas.df(["classID"])

sigFac = (len(test_tree[test_tree["classID"]==0])+len(train_tree[train_tree["classID"]==0]))/len(test_tree[test_tree["classID"]==0])
bkgFac = (len(test_tree[test_tree["classID"]==1])+len(train_tree[train_tree["classID"]==1]))/len(test_tree[test_tree["classID"]==1])

signal_df = test_tree[test_tree["classID"] == 0]
background_df = test_tree[test_tree["classID"] == 1]

def getWeight(df, lumi, fc):
    return lumi*df["newWeight"]*fc

def plotFunc(sig, bkg, lumi, name, bins):
    fig, ax = plt.subplots()
    bkgHist = ax.hist(x=bkg[name], bins=bins, weights=getWeight(bkg, lumi, bkgFac),label="Background", histtype="step", linewidth=1.5)
    sigHist = ax.hist(x=sig[name], bins=bins, weights=getWeight(sig, lumi, sigFac), label="Signal", histtype="step",linewidth=1.5)
    ax.legend()
    ax.set_xlabel(name)
    ax.set_ylabel("Events/bin")
    ax.set_title("Lumi = {} ifb".format(lumi/1000))
    fig.tight_layout()
    plt.show()

#plotFunc(signal_df, background_df, 140000, "BDT", np.linspace(-0.7,0.7,51))

sbBins = np.linspace(-0.75, 0.75, 51)
b = np.histogram(background_df["BDT"], bins=sbBins, weights=getWeight(background_df, lumi, bkgFac))[0]
s = np.histogram(signal_df["BDT"], bins=sbBins, weights=getWeight(signal_df,lumi, sigFac))[0]

import Utilities.helper as helper

helper.StoB(s,b, sbBins, "tester", noSB=True)


