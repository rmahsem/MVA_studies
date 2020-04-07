#!/usr/bin/env python
import sys
import os
from os import environ, path
import ROOT
import copy
import numpy as np
ROOT.gROOT.SetBatch(True)
import argparse
from Utilities.MvaMaker import TMVAMaker, XGBoostMaker
from Utilities.MVAPlotter import MVAPlotter

#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('-o', '--out', required=True, type=str, help='output directory name')
parser.add_argument('-t', '--train', type=str, default="", help="Run the training")
parser.add_argument('--show', action="store_true", help='Set if one wants to see plots when they are made (default is to run in batch mode)')
parser.add_argument('-l', '--lumi', type=float, default=140., help='Luminosity to use for graphs given in ifb: default 140')

args = parser.parse_args()

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps", "LepCos", "JetLep1_Cos", "JetLep2_Cos", "JetBJet_DR", "Lep_DR", "JetBJet_Cos"]
specVar= ["newWeight", "DilepCharge"]

outname = args.out
lumi=args.lumi*1000
cut = 'HT>150&&DilepCharge>0&&MET>25'

if not os.path.isdir(outname):
    os.mkdir(outname)

##################
# Set background #
##################

groups = np.array([["Signal", ["tttj", "tttw"]],
                   ["FourTop", ["tttt2016",]], 
                   ["Background", ["ttw", "ttz", "tth2nonbb", "ttwh", "ttzz", "ttzh", "tthh", "ttww", "ttwz", "www", "wwz", "wzz", "zzz", "zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "ttg_dilep", "wwg", "wzg", "ttg_lepfromTbar", "ttg_lepfromT","ggh2zz", "wg", "tzq", "st_twll", "DYm50",
                   ]],
])
    
############
# training #
############

if args.train:
    if args.train == "tmva":
        mvaRunner = TMVAMaker("inputTrees_new.root", outname)
    elif args.train == "xgb":
        mvaRunner = XGBoostMaker("inputTrees_new.root", outname)
    else:
        raise Exception("Not implimented for mva type ({})".format(mvaType))


    mvaRunner.addVariables(usevar, specVar)
    mvaRunner.addCut(cut)
    for groupName, samples in groups:
        mvaRunner.addGroup(samples, groupName)
    mvaRunner.train()


###############
# Make Plots  #
###############
stobBins = np.linspace(0, 1, 50)

output = MVAPlotter(outname, groups.T[0], lumi)
output.setDoShow(args.show)
#output.applyCut("BDT.Background<0.1")

output.plotStoB("Signal", ["FourTop", "Background"], "BDT.Background", stobBins, "Background.SignalVsAllRev", reverse=True)
output.plotStoB("Signal", ["FourTop", "Background"], "BDT.Signal", stobBins, "Signal.SignalVsAll")
output.plotStoB("Signal", ["FourTop", "Background"], "BDT.FourTop", stobBins, "FourTop.SignalVsAll", reverse=True)


output.plotFunc("Signal", ["FourTop", "Background"], "BDT.Background", np.linspace(0,1,40), "_SigVsAll")

gSet = output.getSample()
output.addVariable("test1", gSet["BDT.Background"]-gSet["BDT.FourTop"])
output.addVariable("test2", gSet["BDT.Background"]-gSet["BDT.Signal"])
output.addVariable("test3", gSet["BDT.FourTop"] - gSet["BDT.Signal"])

output.plotFunc("Signal", ["FourTop"], "test1", np.linspace(-1,1,40), "")
output.plotFunc("Signal", ["FourTop"], "test2", np.linspace(-1,1,40), "")
output.plotFunc("Signal", ["FourTop"], "test3", np.linspace(-1,1,40), "")



output.plotFunc("Signal", ["Background"], "BDT.Signal", np.linspace(0,1,40), "_SigVsBkg")

output.makeROC("Signal", ["Background"], "Signal", "SigvsBkg")
print "Signal: ", output.approxLikelihood("Signal", ["Background"], "BDT.Signal", stobBins)

output.plotFunc("Signal", ["FourTop"], "BDT.Signal", np.linspace(0,1,40), "_SigVsFourTop")
output.plotFunc("Signal", ["FourTop"], "BDT.FourTop", np.linspace(0,1,40), "_SigVsFourTop")

output.makeROC("Signal", ["FourTop", "Background"], "Signal", "SignalvsAll")
output.makeROC("Signal", ["FourTop", "Background"], "Background", "SignalvsAll")
output.makeROC("Signal", ["FourTop"], "FourTop", "SignalvsFourTop")
# output.makeROC("Signal", ["FourTop"], "SignalvsFourTop")
# output.makeROC("FourTop", ["Signal"], "FourTopvsSingal")
# output.makeROC("FourTop", ["Signal", "Background"], "FourTopvsAll")
# output.makeROC("Signal", ["FourTop", "Background"], "SignalvsAll")



output.plotFunc("Signal", ["FourTop", "Background"], "BDT.Signal", np.linspace(0,1,40), "_SigVsAll")
output.plotFunc("Signal", ["Background"], "BDT.Signal", np.linspace(0,1,40), "_SigVsBackground")
output.plotFunc("Signal", ["FourTop", "Background"], "BDT.FourTop", np.linspace(0,1,40), "_SigVsAll")
output.plotFunc("Signal", ["FourTop"], "BDT.Background", np.linspace(0,1,40), "_SigVsFourTop")





print "Signal: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.Signal", stobBins)
print "FourTop: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.FourTop", stobBins)
print "Background: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.Background", stobBins)

    
