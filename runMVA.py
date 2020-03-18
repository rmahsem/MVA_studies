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

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps",]
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
                   ["FourTop", ["4top2016",]], 
                   ["Background", ["ttw", "ttz", "tth2nonbb", "ttwh", "ttzz", "ttzh", "tthh", "ttww", "ttwz", "www", "wwz", "wzz", "zzz", "zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "ttg_dilep", "wwg", "wzg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz", "wg", "tzq", "st_twll", "DYm50"]],
])
    
############
# training #
############

if args.train:
    if args.train == "tmva":
        mvaRunner = TMVAMaker("inputTrees.root", outname)
    elif args.train == "xgb":
        mvaRunner = XGBoostMaker("inputTrees.root", outname)
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
output = MVAPlotter(outname, groups.T[0], lumi)
output.setDoShow(args.show)

output.makeROC("Signal", ["FourTop"], "SignalvsFourTop")
output.makeROC("Signal", ["FourTop", "Background"], "SignalvsAll")
output.makeROC("FourTop", ["Signal"], "FourTopvsSingal")
output.makeROC("FourTop", ["Signal", "Background"], "FourTopvsAll")

output.plotFunc("Signal", ["FourTop", "Background"], "BDT.Signal", np.linspace(0,1,40), "_SigVsAll")
output.plotFunc("Signal", ["FourTop"], "BDT.Signal", np.linspace(0,1,40), "_SigVsFourTop")
output.plotFunc("Signal", ["Background"], "BDT.Signal", np.linspace(0,1,40), "_SigVsBackground")
output.plotFunc("Signal", ["FourTop", "Background"], "BDT.FourTop", np.linspace(0,1,40), "_SigVsAll")
output.plotFunc("Signal", ["FourTop"], "BDT.FourTop", np.linspace(0,1,40), "_SigVsFourTop")
output.plotFunc("Signal", ["FourTop", "Background"], "BDT.Background", np.linspace(0,1,40), "_SigVsAll")
output.plotFunc("Signal", ["FourTop"], "BDT.Background", np.linspace(0,1,40), "_SigVsFourTop")


stobBins = np.linspace(0, 1, 50)
output.plotStoB("Signal", ["FourTop", "Background"], "BDT.Signal", stobBins, "SignalVsAll")

print "Signal: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.Signal", stobBins)
print "FourTop: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.FourTop", stobBins)
print "Background: ", output.approxLikelihood("Signal", ["FourTop", "Background"], "BDT.Background", stobBins)

    
