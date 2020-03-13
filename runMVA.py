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
from Utilities.MvaMaker import TMVAMaker, XGBoostMaker
from Utilities.MVAPlotter import MVAPlotter

#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('type', type=str, help='Run TMVA along with make plots')
parser.add_argument('-o', '--out', required=True, type=str, help='output directory name')
parser.add_argument('-t', '--train', action="store_true", help="Run the training")

parser.add_argument('--useNeg', action="store_true", help='Use Negative event weights in training')
parser.add_argument('--show', action="store_true", help='Set if one wants to see plots when they are made (default is to run in batch mode)')
parser.add_argument('-l', '--lumi', type=float, default=140., help='Luminosity to use for graphs given in ifb: default 140')

args = parser.parse_args()

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps",]
specVar= ["newWeight", "DilepCharge"]

outname = args.out
helper.saveDir = outname
helper.doShow = args.show
lumi=args.lumi*1000
cut = 'HT>150&&DilepCharge>0&&MET>25'
bins = np.linspace(0,1,)


if not os.path.isdir(outname):
    os.mkdir(outname)


groups = np.array([["Signal", ["tttj", "tttw"]],
             ["FourTop", ["4top2016",]], 
             ["Background", ["ttw", "ttz", "tth2nonbb", "ttwh", "ttzz", "ttzh", "tthh", "ttww", "ttwz", "www", "wwz", "wzz", "zzz", "zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "ttg_dilep", "wwg", "wzg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz", "wg", "tzq", "st_twll", "DYm50"]],
])
    

if args.train:
    if args.type == "tmva":
        mvaRunner = TMVAMaker("inputTrees.root", outname)
    elif args.type == "xgb":
        mvaRunner = XGBoostMaker("inputTrees.root", outname)
    else:
        raise Exception("Not implimented for mva type ({})".format(mvaType))


    mvaRunner.addVariables(usevar, specVar)
    mvaRunner.addCut(cut)
    for groupName, samples in groups:
        mvaRunner.addGroup(samples, groupName)
    mvaRunner.train()

output = MVAPlotter(outname, groups.T[0], lumi)

helper.makeROC(output.setupROC("Signal", ["FourTop"]), name="SignalvsFourTop")
helper.makeROC(output.setupROC("Signal", ["FourTop", "Background"]), name="SignalvsAll")
helper.makeROC(output.setupROC("FourTop", ["Signal"]), name="FourTopvsSingal")
helper.makeROC(output.setupROC("FourTop", ["Signal", "Background"]), name="FourTopvsAll")



helper.plotFunc(output.getSample(["Signal"]), output.getSample(["FourTop", "Background"]),
                lumi, "BDT.Signal", np.linspace(0,1,40))
helper.plotFunc(output.getSample(["Signal"]), output.getSample(["FourTop", "Background"]),
                lumi, "BDT.FourTop", np.linspace(0,1,40))
helper.plotFunc(output.getSample(["Signal"]), output.getSample(["FourTop", "Background"]),
                lumi, "BDT.Background", np.linspace(0,1,40))

stobBins = np.linspace(0, 1, 50)
helper.StoB(output.getHist(["Signal"], "BDT.Signal", stobBins),
            output.getHist(["FourTop", "Background"], "BDT.Signal", stobBins), 
            stobBins, "SignalvsAll")
print helper.approxLikelihood(output.getHist(["Signal"], "BDT.Signal", stobBins),
                              output.getHist(["FourTop", "Background"], "BDT.Signal", stobBins), )

#helper.makeROC()

    
