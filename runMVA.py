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

#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('-o', '--out', required=True, type=str, help='output directory name')
parser.add_argument('--tmva', action="store_true", help='Run TMVA along with make plots')
parser.add_argument('--xgb', action="store_true", help='Run XGBoost along with make plots')
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

if not os.path.isdir(outname):
    os.mkdir(outname)



if args.tmva or args.xgb:
    if args.tmva:
        mvaRunner = TMVAMaker("inputTrees.root", outname)
    elif args.xgb:
        mvaRunner = XGBoostMaker("inputTrees.root", outname)
    else:
        raise Exception("Not implimented for mva type ({})".format(mvaType))

    mvaRunner.addVariables(usevar, specVar)
    mvaRunner.addCut(cut)
    mvaRunner.addGroup(["tttj", "tttw"], "Signal", True)
