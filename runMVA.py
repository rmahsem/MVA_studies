#!/usr/bin/env python
import os, math, sys
import copy
import numpy as np
import argparse
from Utilities.MvaMaker import XGBoostMaker
from Utilities.MVAPlotter import MVAPlotter
import matplotlib.pyplot as plt


# Command line arguments
parser = argparse.ArgumentParser(description='Run TMVA over 4top')
parser.add_argument('-o', '--out', required=True, type=str, help='output directory name')
parser.add_argument('-t', '--train', action="store_true", help="Run the training")
parser.add_argument('--show', action="store_true", help='Set if one wants to see plots when they are made (default is to run in batch mode)')
parser.add_argument('-l', '--lumi', type=float, default=140., help='Luminosity to use for graphs given in ifb: default 140')
args = parser.parse_args()


# Variables used in Training
usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps", "LepCos", "JetLep1_Cos", "JetLep2_Cos", "JetBJet_DR", "Lep_DR", "JetBJet_Cos"]

# Variables outputed (not used in training)
specVar= ["newWeight", "DilepCharge", "weight"]

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
                   ["Background", ["ttw", "ttz", "tth2nonbb", "ttww", "ttg_lepfromTbar", "ttg_lepfromT", "ttwh", "ttzz", "ttzh",  "ttwz", "www", "wwz", "wzz", "zzz", "zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "ttg_dilep", "wwg", "wzg","ggh2zz", "wg", "tzq", "st_twll", "DYm50",]],
])

inputTree = "inputTrees_new.root"
groupOrdered = [item for sublist in [l[1] for l in groups] for item in sublist]

############
# training #
############

if args.train:
    mvaRunner = XGBoostMaker(inputTree)       
    mvaRunner.addVariables(usevar, specVar)
    mvaRunner.addCut(cut)
    for groupName, samples in groups:
        mvaRunner.addGroup(samples, groupName)
    
    # Use this if multiclass Train
    mvaRunner.train()
    # Use something like this if want binary (trains name against all)
    #
    # mvaRunner.train("Signal")
    mvaRunner.output(outname)

###############
# Make Plots  #
###############
stobBins = np.linspace(0.0, 1, 50)
nbins2d=50
stob2d = np.linspace(0.0,1.0,nbins2d+1)

output = MVAPlotter(outname, groups.T[0], lumi)
output.set_show(args.show)
gSet = output.get_sample()

output.write_out("preSelection_BDT.2020.06.03_single.root", inputTree)
output.plot_fom("Signal", ["Background"], "BDT.Signal", stobBins, "")
output.make_roc("Signal", ["FourTop", "Background"], "Signal", "SignalvsAll")
output.print_info("BDT.Signal", groupOrdered)
output.plot_all_shapes("NJets", np.linspace(0,15,16), "allGroups")

print("FourTop: ", output.approx_likelihood("Signal", ["Background", "FourTop"], "BDT.FourTop", stobBins))
print("Background: ", output.approx_likelihood("Signal", ["Background", "FourTop"], "BDT.Background", stobBins))


maxSBVal = output.plot_fom_2d("Signal", "BDT.Background", "BDT.FourTop", stob2d, stob2d)

output.plot_func_2d("Signal", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "Signal", lines=maxSBVal[1:])
output.plot_func_2d("Background", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "Background", lines=maxSBVal[1:])
output.plot_func_2d("FourTop", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "FourTop", lines=maxSBVal[1:])

        
output.apply_cut("BDT.FourTop>{}".format(maxSBVal[2]))
output.apply_cut("BDT.Background>{}".format(maxSBVal[1]))

output.write_out("postSelection_BDT.2020.06.03_SignalSingle.root", inputTree)






