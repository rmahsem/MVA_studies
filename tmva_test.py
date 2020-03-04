#!/usr/bin/env python
import sys
import os
from os import environ, path
from TMVA_cfg import batchs, methodList
import ROOT
import copy
import numpy as np
ROOT.gROOT.SetBatch(True)

#More info at: https://root.cern.ch/download/doc/tmva/TMVAUsersGuide.pdf

usevar = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "NlooseBJets", "NtightBJets", "NlooseLeps",]
specVar= ["newWeight"]
outname = 'BDT_FT_2' #Output file name
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
    factoryOptions = ["!V", "!Silent", "Color", "DrawProgressBar", "Transformations=I", analysistype]
    ROOT.TMVA.Tools.Instance()
    factory = ROOT.TMVA.Factory("TMVAClassification", fout,":".join(factoryOptions))
    # Weight folder name to store the output

    dataset = ROOT.TMVA.DataLoader('MVA_weights')

    for var in usevar:
        print var
        dataset.AddVariable(var) #my int variables have a n_* in the name


    st = list()
    bot = list()
    bt = list()


    for pair in sigNames:
        sumweight = infile.FindObjectAny("sumweight_%s" % pair[0])
        #dataset.AddSignalTree(infile.Get(pair[0]), pair[1]*sumweight.GetBinContent(2)/sumweight.GetBinContent(1))
        dataset.AddSignalTree(infile.Get(pair[0]))

    for pair in bkgNames:
        sumweight = infile.FindObjectAny("sumweight_%s" % pair[0])
        #dataset.AddBackgroundTree(infile.Get(pair[0]), pair[1]/sumweight.GetBinContent(1))
        dataset.AddBackgroundTree(infile.Get(pair[0]))

    dataset.SetBackgroundWeightExpression("newWeight")
    dataset.SetSignalWeightExpression("newWeight")

    addcut = 'HT>150&&DilepCharge>0&&MET>25' 
    cut = ROOT.TCut(addcut)

    modelname = 'trained_weights' #For the output name

    dataset.PrepareTrainingAndTestTree(cut, ":".join(["SplitMode=Random:NormMode=NumEvents", "!V" ]))
    # ROOT.TMVA.VariableImportance(dataset)

    # "CrossEntropy"
    # "GiniIndex"
    # "GiniIndexWithLaplace"
    # "MisClassificationError"
    # "SDivSqrtSPlusB"
    sepType="GiniIndex"

    methodInput =["!H", "!V", "NTrees=500", "nEventsMin=150", "MaxDepth=5", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType={}".format(sepType),"nCuts=20","PruneMethod=NoPruning","IgnoreNegWeightsInTraining",]
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




f_out = ROOT.TFile(outname+".root")
 
normfact = 10
# xmin,xmax = -0.55, 0.95
# xmin,xmax = -0.4, 0.7
xmin,xmax = -0.45, 0.95
Histo_training_S = ROOT.TH1D('Histo_training_S' , '%i x S (Train)'%normfact , 25 , xmin,xmax)
Histo_training_B = ROOT.TH1D('Histo_training_B' , 'B (Train)'               , 25 , xmin,xmax)
Histo_testing_S = ROOT.TH1D('Histo_testing_S'   , '%i x S (Test)'%normfact  , 25 , xmin,xmax)
Histo_testing_B = ROOT.TH1D('Histo_testing_B'   , 'B (Test)'                , 25 , xmin,xmax)
 
# Fetch the trees of events from the root file 
TrainTree = f_out.Get("MVA_weights/TrainTree") 
TestTree = f_out.Get("MVA_weights/TestTree") 
 
# Cutting on these objects in the trees will allow to separate true S/B SCut_Tree = 'classID>0.5'
BCut_Tree = 'classID<0.5'
SCut_Tree = 'classID>0.5'
 
TrainTree.Draw("BDT>>Histo_training_S",("%i*weight*("%normfact)+SCut_Tree+")")
TrainTree.Draw("BDT>>Histo_training_B","weight*("+BCut_Tree+")")
TestTree.Draw( "BDT>>Histo_testing_S",("%i*weight*("%normfact)+SCut_Tree+")")
TestTree.Draw( "BDT>>Histo_testing_B","weight*("+BCut_Tree+")")
 
# Create the color styles
Histo_training_S.SetLineColor(2)
Histo_training_S.SetMarkerColor(2)
Histo_training_S.SetFillColor(2)
Histo_testing_S.SetLineColor(2)
Histo_testing_S.SetMarkerColor(2)
Histo_testing_S.SetFillColor(2)
 
Histo_training_B.SetLineColor(4)
Histo_training_B.SetMarkerColor(4)
Histo_training_B.SetFillColor(4)
Histo_testing_B.SetLineColor(4)
Histo_testing_B.SetMarkerColor(4)
Histo_testing_B.SetFillColor(4)
 
# Histogram fill styles
Histo_training_S.SetFillStyle(4501)
Histo_training_B.SetFillStyle(4501)
Histo_training_S.SetFillColorAlpha(Histo_training_S.GetLineColor(),0.2)
Histo_training_B.SetFillColorAlpha(Histo_training_B.GetLineColor(),0.2)
Histo_testing_S.SetFillStyle(0)
Histo_testing_B.SetFillStyle(0)
 
# Histogram marker styles
Histo_testing_S.SetMarkerStyle(20)
Histo_testing_B.SetMarkerStyle(20)
Histo_testing_S.SetMarkerSize(0.7)
Histo_testing_B.SetMarkerSize(0.7)
 
# Set titles
Histo_training_S.GetXaxis().SetTitle("Discriminant")
Histo_training_S.GetYaxis().SetTitle("Counts/Bin")
 
# Draw the objects
# c1 = ROOT.TCanvas("c1","",800,600)
c1 = ROOT.TCanvas("c1","",400,400)
p1 = ROOT.TPad("p1","p1",0., 0.23, 1.0, 1.0)
p2 = ROOT.TPad("p2","p2",0., 0.0, 1.0, 0.23)
p2.Draw()
# p2.cd()
p1.Draw()
# p1.SetLogy(1)
p1.SetLogy(0)
p1.cd()
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
Histo_training_S.Draw("HIST")
Histo_training_B.Draw("HISTSAME")
Histo_testing_S.Draw("EPSAME")
Histo_testing_B.Draw("EPSAME")

# Reset the y-max of the plot
ymax = max([h.GetMaximum() for h in [Histo_training_S,Histo_training_B,Histo_testing_S,Histo_testing_B] ])
ymax *=1.4
Histo_training_S.SetMaximum(ymax)
Histo_training_S.SetMinimum(0.01)
 
# Create Legend
c1.cd(1).BuildLegend( 0.42+0.3,  0.72,  0.57+0.3,  0.88).SetFillColor(0)
 

auc = -1.

# make soverb pad
soverb_cumulative = Histo_testing_S.Clone("soverb_cumulative")
soverb_cumulative.GetYaxis().SetRangeUser(0.01,9.)
soverb_cumulative.GetYaxis().SetNdivisions(505)
soverb_cumulative.GetYaxis().SetTitleSize(0.11)
soverb_cumulative.GetYaxis().SetTitleOffset(0.31)
soverb_cumulative.GetYaxis().SetLabelSize(0.13)
soverb_cumulative.GetYaxis().CenterTitle()
soverb_cumulative.GetXaxis().SetLabelSize(0.0)
soverb_cumulative.GetXaxis().SetTitle("")
soverb_cumulative.GetXaxis().SetTickSize(0.06)
soverb_cumulative.SetMarkerStyle(20)
soverb_cumulative.SetMarkerSize(0.7)    

p2.cd()
# arrs = np.array(list(Histo_training_S))/normfact
# arrb = np.array(list(Histo_training_B))
arrs = np.array(list(Histo_testing_S))/normfact
arrb = np.array(list(Histo_testing_B))
cumsum_s = np.cumsum(arrs[::-1])[::-1]
cumsum_b = np.cumsum(arrb[::-1])[::-1]
auc = abs(np.trapz(cumsum_s/np.sum(arrs),cumsum_b/np.sum(arrb)))
cumsum = (cumsum_s/cumsum_b)
cumsum = np.array([i if i==i else 0 for i in cumsum])

cumsum[np.abs(cumsum)>1e5] = 1000.


cumsum_sqrtsb = cumsum_s/np.sqrt(cumsum_s+cumsum_b)
cumsum_sqrtsb[np.abs(cumsum_sqrtsb)>1e5] = 1000.
# for ibz, val in enumerate(cumsum):
#     soverb_cumulative.SetBinContent(ibz,val)
#     soverb_cumulative.SetBinError(ibz,0.)
for ibz, val in enumerate(cumsum_sqrtsb):
    if np.isnan(val): continue
    soverb_cumulative.SetBinContent(ibz,val*10)
    soverb_cumulative.SetBinError(ibz,0.)
maxsb = max(list(soverb_cumulative))
# print "GREP", maxsb, auc, max_vars, ",".join(new_vars)
soverb_cumulative.SetLineColor(ROOT.kBlue-2)
soverb_cumulative.SetMarkerColor(ROOT.kBlue-2)
# soverb_cumulative.GetYaxis().SetTitle("Cumulative s/b")
soverb_cumulative.GetYaxis().SetTitle("Cumulative s/#sqrt{s+b}")
soverb_cumulative.GetYaxis().SetRangeUser(0.01,2.)
soverb_cumulative.Draw("samepe")

p1.cd()
l1=ROOT.TLatex()
l1.SetNDC();
l1.DrawLatex(0.26,0.93,"BDT [AUC = %.3f] [%.3f]" % (auc,maxsb))

l1.SetTextSize(l1.GetTextSize()*0.5)
l1.DrawLatex(0.76,0.63,"N^{sig}_{train} = %.1f" % (Histo_training_S.Integral()/normfact))
l1.DrawLatex(0.76,0.58,"N^{sig}_{test} = %.1f" % (Histo_testing_S.Integral()/normfact))
l1.DrawLatex(0.76,0.53,"N^{bg}_{train} = %.1f" % Histo_training_B.Integral())
l1.DrawLatex(0.76,0.48,"N^{bg}_{test} = %.1f" % Histo_testing_B.Integral())
 
pname = 'validation_bdt_{}.pdf'.format(outname.rsplit(".",1)[0].split("/")[-1])
c1.Print(pname)
