import uproot
from treeGroup import treeGroup
import numpy as np


bins = np.linspace(0,1,41)
treeGroup.bins = bins

infile3top = uproot.open("ttt_multiYear_1126.root")
infileAll = uproot.open("all_met25_mvaTest_1213.root")
infileAll2 = uproot.open("full_MC_tree_only2lep_1126.root")

top3SF = 1.0
top4_g = treeGroup(infileAll2, ["4top"], [0.0092], False)
top3_g = treeGroup(infile3top, ["tttj", "tttw"], [top3SF*0.000474, top3SF*0.000788], True)

signal_g = top3_g
signal_g.histOpt["alpha"] = 0.5
other_g = top4_g

ttv_g = treeGroup(infileAll,
                  ["ttw", "ttz", "tth2nonbb"],
                  [0.2043, 0.2529, 0.2151])
ttvv_g = treeGroup(infileAll,
                  ["ttwh", "ttwz", "ttww", "tthh", "ttzh", "ttzz"],
                  [0.001582, 0.003884, 0.01150, 0.000757, 0.001535, 0.001982])
vvv_g = treeGroup(infileAll,
                   ["www", "wwz", "wzz", "zzz"],
                   [0.2086, 0.1651, 0.5565, 0.01398])
vv_g = treeGroup(infileAll,
                 ["zz4l_powheg", "wz3lnu_mg5amcnlo", "ww_doubleScatter", "wpwpjj_ewk", "vh2nonbb"],
                 [1.256, 4.4297, 0.16975, 0.03711, 0.9561])
xg_g = treeGroup(infileAll,
                 ["ttg_dilep", "wwg", "wzg", "zg", "ttg_lepfromTbar", "ttg_lepfromT", "ggh2zz", "wg"],
                 [0.632, 0.2147, 0.04123, 123.9, 0.769, 0.77, 0.01181, 405.271])
extra_g = treeGroup(infileAll,
                    ["tzq", "st_twll", "DYm50", "ttbar", "DYm10-50", "wjets"],
                    [0.0758, 0.01123, 6020.85, 831.762, 18610, 61334.9])

all_samples = [signal_g, other_g, ttv_g, ttvv_g, vvv_g, vv_g, xg_g, extra_g]
special_samples = [signal_g, other_g]
