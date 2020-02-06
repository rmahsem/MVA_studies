import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import helper

from samples import *

tops_mod = xgb.Booster({'nthread': 4})
tops_mod.load_model('model_34.bin')

alls_mod = xgb.Booster({'nthread': 4})
alls_mod.load_model('model_3all.bin')

name = []
pred_all = []
pred_top = []
pred_wgt = []
for frame in all_g:
    name.append(frame.label)
    frame.makePred(alls_mod)
    pred_all.append(frame.pred)
    frame.makePred(tops_mod)
    pred_top.append(frame.pred)
    frame.makeWgt()
    pred_wgt.append(frame.wgt)

hists = []
blah = True
scale = 30
for f_name, f_all, f_top, f_wgt in zip(name, pred_all, pred_top, pred_wgt):
    print(f_name)
    if blah:
        blah = False
        print(len(f_wgt))
        f_wgt = [scale*w for w in f_wgt]
        print(len(f_wgt))
    
    h, b1, b2, c = plt.hist2d(np.concatenate(f_all), np.concatenate(f_top), weights=np.concatenate(f_wgt))
    hists.append(h)
    # plt.show()

sig = hists[0]
bkg = np.sum(hists[1:], axis=0)

sig0 = np.sum(sig, axis=0)
bkg0 = np.sum(bkg, axis=0)
sig1 = np.sum(sig, axis=1)
bkg1 = np.sum(bkg, axis=1)

print(helper.approxLikelihood(sig.flatten(), bkg.flatten()))
print(helper.approxLikelihood(sig0, bkg0))
print(helper.approxLikelihood(sig1, bkg1))
