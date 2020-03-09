# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import uproot

saveDir = "."
doShow = False

def makeROC(roc_in, name):
    global saveDir
    pred_all,truth_all,wgt_all = list(), list(), list()
    for item in roc_in:
        for pred, frame in zip(item.pred, item.f):
            pred_all.extend(pred)
            truth_all.extend(frame["isSignal"])
            wgt_all.extend(np.abs(frame["weight"]))
            
    fpr, tpr,_ = roc_curve(truth_all, pred_all, sample_weight=wgt_all)
    auc_test = roc_auc_score(truth_all, pred_all, sample_weight=wgt_all)

    plt.plot(fpr,tpr, label="test AUC = {:.3f}".format(auc_test))
    plt.plot(np.linspace(0,1,5), np.linspace(0,1,5), linestyle=':')
    plt.legend()
    plt.xlabel("False Positive Rate", horizontalalignment='right', x=1.0)
    plt.ylabel("True Positive Rate", horizontalalignment='right', y=1.0)
    plt.savefig("%s/roc_curve_%s.png" % (saveDir, name))
    if doShow: plt.show()
    plt.clf()
    plt.cla()

def approxLikelihood(sig_hist, bkgd_hist):
    term1 = 0
    term2 = 0
    for sig, bkgd in zip(sig_hist, bkgd_hist):
        if bkgd <= 0 or sig <= 0: continue
        term1 += (sig+bkgd)*math.log(1+sig/bkgd)
        term2 += sig
    return math.sqrt(2*(term1 - term2))

def StoB(sig, back, bins, name, noSB=False):
    global doShow
    nSig = [np.sum(sig[i:]) for i in range(len(bins))]
    nBack = [np.sum(back[i:]) for i in range(len(bins))]
    StoB  = [s/math.sqrt(b) if b > 0 else 0 for s, b in zip(nSig, nBack)]
    StoSB = [s/math.sqrt(s+b) if b+s > 0 else 0 for s, b in zip(nSig, nBack)]

    StoBmb = bins[StoB.index(max(StoB))]
    StoSBmb = bins[StoSB.index(max(StoSB))]

    if not noSB:
        p = plt.plot(bins, StoB, label="$S/\sqrt{B}=%.3f$\n cut=%.2f"%(max(StoB), StoBmb))
        plt.plot(np.linspace(bins[0],bins[-1],5), [max(StoB)]*5, linestyle=':', color=p[-1].get_color())
    p = plt.plot(bins, StoSB, label="$S/\sqrt{S+B}=%.3f$\n cut=%.2f"%(max(StoSB), StoSBmb))
    plt.plot(np.linspace(bins[0],bins[-1],5), [max(StoSB)]*5, linestyle=':', color=p[-1].get_color())
    plt.legend()
    plt.xlabel("BDT value", horizontalalignment='right', x=1.0)
    plt.ylabel("A.U.", horizontalalignment='right', y=1.0)
    plt.savefig("%s/StoB_%s.png" % (saveDir, name))
    if doShow: plt.show()
    plt.close()
    

def createPlot(p1, p2, name, bins):
    global doShow
    global saveDir
    s = max(p2.getHist(bins))
    b = max(p1.getHist(bins))
    sc = findScale(s,b)
    
    p1.makePlot(isDen=False)
    if sc != 1:
        p2.makePlot(sc=sc, exLabel=u' × %d'%sc, isDen=False)
    else:
        p2.makePlot(isDen=False)
    plt.legend()
    plt.xlabel("BDT value", horizontalalignment='right', x=1.0)
    plt.ylabel("Events / Bin", horizontalalignment='right', y=1.0)
    #plt.yscale('log')
    plt.savefig("%s/%s.png"%(saveDir, name))
    if doShow: plt.show()
    plt.clf()
    plt.cla()
    p1.makePlot(True)
    p2.makePlot(True)
    plt.legend()
    plt.xlabel("BDT value", horizontalalignment='right', x=1.0)
    plt.ylabel("A.U.", horizontalalignment='right', y=1.0)
    #plt.yscale('log')
    plt.savefig("%s/%s_shape.png"%(saveDir, name))
    if doShow: plt.show()
    plt.close()

    
from math import log10
def findScale(s, b):
    scale = 1
    prevS = 1
    while b//(scale*s) != 0:
        prevS = scale
        if int(log10(scale)) == log10(scale):
            scale *= 5
        else:
            scale *= 2
    return prevS

def plotRocTMVA(outname):
    global doshow
    global saveDir
    hist = uproot.open("{}/BDT.root".format(outname))["MVA_weights/Method_BDT/BDT"]["MVA_BDT_rejBvsS"]
    fig, ax = plt.subplots()
    
    y = hist.edges
    x = 1-np.concatenate((hist.values, [0]))
    ax.plot(x, y, label="test AUC = {:.3f}".format(sum(hist.values)/100))
    
    ax.plot(np.linspace(0,1,5), np.linspace(0,1,5), linestyle=':')
    ax.legend()
    ax.set_xlabel("False Positive Rate", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True Positive Rate", horizontalalignment='right', y=1.0)
    fig.tight_layout()
    if doShow: plt.show()
    plt.savefig("%s/roc_curve.png" % (saveDir))
    plt.close()
    return

def getWeightTMVA(df, lumi, fc):
    return lumi*df["newWeight"]*fc


def plotFuncTMVA(sig, bkg, lumi, name, bins, scale=True):
    global doShow
    global saveDir
    fig, ax = plt.subplots()
    bkgHist = ax.hist(x=bkg[name], bins=bins, weights=bkg["finalWeight"],label="Background", histtype="step", linewidth=1.5)
    if scale:
        sigMax = np.max(np.histogram(sig[name], bins=bins, weights=sig["finalWeight"])[0])
        
        scaleFac = findScale(sigMax, max(bkgHist[0]))
        sigHist = ax.hist(x=sig[name], bins=bins, weights=sig["finalWeight"]*scaleFac, label="Signal x {}".format(scaleFac), histtype="step",linewidth=1.5)
    else:
        sigHist = ax.hist(x=sig[name], bins=bins, weights=sig["finalWeight"], label="Signal", histtype="step",linewidth=1.5)
    ax.legend()
    ax.set_xlabel(name)
    ax.set_ylabel("Events/bin")
    ax.set_title("Lumi = {} ifb".format(lumi/1000))
    fig.tight_layout()
    if doShow: plt.show()
    plt.savefig("%s/%s.png" % (saveDir, name))
    plt.close()
