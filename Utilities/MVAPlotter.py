import uproot,pandas
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, roc_auc_score

class MVAPlotter(object):
    def __init__(self, workDir, groups, lumi=140000):
        self.groups = list(groups)
        self.testGroup = list()
        self.trainGroup = list()
        self.doShow = False
        self.saveDir = workDir
        self.lumi=lumi
        
        prefix=""
        with uproot.open("{}/BDT.root".format(workDir)) as outfile:
            if "MVA_weights" in outfile:
                prefix = "MVA_weights/"
            self.trainSet = outfile["{}TrainTree".format(prefix)].pandas.df(["*"])
            self.testSet = outfile["{}TestTree".format(prefix)].pandas.df(["*"])
                        
        train2test = 1.0*len(self.trainSet)/len(self.testSet)
        self.testSet.insert(0, "finalWeight", self.testSet["newWeight"]*(1+train2test))
        self.trainSet.insert(0, "finalWeight", self.trainSet["newWeight"]*(1+1/train2test))
        self.trainSet["finalWeight"] *= lumi
        self.testSet["finalWeight"] *= lumi
        
        renameDict = {group : "BDT.{}".format(group) for group in self.groups}
        self.trainSet = self.trainSet.rename(columns=renameDict)
        self.testSet = self.testSet.rename(columns=renameDict)
                
        for idx, group in enumerate(self.groups):
            self.testGroup.append(self.testSet[self.testSet["classID"] == idx])
            self.trainGroup.append(self.trainSet[self.trainSet["classID"] == idx])

            
    def setupROC(self, mainGroup, otherGroups, isTrain=False):
        finalSet = self.getSample([mainGroup]+otherGroups, isTrain)
        
        pred = finalSet["BDT.{}".format(mainGroup)].array
        truth = [1 if i == self.groups.index(mainGroup) else 0 for i in finalSet["classID"].array]
        return pred, truth

    def setDoShow(self, doShow):
        self.doShow = doShow

    
    def getSample(self, groups, isTrain=False):
        workSet = self.testGroup if not isTrain else self.trainGroup
        finalSet = pandas.DataFrame()
        for group in groups:
            if finalSet.empty:
                finalSet = workSet[self.groups.index(group)]
            else:
                finalSet = pandas.concat((finalSet, workSet[self.groups.index(group)]))
                
        return finalSet

    def getVariable(self, groups, var, isTrain=False):
        return self.getSample(groups, isTrain)[var]
        
    def getHist(self, groups, var, bins, isTrain=False):
        finalSet = self.getSample(groups, isTrain)
        return np.histogram(finalSet[var], bins=bins, weights=finalSet["finalWeight"])[0]

    def plotFunc(self, sig, bkg, var, bins, name, scale=True):
        sigHist = self.getHist([sig], var, bins)
        bkgHist = self.getHist(bkg, var, bins)
        scaleFac = findScale(max(sigHist), max(bkgHist)) if scale else 1.
        bkgName = "all" if len(bkg) > 1 else bkg[0]
        sigName = sig if scaleFac == 1 else "{} x {}".format(sig, scaleFac)

        fig, ax = plt.subplots()

        ax.hist(x=bins[:-1], weights=sigHist*scaleFac, bins=bins, label=sigName, histtype="step", linewidth=1.5)
        ax.hist(x=bins[:-1], weights=bkgHist, bins=bins, label=bkgName, histtype="step", linewidth=1.5)
        ax.legend()
        ax.set_xlabel(var)
        ax.set_ylabel("Events/bin")
        ax.set_title("Lumi = {} ifb".format(self.lumi/1000.))
        fig.tight_layout()
        plt.savefig("%s/%s%s.png" % (self.saveDir, var, name))
        if self.doShow: plt.show()
        plt.close()

    def plotStoB(self, sig, bkg, var, bins, name, noSB=False):
        nSig = [np.sum(self.getHist([sig], var, bins)[i:]) for i in range(len(bins))]
        nBack = [np.sum(self.getHist(bkg, var, bins)[i:]) for i in range(len(bins))]
        StoB  = [s/math.sqrt(b) if b > 0 else 0 for s, b in zip(nSig, nBack)]
        StoSB = [s/math.sqrt(s+b) if b+s > 0 else 0 for s, b in zip(nSig, nBack)]
        
        StoBmb = bins[StoB.index(max(StoB))]
        StoSBmb = bins[StoSB.index(max(StoSB))]

        fig, ax = plt.subplots()

        if not noSB:
            p = ax.plot(bins, StoB, label="$S/\sqrt{B}=%.3f$\n cut=%.2f"%(max(StoB), StoBmb))
            ax.plot(np.linspace(bins[0],bins[-1],5), [max(StoB)]*5, linestyle=':', color=p[-1].get_color())
        p = ax.plot(bins, StoSB, label="$S/\sqrt{S+B}=%.3f$\n cut=%.2f"%(max(StoSB), StoSBmb))
        ax.plot(np.linspace(bins[0],bins[-1],5), [max(StoSB)]*5, linestyle=':', color=p[-1].get_color())
        ax.legend()
        ax.set_xlabel("BDT value", horizontalalignment='right', x=1.0)
        ax.set_ylabel("A.U.", horizontalalignment='right', y=1.0)
        fig.tight_layout()
        plt.savefig("%s/StoB_%s.png" % (self.saveDir, name))
        if self.doShow: plt.show()
        plt.close()

    def approxLikelihood(self, sig, bkg, var, bins):
        sigHist = self.getHist([sig], var, bins)
        bkgHist = self.getHist(bkg, var, bins)
        term1 = 0
        term2 = 0
        for sigVal, bkgVal in zip(sigHist, bkgHist):
            if bkgVal <= 0 or sigVal <= 0: continue
            term1 += (sigVal+bkgVal)*math.log(1+sigVal/bkgVal)
            term2 += sigVal
        return math.sqrt(2*(term1 - term2))


    def makeROC(self, sig, bkg, name):
        finalSet = self.getSample([sig]+bkg)
        
        pred = finalSet["BDT.{}".format(sig)].array
        truth = [1 if i == self.groups.index(sig) else 0 for i in finalSet["classID"].array]
        if name: name = "_{}".format(name)
                
        fpr, tpr,_ = roc_curve(truth, pred)#, sample_weight=wgt)
        auc = roc_auc_score(truth, pred)#, sample_weight=wgt)

        fig, ax = plt.subplots()
        ax.plot(fpr,tpr, label="AUC = {:.3f}".format(auc))
        ax.plot(np.linspace(0,1,5), np.linspace(0,1,5), linestyle=':')
        ax.legend()
        ax.set_xlabel("False Positive Rate", horizontalalignment='right', x=1.0)
        ax.set_ylabel("True Positive Rate", horizontalalignment='right', y=1.0)
        fig.tight_layout()
        plt.savefig("{}/roc_curve{}.png".format(self.saveDir, name))
        if self.doShow: plt.show()
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


