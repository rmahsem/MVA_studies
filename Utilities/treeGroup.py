import pandas
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np


class TreeGroup:
    bins=np.linspace(0,1,11)
    bNames = []
    exclude = ["isSignal", "weight", "newWeight", "genWeight"]
    def __init__(self, infile, label, xsec, isSig=False):
        
        self.f = list()
        removeList = list()
        for dirN in label:
            dataF = self.getFrame(infile, dirN, isSig)
            if dataF.empty:
                removeList.append(dirN)
            else:
                self.f.append(dataF)
        for dirN in removeList:
            label.remove(dirN)
        self.xsec = xsec
        self.label = label
        self.testFact = np.ones(len(self.f))
        
        self.histOpt = {"stacked":True, "alpha":1.0, "edgecolor":'black', "histtype":'stepfilled'}
        self.madeWeight = False
        
    def __add__(self, other):
        self.f.extend(other.f)
        self.xsec.extend(other.xsec)
        self.label.extend(other.label)
        self.testFact = np.concatenate([self.testFact, other.testFact])
        self.pred.extend(other.pred)
        self.madeWeight = False
        return self

    def __len__(self):
        return len(self.f)

    def __getitem__(self, index):
        return self.f[index]

    def makePred(self, func):
        self.pred = [func.predict(xgb.DMatrix(frame.drop(TreeGroup.exclude, axis=1))) for frame in self.f]

    def makeWgt(self, lumi=140000):
        self.wgt = [mframe["newWeight"].values*mExtraWgt*lumi for mframe, mExtraWgt in zip(self.f, self.testFact)]
        self.madeWeight = True
        # for name, wgt in zip(self.label, self.wgt):
        #     print(name, sum(wgt))

    def getFrame(self, infile, dirName, signal):
        frame = infile[dirName].pandas.df(TreeGroup.bNames+TreeGroup.exclude[1:])
        if signal:
            frame["isSignal"] = 1
        else:
            frame["isSignal"] = 0
        #Cuts
        frame = frame[frame["HT"] > 150]
        # frame = frame[frame["NJets"] >= 2]
        frame = frame[frame["MET"] > 25]
        frame = frame[frame["DilepCharge"] > 0]
        # frame = frame[frame["DilepCharge"] < 0]
        frame = frame.reset_index(drop=True)
        return frame

    def makeTest(self, num, idx=0):
        tot = 1.0*len(self.f[idx])
        self.testFact[idx] *= tot
        train_return = self.f[idx].truncate(after=num-1)
        train_return["weight"] *= tot/len(train_return)
        self.f[idx] = self.f[idx].truncate(before=num)
        self.testFact[idx] *= 1./len(self.f[idx])
        print( "Train %s with %d" % (self.label[idx], num))
        return train_return

    def makePlot(self, sc=-1, exLabel="", isDen=True):
        if not self.madeWeight:  self.makeWgt(lumi=140000)
        wgt = self.wgt
        if sc >0:
            wgt = [inWgt*sc for inWgt in wgt]
        lab = [name+exLabel for name in self.label]
        return  plt.hist(self.pred, TreeGroup.bins, weights=wgt, label=lab, density=isDen, **self.histOpt)
        
    def getHist(self, bins):
        if not self.madeWeight: self.makeWgt(lumi=140000)
        hist = [0]
        for pred, wgt in zip(self.pred, self.wgt):
            hist = np.add(hist, np.histogram(pred, bins, weights=wgt)[0])
            
        return hist

    def getTotal(self):
        if not self.madeWeight: self.makeWgt(lumi=140000)
        total_arr = [np.sum(wgt) for wgt in self.wgt]
        return np.sum(total_arr)


