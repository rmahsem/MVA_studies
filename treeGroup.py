import pandas
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

class treeGroup:
    bins=np.linspace(0,1,11)
    def __init__(self, infile, label, xsec, isSig=False):
        self.f = list()
        removeList = list()
        for dirN in label:
            dataF = self.getFrameYear(infile, dirN, isSig)
            if dataF.empty:
                removeList.append(dirN)
            else:
                self.f.append(dataF)
        for dirN in removeList:
            label.remove(dirN)
        self.xsec = xsec
        self.label = label
        self.sumweight = [self.getSumWeight(infile, dirN) for dirN in label]
        self.extraWeight = np.ones(len(self.f))
        for frame,xs,sw in zip(self.f, self.xsec, self.sumweight):
            frame["weight"] *= xs/sw
            
        self.histOpt = {"stacked":True, "alpha":1.0, "edgecolor":'black', "histtype":'stepfilled'}
        self.madeWeight = False
        
    def __add__(self, other):
        self.f.extend(other.f)
        self.xsec.extend(other.xsec)
        self.label.extend(other.label)
        self.sumweight.extend(other.sumweight)
        self.extraWeight = np.concatenate([self.extraWeight, other.extraWeight])
        self.pred.extend(other.pred)
        self.madeWeight = False
        return self

    def __len__(self):
        return len(self.f)

    def __getitem__(self, index):
        return self.f[index]

    def makePred(self, func):
        self.pred = [func.predict(xgb.DMatrix(frame.drop(["isSignal","weight"], axis=1))) for frame in self.f]

    def makeWgt(self, lumi=140000):
        self.wgt = [mframe["weight"].values*mExtraWgt*lumi for mframe, mExtraWgt in zip(self.f, self.extraWeight)]
        self.madeWeight = True
        # for name, wgt in zip(self.label, self.wgt):
        #     print(name, sum(wgt))

    def getSumWeight(self, infile, dirName):
        if dirName in infile:
            return infile[dirName]["sumweights"].values.sum()
        sumTot = 0
        for year in ["2016", "2017", "2018"]:
            if "%s%s" % (dirName, year) in infile:
                sumTot += infile["%s%s" % (dirName, year)]["sumweights"].values.sum()
        return sumTot

            
    def getFrameYear(self, infile, dirName, signal):
        frame = self.getFrame(infile, dirName, signal)
        if not frame.empty:
            return frame
        frameList = list()
        for year in ["2016", "2017", "2018"]:
            frameTmp = self.getFrame(infile, "%s%s" % (dirName, year), signal)
            if not frameTmp.empty: frameList.append(frameTmp)
        if not frameList:
            return pandas.DataFrame()

        frame = pandas.concat(frameList, ignore_index=True)
        if not frame.empty:
            return frame
        else:
            print( "PROBLEM: %s" % dirName)
            exit()
    
    def getFrame(self, infile, dirName, signal):
        global bNames
        if dirName not in infile:
            return pandas.DataFrame()
        dir = infile[dirName]
        i=1
        valid = list()
        while("testTree;%i"%i in dir):
            valid.append(dir["testTree;%i"%i])
            i += 1
        frames = [tree.pandas.df(bNames) for tree in valid]
        frame = pandas.concat(frames, ignore_index=True)
        if signal:
            frame["isSignal"] = 1
        else:
            frame["isSignal"] = 0
            #Cuts
        frame = frame[frame["HT"] > 150]
        # frame = frame[frame["NJets"] >= 2]
        frxame = frame[frame["MET"] > 25]
        frame = frame[frame["DilepCharge"] > 0]
        # frame = frame[frame["DilepCharge"] < 0]
        frame = frame.reset_index(drop=True)
        return frame

    def makeTest(self, num, idx=0):
        tot = 1.0*len(self.f[idx])
        self.extraWeight[idx] *= tot
        train_return = self.f[idx].truncate(after=num-1)
        train_return["weight"] *= tot/len(train_return)
        self.f[idx] = self.f[idx].truncate(before=num)
        self.extraWeight[idx] *= 1./len(self.f[idx])
        print( "Train %s with %d" % (self.label[idx], num))
        return train_return

    def makePlot(self, sc=-1, exLabel="", isDen=True):
        if not self.madeWeight:  self.makeWgt(lumi=140000)
        wgt = self.wgt
        if sc >0:
            wgt = [inWgt*sc for inWgt in wgt]
        lab = [name+exLabel for name in self.label]
        return  plt.hist(self.pred, treeGroup.bins, weights=wgt, label=lab, density=isDen, **self.histOpt)
        
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

bNames = ["NJets", "NBJets", "HT", "MET", "l1Pt", "l2Pt", "lepMass", "centrality", \
          "j1Pt", "j2Pt", "j3Pt", "j4Pt", "j5Pt", "j6Pt", "j7Pt", "j8Pt", "jetMass", "jetDR", \
          "b1Pt", "b2Pt", "b3Pt", "b4Pt", "Shape1", "Shape2", "DilepCharge", "NlooseBJets", "NtightBJets", "NlooseLeps", "weight",]
