import uproot,pandas, numpy
import matplotlib.pyplot as plt


class MVAPlotter(object):
    def __init__(self, infile, groups, lumi=140000):
        self.groups = list(groups)
        self.testGroup = list()
        self.trainGroup = list()

        prefix=""
        with uproot.open("{}/BDT.root".format(infile)) as outfile:
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
        return numpy.histogram(finalSet[var], bins=bins, weights=finalSet["finalWeight"])[0]


