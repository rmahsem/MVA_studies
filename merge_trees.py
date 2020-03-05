import uproot
import numpy as np
import uproot_methods.classes.TH1

class SimpleNamespace (object):
    def __init__ (self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__ (self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    def __eq__ (self, other):
        return self.__dict__ == other.__dict__

class MyTH1(uproot_methods.classes.TH1.Methods, list):
    def __init__(self, low, high, values, title=""):
        self._fXaxis = SimpleNamespace()
        self._fXaxis._fNbins = len(values)
        self._fXaxis._fXmin = low
        self._fXaxis._fXmax = high
        for x in values:
            self.append(float(x))
        self._fTitle = title
        self._classname = "TH1F"


infile3top = uproot.open("ttt_multiYear_1126.root")
infileAll = uproot.open("all_met25_mvaTest_1213.root")
infileAll2 = uproot.open("full_MC_tree_only2lep_1126.root")

class upGet:
    def __init__(self, infile, innames, outname=None, xsec=1):
        self.outname = outname
        self.valid = list()
        if not isinstance(innames, list) :
            self.outname = innames
            innames = [innames]
        if not self.outname:
            raise Exception

        sumAll = 0
        sumPass = 0
        self.newWeight = list()
        for name in innames:
            sumAll += sum(infile[name]["sumweights"].values)

        
        for name in innames:
            dir = infile[name]
            i=1
            while("testTree;%i" %i in dir):
                self.valid.append(dir["testTree;%i"%i])
                i += 1
                sumPass += np.sum(self.valid[-1]["weight"].array())
                scale = xsec/sum(dir["sumweights"].values)*np.sum(dir["sumweights"].values)/sumAll
                self.newWeight.append(self.valid[-1]["weight"].array()*scale)
                
        self.branches = self.valid[0].keys()
        self.sumweight = MyTH1(0, 2, [0, sumAll, sumPass, 0])
        
    def addData(self, inTree):
        bad = list()
        
        for i, tree in enumerate(self.valid):
            
            branchDict = {b:tree[b].array() for b in self.branches}
            branchDict["newWeight"] = self.newWeight[i]
            try:
                inTree[self.outname].extend(branchDict)
            except:
                bad.append(i)
                continue
        return bad

allTrees = list()
allTrees.append(upGet(infile3top, ["tttj2016", "tttj2017", "tttj2018"], "tttj", 0.000474))
allTrees.append(upGet(infile3top, ["tttw2016", "tttw2017", "tttw2018"], "tttw", 0.000788))
allTrees.append(upGet(infileAll2, "4top2016", xsec=0.0092))

allNames = [["ttw", 0.2043], ["ttz", 0.2529], ["tth2nonbb", 0.2151] , ["ttwh", 0.001582], ["ttzz", 0.001982], ["ttzh", 0.001535], ["tthh", 0.000757], ["ttww", 0.01150], ["ttwz", 0.003884], ["www", 0.2086], ["wwz", 0.1651],  ["wzz", 0.5565],  ["zzz", 0.01398], ["zz4l_powheg", 1.256], ["wz3lnu_mg5amcnlo", 4.4297], ["ww_doubleScatter", 0.16975],  ["wpwpjj_ewk", 0.03711],  ["vh2nonbb", 0.9561], ["ttg_dilep", 0.632],  ["wwg", 0.2147],  ["wzg", 0.04123],  ["zg", 123.9], ["ttg_lepfromTbar", 0.769], ["ttg_lepfromT", 0.77], ["ggh2zz", 0.01181], ["wg", 405.271], ["tzq", 0.0758], ["st_twll", 0.01123], ["DYm50", 6020.85], ["ttbar", 831.762], ["DYm10-50", 18610], ["wjets", 61334.9],]
#, 

for name in allNames:
    allTrees.append(upGet(infileAll, name[0], xsec=name[1]))

# exit()
branches = allTrees[0].branches
branchDict = {i:"float32" for i in branches}
branchDict["newWeight"] = "float32"


with uproot.recreate("inputTrees.root") as f:
    for tree in allTrees:
        print(tree.outname)
        f["sumweight_%s"%tree.outname] = tree.sumweight
        f[tree.outname] = uproot.newtree(branchDict)
        bad = tree.addData(f)
        for i in bad:
            bDict = {b:tree.valid[i][b].array() for b in branches}
            bDict["newWeight"] = tree.newWeight[i]
            f[tree.outname].extend(bDict)
        
        


 
