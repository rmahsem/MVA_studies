import pandas, uproot, ROOT
from sklearn.model_selection import train_test_split

class MvaMaker(object):
    def __init__(self, infileName, outname):
        self.outname = outname
        self.infileName = infileName
        
        

class TMVAMaker(MvaMaker):
    def __init__(self, *args):
        super(TMVAMaker, self).__init__(*args)
        analysistype = 'AnalysisType=Classification'                     
        # analysistype = 'AnalysisType=MultiClass' #For many backgrounds
        factoryOptions = [ "!Silent", "Color", "DrawProgressBar", "Transformations=I", analysistype]
        ROOT.TMVA.Tools.Instance()
        self.infile = ROOT.TFile(self.infileName)
        self.fout = ROOT.TFile("{}/BDT.root".format(self.outname),"RECREATE")
        self.factory = ROOT.TMVA.Factory("TMVAClassification", self.fout,":".join(factoryOptions))
        self.dataset = ROOT.TMVA.DataLoader('MVA_weights')

    def addVariables(self, trainVars, specVars):
        for var in trainVars:
            self.dataset.AddVariable(var)
        for var in specVars:
            self.dataset.AddSpectator(var)
        
    def addCut(self, cut):
        self.cut = cut
        self.rootCut = ROOT.TCut(cut)
        
    def addGroup(self, inNames, outName, isSignal=False):
        sumW = dict()
        for name in inNames:
            tree = self.infile.Get(name)
            tree.Draw("(newWeight)>>tmp_{}".format(name), self.cut)
            tree.Draw("(genWeight)>>tmp2_{}".format(name), self.cut)
            eventRate = ROOT.gDirectory.Get("tmp_{}".format(name)).GetMean()*tree.GetEntries(self.cut)
            genVsAll = 1/ROOT.gDirectory.Get("tmp2_{}".format(name)).GetMean()
            sumW[name] = (eventRate, genVsAll)
        for name in inNames:
            tree = self.infile.Get(name)
            fac = sumW[name][0]/sum([i for i, j in sumW.values()])
            #if useNeg:  fac *= bkgSum[name][1]
            if isSignal:
                self.dataset.AddSignalTree(tree, fac/tree.GetEntries(self.cut))
            else:
                self.dataset.AddTree(tree, outName, fac/tree.GetEntries(self.cut))
            
    def train(self):
        self.dataset.PrepareTrainingAndTestTree(rootCut, ":".join(["SplitMode=Random:NormMode=EqualNumEvents",]))
        # "CrossEntropy", "GiniIndex", "GiniIndexWithLaplace", "MisClassificationError", "SDivSqrtSPlusB"
        sepType="CrossEntropy"

        #methodInput =["!H", "NTrees=500", "nEventsMin=150", "MaxDepth=5", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType={}".format(sepType),"nCuts=20","PruneMethod=NoPruning","IgnoreNegWeightsInTraining",]
        methodInput = ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType=SDivSqrtSPlusB","nCuts=50","Pray"]

        method = self.factory.BookMethod(dataset, ROOT.TMVA.Types.kBDT, "BDT", ":".join(methodInput))
        self.factory.TrainAllMethods() 
        self.factory.TestAllMethods() 
        self.factory.EvaluateAllMethods() 


            
class XGBoostMaker(MvaMaker):
    def __init__(self, *args):
        super(XGBoostMaker, self).__init__(*args)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.splitRatio = 0.5
        self.infile = uproot.open(self.infileName)
        
    def addVariables(self, trainVars, specVars):
        self.trainVars = trainVars
        self.specVars = specVars
        self.X_train = pandas.DataFrame(self.trainVars+self.specVars+["isSignal"])
        self.X_test = pandas.DataFrame(self.trainVars+self.specVars+["isSignal"])
        self.y_train = pandas.DataFrame(self.trainVars+self.specVars+["isSignal"])
        self.y_test = pandas.DataFrame(self.trainVars+self.specVars+["isSignal"])
        
    def addCut(self, cut):
        self.cut = cut.split("&&")

    def addGroup(self, inNames, outName, isSignal=False):
        for name in inNames:
            df = self.infile[name].pandas.df(self.trainVars+self.specVars)
            df = self.cutFrame(df)
            if isSignal:    df["isSignal"] = 1
            else:           df["isSignal"] = 0

            X_train, X_test, y_train, y_test= train_test_split(df[self.trainVars], df["isSignal"], test_size=self.splitRatio, random_state=12345)
            self.X_train = pandas.concat([X_train, self.X_train], sort=True)
            self.X_test = pandas.concat([X_test, self.X_test], sort=True)
            self.y_train = pandas.concat([y_train, self.y_train], sort=True)
            self.y_test = pandas.concat([y_test, self.y_test], sort=True)

    def cutFrame(self, frame):
        for cut in self.cut:
            if cut.find("<") != -1:
                tmp = cut.split("<")
                frame = frame[frame[tmp[0]] < float(tmp[1])]
            elif cut.find(">") != -1:
                tmp = cut.split(">")
                frame = frame[frame[tmp[0]] > float(tmp[1])]
            elif cut.find("==") != -1:
                tmp = cut.split("==")
                frame = frame[frame[tmp[0]] == float(tmp[1])]
        return frame
        
    def train(self):
        X_train = pandas.concat(split_train,ignore_index=True).drop(TreeGroup.exclude, axis=1)
        y_train = pandas.concat(split_train, ignore_index=True)["isSignal"]

        
        # XGBoost training
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)

        evallist  = [(dtrain,'train')]
        num_round=200
        param = {}
        param['objective'] = 'binary:logistic'
        param['eta'] = 0.09
        param['max_depth'] = 5
        param['silent'] = 1
        param['nthread'] = 2
        param['eval_metric'] = "auc"
        param['subsample'] = 0.9
        param['colsample_bytree'] = 0.5
        fitModel = xgb.train(param.items(), dtrain, num_round, evallist,early_stopping_rounds=100, verbose_eval=100 )

        fitModel.save_model("{}/model.bin".format(args.infile))





