import pandas, uproot, ROOT
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate

class MvaMaker(object):
    def __init__(self, infileName, outname):
        self.outname = outname
        self.infileName = infileName
        

class TMVAMaker(MvaMaker):
    def __init__(self, *args):
        super(TMVAMaker, self).__init__(*args)
        # analysistype = 'AnalysisType=Classification'                     
        analysistype = 'AnalysisType=MultiClass' #For many backgrounds
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
        
    def addGroup(self, inNames, outName):
        isSignal = True if outName=="Signal" else False

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
        self.dataset.PrepareTrainingAndTestTree(self.rootCut, ":".join(["SplitMode=Random:NormMode=EqualNumEvents",]))
        # "CrossEntropy", "GiniIndex", "GiniIndexWithLaplace", "MisClassificationError", "SDivSqrtSPlusB"
        sepType="CrossEntropy"

        #methodInput =["!H", "NTrees=500", "nEventsMin=150", "MaxDepth=5", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType={}".format(sepType),"nCuts=20","PruneMethod=NoPruning","IgnoreNegWeightsInTraining",]
        methodInput = ["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType=SDivSqrtSPlusB","nCuts=50","Pray"]

        method = self.factory.BookMethod(self.dataset, ROOT.TMVA.Types.kBDT, "BDT", ":".join(methodInput))
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
        self.groupNames = ["Signal"]
        self.infile = uproot.open(self.infileName)
        self.groupTotal = list()
        
    def addVariables(self, trainVars, specVars):
        self.trainVars = trainVars
        self.specVars = specVars
        self.allVars = ["classID"] + self.trainVars + self.specVars
        self.trainSet = pandas.DataFrame(columns=self.allVars)
        self.testSet = pandas.DataFrame(columns=self.allVars[1:])
        
    def addCut(self, cut):
        self.cut = cut.split("&&")

    def addGroup(self, inNames, outName):
        isSignal = True if outName=="Signal" else False
        if not isSignal:
           self.groupNames.append(outName) 

        totalSW = 0
        totalEv = 0
        for name in inNames:
            df = self.infile[name].pandas.df(self.allVars[1:])
            df = self.cutFrame(df)
            totalSW += np.sum(np.abs(df["newWeight"]))
            totalEv += len(df)
        scale = 1.*totalEv/totalSW
        for name in inNames:
            df = self.infile[name].pandas.df(self.allVars[1:])
            df = self.cutFrame(df)
            if isSignal:
                df["classID"] = 0
            else:
                df["classID"] = len(self.groupNames)-1
            df.insert(0, "finalWeight", np.abs(df["newWeight"])*scale)
            
            train, test = train_test_split(df, test_size=self.splitRatio, random_state=12345)
            print("Add Tree {} of type {} with {} event".format(name, outName, len(train)))
            self.trainSet = pandas.concat([train.reset_index(drop=True), self.trainSet], sort=True)
            self.testSet = pandas.concat([test.reset_index(drop=True), self.testSet], sort=True)
        self.groupTotal.append(1.*len(self.trainSet[self.trainSet["classID"] == len(self.groupNames)-1]))
        
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
        X_train = self.trainSet.drop(self.specVars+["classID", "finalWeight"], axis=1)
        y_train = self.trainSet["classID"]
        w_train = self.trainSet["finalWeight"].copy()
        X_test = self.testSet.drop(self.specVars+["classID", "finalWeight"], axis=1)
        y_test = self.testSet["classID"]

        sigImpFac = 1
        w_train[self.trainSet["classID"] == 0] *= sigImpFac
        for i, groupTot in enumerate(self.groupTotal):
            w_train[self.trainSet["classID"] == i] *= min(self.groupTotal)/groupTot
            print i, groupTot, sum(w_train[self.trainSet["classID"] == i])
        
        # XGBoost training
        param = {}
        param['objective'] = 'multi:softmax'
        param['booster'] = 'dart'
        param['eta'] = 0.09
        param['max_depth'] = 5
        param['silent'] = 1
        param['nthread'] = 3
        param['eval_metric'] = "mlogloss"
        param['subsample'] = 0.9
        param['colsample_bytree'] = 0.5
        param['num_class'] = len(np.unique(y_train))
        num_rounds = 150

        
        # dtrain = xgboost.DMatrix(X_train, y_train)
        # dtest = xgboost.DMatrix(X_test, y_test)
        # fitModel =  xgboost.train(param, dtrain, num_rounds, [(dtrain, "train"), (dtest, "test")])
        fitModel = xgb.XGBClassifier(**param)
        fitModel.fit(X_train, y_train, w_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=100)
        

        with uproot.recreate("{}/BDT.root".format(self.outname)) as outfile:
            self.writeOut(outfile, "TestTree", self.testSet, fitModel.predict_proba(X_test).T)
            self.writeOut(outfile, "TrainTree", self.trainSet, fitModel.predict_proba(X_train).T)
        valid_result = fitModel.evals_result()
        best_n_trees = fitModel.best_ntree_limit
                        
        fitModel.save_model("{}/model.bin".format(self.outname))

    
        
    def writeOut(self, outfile, treeName, workSet, prediction):
        outDict = {name: workSet[name] for name in self.allVars}
        for i, name in enumerate(self.groupNames):
            outDict[name] = prediction[i]

        outfile[treeName] = uproot.newtree({name:"float32" for name in self.allVars+self.groupNames})    
        outfile[treeName].extend(outDict)

        
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
