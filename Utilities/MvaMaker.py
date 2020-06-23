import uproot
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate

class XGBoostMaker:
    def __init__(self, infileName):
        self.filename = "BDT"
        self.splitRatio = 0.5
        self.groupNames = ["Signal"]
        self.infile = uproot.open(infileName)
        self.groupTotal = list()
        self.predTrain = dict()
        self.predTest = dict()

    def addVariables(self, trainVars, specVars):
        self.trainVars = trainVars
        self.specVars = specVars 
        self.extraVars = ["classID", "GroupName"]
        self.allVars = self.extraVars + self.trainVars + self.specVars
        self.trainSet = pd.DataFrame(columns=self.allVars)
        self.testSet = pd.DataFrame(columns=self.allVars)
        
    def addCut(self, cut):
        self.cut = cut.split("&&")

    def setFilename(self, name):
        self.filename = name
        
    def addGroup(self, inNames, outName):
        isSignal = True if outName=="Signal" else False
        if not isSignal:
           self.groupNames.append(outName) 

        # Get scale for group
        #
        # Scales each component of group by (# raw Events)/(# scaled Events)
        # This is done so each effective xsec is used as a ratio of the group
        # and the number of raw Events is so the average weight is 1 (what xgb wants)
        totalSW, totalEv = 0, 0
        for name in inNames:
            df = self.infile[name].pandas.df(self.trainVars+self.specVars)
            df = self.cutFrame(df)
            totalSW += np.sum(np.abs(df["newWeight"]))
            totalEv += len(df)
        scale = 1.*totalEv/totalSW
        
        for name in inNames:
            df = self.infile[name].pandas.df(self.trainVars+self.specVars)
            df = self.cutFrame(df)
            df["GroupName"] = name
            if isSignal:   df["classID"] = 0
            else:          df["classID"] = len(self.groupNames)-1
            
            df.insert(0, "finalWeight", np.abs(df["newWeight"])*scale)
            train, test = train_test_split(df, test_size=self.splitRatio, random_state=12345)
            print("Add Tree {} of type {} with {} event".format(name, outName, len(train)))
            self.trainSet = pd.concat([train.reset_index(drop=True), self.trainSet], sort=True)
            self.testSet = pd.concat([test.reset_index(drop=True), self.testSet], sort=True)
        self.groupTotal.append(1.*len(self.trainSet[self.trainSet["classID"] == len(self.groupNames)-1]))

    # Reduce frame using root style cut string
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
        
    def train(self, groupName=None):
        workTrain = self.trainSet
        workNames = self.groupNames
        useMulti=True
        if groupName != None:
            useMulti=False
            rmIdx = 3-self.groupNames.index(groupName)
            workTrain = workTrain[workTrain["classID"] != rmIdx]
            workNames = [groupName]
        X_train = workTrain.drop(self.specVars+self.extraVars+["finalWeight"], axis=1)
        w_train = workTrain["finalWeight"].copy()
        y_train = workTrain["classID"] if useMulti else [1 if cID==0 else 0 for cID in workTrain["classID"]]
        
        X_test = self.testSet.drop(self.specVars+self.extraVars+["finalWeight"], axis=1)
        y_test = self.testSet["classID"] if useMulti else [1 if cID==0 else 0 for cID in self.testSet["classID"]]
        
        sigImpFac = 1
        w_train[workTrain["classID"] == 0] *= sigImpFac
        for i, groupTot in enumerate(self.groupTotal):
            w_train[workTrain["classID"] == i] *= min(self.groupTotal)/groupTot
            #print i, groupTot, sum(w_train[self.trainSet["classID"] == i])
                
        
        # XGBoost training
        param = {}
        
        param['eta'] = 0.09
        param['silent'] = 1
        param['nthread'] = 3
        
        #param['subsample'] = 0.9
        #param['max_depth'] = 5
        #param['colsample_bytree'] = 0.5
        param['min_child_weight']=1e-06
        param['n_estimators']=200
        param['reg_alpha']=0.0
        param['reg_lambda']=0.05
        param['scale_pos_weight']=1
        param['subsample']=1
        param['base_score'] = 0.5
        param['colsample_bylevel'] = 1
        param['colsample_bytree'] = 1
        param['gamma'] = 0
        param['learning_rate'] = 0.1
        param['max_delta_step'] = 0
        param['max_depth'] = 5

        if useMulti:
            param['objective'] = 'multi:softprob'
            param['eval_metric'] = "mlogloss"
            param['num_class'] = len(np.unique(y_train))
        else:
            param['objective'] = "binary:logistic"#'multi:softprob'
            param['eval_metric'] = 'logloss'#"mlogloss"
        num_rounds = 150
        
        self.fitModel = None
        if useMulti:
            self.fitModel = xgb.XGBClassifier(**param)
            self.fitModel.fit(X_train, y_train, w_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=100, verbose=50)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
            dtrainAll = xgb.DMatrix(self.trainSet.drop(self.specVars+self.extraVars+["finalWeight"], axis=1))
            dtest = xgb.DMatrix(X_test, label=y_test, weight=self.testSet["finalWeight"])
            evallist  = [(dtrain,'train'), (dtest, 'test')]
            self.fitModel =  xgb.train(param, dtrain, num_rounds, evallist, verbose_eval=50)
        
        for i, grp in enumerate(workNames):
            self.predTest[grp] = self.fitModel.predict_proba(X_test).T[i] if useMulti else self.fitModel.predict(dtest)
            self.predTrain[grp] = self.fitModel.predict_proba(X_train).T[i] if useMulti else self.fitModel.predict(dtrainAll)

        # valid_result = self.fitModel.evals_result()
        # best_n_trees = self.fitModel.best_ntree_limit
                        
        
    # Wrapper for write out commands
    def output(self, outname):
        with uproot.recreate("{}/{}.root".format(outname, self.filename)) as outfile:
            self.writeOutRoot(outfile, "TestTree", self.testSet, self.predTest)
            self.writeOutRoot(outfile, "TrainTree", self.trainSet, self.predTrain)
        self.writeOutPandas("{}/testTree.pkl.gz".format(outname), self.testSet, self.predTest)
        self.writeOutPandas("{}/trainTree.pkl.gz".format(outname), self.trainSet, self.predTrain)
        self.fitModel.save_model("{}/model.bin".format(outname))

    # Write out pandas file as a pickle file that is compressed
    def writeOutPandas(self, outname, workSet, prediction):
        set_difference = set(workSet.columns) - set(self.allVars)
        workSet = workSet.drop(list(set_difference), axis=1)
        for key, arr in prediction.items():
            workSet.insert(0, key, arr)
        workSet.to_pickle(outname, compression="gzip")

    # Write out as a rootfile
    def writeOutRoot(self, outfile, treeName, workSet, prediction):
        outDict = {name: workSet[name] for name in self.allVars}
        outDict.update(prediction)
        del outDict["GroupName"]
        outfile[treeName] = uproot.newtree({name:"float32" for name in outDict.keys()})    
        outfile[treeName].extend(outDict)
        
        
