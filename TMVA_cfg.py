from os import environ
import ROOT
    
batchs = 64
#nneurons = 64

#"VarTransform=I,D,P",
methodList = {
    "BDTCW":[ROOT.TMVA.Types.kBDT,":".join(["!H","!V","NTrees=500","MaxDepth=8","BoostType=Grad","Shrinkage=0.01","UseBaggedBoost","BaggedSampleFraction=0.50","SeparationType=GiniIndex","nCuts=50"])], #Gradient boosting BDT
    "Cuts":[ROOT.TMVA.Types.kCuts,"H:!V:PopSize=500:Steps=50"], #Used for cuts, get the best cuts on input variables
    "BDTA": [ROOT.TMVA.Types.kBDT, "!H:!V:NTrees=850:MaxDepth=6:BoostType=AdaBoost:AdaBoostBeta=0.05:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=30"], #Adaboost BDT
    "PyGTB": [ROOT.TMVA.Types.kPyGTB,"!V:NEstimators=850:NJobs=4"], #Gradient boosting from pymva
    "PyAda": [ROOT.TMVA.Types.kPyAdaBoost,"!V:NEstimators=1000"], # AdaBoosting from pymva
    "PyForest": [ROOT.TMVA.Types.kPyRandomForest, "!V:VarTransform=None:NEstimators=850:Criterion=gini:MaxFeatures=auto:MaxDepth=4:MinSamplesLeaf=1:MinWeightFractionLeaf=0:Bootstrap=kTRUE"] #Random forest
}


'''
SeparationType is the splitting criteria for BDTs, the other options are:
CrossEntropy
GiniIndex
GiniIndexWithLaplace
MisClassificationError
SDivSqrtSPlusB -> Probably the most relevant in your case, but GiniIndex/CrossEntropy should also yield a nice result




'''
