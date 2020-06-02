# Setup
I would highly suggest making a virtual environment for this considering the code is in python 2.7 (only thing that works with ROOT on my machine).
```
python2.7 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
That should install all of the python packages. For the input files, simply grab from my public area and put in the main directory of this repo:
```
/afs/cern.ch/user/d/dteague/public/For_Deborah/inputTrees.root
```
Note: ttbar is having some problems so this branch may not work. It will be replaced by data driven events anyway though.

# Running
The code is run totally out of the file `runMVA.py`. The code is based on two main jobs--running the training and making output plots. The code is run so all of the output is put into a folder that is specified by the `-o` or `--out` option. 

## Training
Without options, the training does not run and only the output graphs are made. To specify that you want to train, you must give the `-t/--train` option with the type of training. There are two modes that the training can be run over: TMVA and XGBoost, noted by `tmva` and `xgb` respectively. 

Other features that can be changed on the train are the cut, input groups, and variables. The types of variables are use variables (ones used for training), and spectator variables (ones saved but not used in training). Note: in code, "newWeight" assigns cross-section information and is not used in training, so this MUST be in the `specVar` array.

All training is done with multiclassification modes. To get old functionality (not multiclass mode), you'll have to just peek in old commits and use the code there, sorry

## Making Plots
The plot code is still pretty dirty and will need updating as time goes on, forewarning.

The basic workflow is the output of the training is read by the MVAPlotter object and the plots are made with the helper functions. Currently in the code are ways to make ROC curves (`helper.makeROC`), ways to make S/Sqrt(S+B) curves (`helper.StoB`), and ways to plot generic variables (`helper.plotFunc`). All graphs aren't shown by default. If you wish to see the graphs as they are made, use the `--show` option

You can change the luminosity of the graphs by using the `-l/--lumi` option. 

## Example usage
```sh
python runMVA.py -t tmva -o firstTMVARun -l 140
python runMVA.py -t xgb -o firstXGBoostRun -l 35.9
```
hello
