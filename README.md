# Setup
I would highly suggest making a virtual environment for this considering the codeis in python 2.7 (only thing that works with ROOT on my machine).
```
python2.7 -m virtualenv env
source env/bin/activate
pip -r requirements
```
That should install all of the python packages. For the input files, simply grab from my public area and put in the main directory of this repo:
```
/afs/cern.ch/user/d/dteague/public/For_Deborah/inputTrees.root
```
Note: ttbar is having some problems so this branch may not work. It will be replaced by data driven events anyway though.

# Running
There are two main files: `mva_test.py` (uses xgboost) and `tmva_test.py`. They both use similar code. You must give a directory name that all of the plots will be saved and if you want to run the actual training, add the `--mva` flag, else the code with use the previous training in the directory supplied (in case you want to change plots). i.e
```
python mva_test.py -f testDir --mva
```
The `--useNeg` option only works on the TMVA code (tbd on xgboost). This will train the BDT using the negative weights from generator level. So far, this shows little change to the actual shape and performance of the BDT, so from current understanding, this can be ignored.

To add plots, in the TMVA code, simple add a line similar to that in the code provided, namely 
```
helper.plotFuncTMVA(signal_df, background_df, 140000, "BDT", np.linspace(-1,1.,51))
```
As will the useNeg option, this feature hasn't been extended to the xgboost script, but hopefully it will be soon

