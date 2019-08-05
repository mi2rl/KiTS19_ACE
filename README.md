# KiTS19_ACE
KiTS 2019 challenge in MICCAI 2019  
Team name : ACE (Asan Coreline Ensemble)  
  
## Training
- **TO DO**

## Prediction
- For searching ROI of kidney  
  `python evaluation.py --mode 1 --testset /path/testset`
- For predicting kidney and tumor  
  Select a mode using prediction. Before predicting kidney and tumor, **RUN** the mode 1 first.  
  2_1 : coreline's model
  2_2 : model with dice loss, normalization with tumor's mean and std and using **ONLY ONE** kidney in CT.  
  2_3 : model with dice loss, minmax scaling and using **ALL** kidney in CT.  
  2_4 : model with focaldice loss, minmax scaling and using **ALL** kidney in CT.  
  2_5 : model with dice loss, normalization with tumor's mean and std and using **ALL** kidney in CT.  
  `python evaluation.py --mode 2_3 --testset /path/testset`
