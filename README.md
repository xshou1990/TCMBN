# Transformer Conditional Mixture Bernoulli Network

Source code for Transformer Conditional Mixture Bernoulli Network.

### Instructions 
1. folders:
a.data. The **data** folder is already placed inside the root folder. Each dataset contains train dev test pickled files. 
b.preprocess. We used to convert data into standard input for our model.
c.transformer. This folder contains our main models, modules that supports the training of our models.
d.dunnhumby_qualitative. This folder contains qualitative results on dunnhumby, supporting result on Figure 3 in main text. 

2. **bash run.sh** to run the code. If permission denied, put chmod u+x run.sh on commandline. and then ./run.sh.  See discussion https://stackoverflow.com/questions/18960689/ubuntu-says-bash-program-permission-denied. Default (hyper)parameters are given for our model. run.sh calls either Main.py and evaluate.py. 

3. Change the num_types accordingly for each dataset:
num_types = 24   #dunnhumby
num_types = 39  # defi
num_types = 169 # mimic
num_types = 232 #synthea
num_types = 5 # synthetic data
It is located under transformer/Constants.py.


4. Other datasets are [here]: (https://drive.google.com/drive/u/0/folders/1qFczWc_sVwOsVIga0oywhnVs2vyqVr5r)

5. Consider citing our paper : Concurrent Multi-Label Prediction in Event Streams, in AAAI 2023




