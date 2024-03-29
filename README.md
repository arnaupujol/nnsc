# NNSC

> Author: **Arnau Pujol**  
> Year: **2020**  
> Version: **1.0.1**  
> Reference Paper: [arXiv:2006.07011](https://arxiv.org/abs/2006.07011)



This repository have been built to apply the model Neural Network Shear Correction (NNSC) from [Pujol et al. (2020)](https://arxiv.org/abs/2006.07011) to model
shear bias as a function of several measured properties obtained from image simulations.

Software requirements:
----------------------
All the packages that are required so that all the codes can run are:
- numpy
- scipy
- matplotlib
- tensorflow

Installation:
----------------------
To install the repository, first you have to clone it to your local machine.
Then you can simply run:

```
$ python setup.py install
```

In the text file `requirements.txt` the required packages are specified and
installed if needed. The versions required here work at least for Python 3.7.0.

Structure of the repository:
----------------------------
The main directory has the following elements:
- `run_nnsc.py`: the main executable of the code.

We can run the code by executing `run_nnsc.py`:

```
python run_nnsc.py [arguments]
```

where the arguments after the command define the parameters of the calculations.

- nnsc_data directory: directory where mock data is located. The data is stored with numpy arrays.
- nnsc directory: contains python modules that are used for the code executable.
Example: `deep_reg.py` defines functions to generate the DNN for a regression to estimate shear bias as in [Pujol et al. (2020)](https://arxiv.org/abs/2006.07011).
Example: `param.py` defines all the parameters used in this repository. The variable params is a dictionary with the names of all the parameters that can be used and their respective original values.
- nnsc_notebooks directory: it contains notebooks showing script examples.
- nnsc_output: directory where output is stored.

How to use it:
--------------

To run NNSC with the default parameters, we just need to run:

```
python run_nnsc.py
```

It will return some mat, npy and pdf files with:
- The learned machine learning model to predict shear bias (.mat).
- The indeces pointing to the objects used for the training (.npy).
- The plots showing the shear bias dependencies and performances (.pdf).
The names of these output files are specified in the output text of the code.

We can use different parameters in order to obtain different calculations and results.
The default parameters used are defined in `nnsc/param.py`, where the meaning of each parameter is defined.

We can specify the parameters we want to use in two different ways:
1- modifying them in `nnsc/param.py`, which is going to be a permanent change.
2- running the script with the arguments:

```
python run_nnsc.py par_name1 par_val1 par_name2 par_val2 … par_nameN par_valN
```

where `par_nameX` specifies the names of the parameter we want to define and `par_valX` corresponds to the value that we want to give to the parameter.

For example, if we want to run `run_nnsc.py` with:

`param[‘output_data’] = 'm1'`

`param[‘n_train’] = 10000`

`param[‘n_epochs’] = 100`

we have to run:

```
python run_nnsc.py output_data m1 n_train 10000 n_epochs 100
```



#### Example:

```
python run_nnsc.py version deep_reg_est n_epochs 1000 n_train 500000 n_test 500000 selection original dim1 30 dim2 30 dim3 30 dim4 30 output_data m learning_rate .0001 ct .0 batch_size 32 show False save True where make_plots False
```

This runs a NNSC to learn shear bias as a function of input properties.
The training takes 1000 epochs, and 500,000 objects for the train and test sets.
The input properties selected are defined as the 'original' (it is just a name used when importing the data) selection.
The dimensionality of the NN is 4 hidden layers of 30 nodes.
It learns all 4 shear biases at the same time (output_data m).
The learning rate, contamination level and batch sizes are .0001, .0 and 32 respectively.
It does not show the plots, it saves the output files, but it does not make the plots.
