# Inter-Pol: An Interpretable Machine Learning Framework for Solvent Polarity Prediction

# Reference Implementation of Inter-POL algorithm
This readme file documents all of the required steps to run Inter-POL.

Note that the code was implemented and tested on a Linux operating system only.

## How to set up the environment
We have provided an Anaconda environment file for easy setup.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `inter_pol` environment using the following command:
```bash
conda env create -n inter_pol -f environment.yml
conda activate inter_pol
```

# In order to install requirement packages
```bash
pip install -r requirements.txt
```

# Step by step files:

1_Generate_features.py: prepare features

2_Inter-Pol_train_with_fingerprint.py: train and test the model


## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com
