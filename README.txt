#AMPScanner README.txt
#By: Dan Veltri - 11.1.2020

NOTE: While best efforts have been made to ensure the integrity of this script, we take no
responsibility for damages that may result from its use!

---
INSTALLATION

This script requires Python(2.7+ on Unix, 3.5+ on Windows) to be installed on your system.

If you have Python and PIP installed (see: https://pypi.org/project/pip/) you can use the appropriate "requirements.txt" file to install the needed libraries. Note, that the package versions are slightly different depending if you are using the pertained model from the original paper or newer pre-trained models.

If you are only interested in running the original paper model use:
"pip install -r requirements_original_paper_model.txt"

If you plan to use one of the trained models from 2019 or newer, use:
"pip install -r requirements_2019_and_newer_model.txt"


You may need admin (sudo) access on your system to install some packages. You may also be able to install them locally.
See: https://stackoverflow.com/questions/7465445/how-to-install-python-modules-without-root-access

If you use Mac OSX or Windows - many people report better luck installing TensorFlow using the
Anaconda package manager (https://conda.io). On a Windows machine you will also need Python3 rather than 2 for the install.

If you plan to use both the original and newer models, I highly recommend using Python environments so that you can easily switch to the appropriate set of packages. For more details see: 
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

*** Unfortunately, we can't provide support for getting packages installed ***
However, the TensorFlow and Keras communities are very active in helping users with problems. Please see: https://www.tensorflow.org/install/ for TensorFlow and https://keras.io/#installation for Keras-related issues.

---
USAGE:

It is recommended to only use sequences between 10-200AA in length if possible! See the Paper and https://www.dveltri.com/ascan/v2/about.html for details.

To run the program using a pre-trained model, simply type in the command line:

python amp_scanner_predict.py <my_file.fasta> <my_trained_model>

Where <my_file.fasta> is the path to your valid FASTA file with your peptide sequences and <my_trained_model> is the path to the pre-trained model you are using. The program will make predictions save in two files:

"my_file_Prediction_Summary.csv" which is a CSV file (can be opened in Excel) with your sequence prediction information and
"my_file_AMPCandidates.fa" which is a FASTA file with the peptides that were predicted as AMPS.

Example if your FASTA file is called "mydata.fasta" and saved in the same folder as the script: 

python amp_scanner_predict.py mydata.fasta TrainedModels/020419_FULL_MODEL.h5



To train your own model using your own AMP and DECOY sequences, use the "amp_train_and_predict_cnn-lstm_model.py" file. You will need to edit this file to point to the training, validation, and testing scripts for both your AMP and DECOY files saved in standard FASTA format.

Example if you have already edited the script to point to where your FASTA files are saved on your machine:

python amp_train_and_predict_cnn-lstm_models.py

This will save your model as "my_amp_model.h5" in the same directory as the script.

---
A NOTE ON REPRODUCIBILITY: 
Because these scripts were developed using the older Tensorflow vr 1.x backend - training a model on a multicore (or GPU) machine, even using the same random seed on the original dataset will result in slightly different results from that of the paper (but should still be in the ballpark of the standard deviations of the 10-fold cross-validation experiments in the paper). A Google search of this issue will bring back a lot of discussions on the topic. I believe the multicore reproducibility problem has finally been addressed with Tensorflow 2.x but you will need to update the code and packages to make things compatible. If you update to TensorFlow 2.x code see: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development.
