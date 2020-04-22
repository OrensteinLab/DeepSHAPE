# DeepSHAPE 
Command line for training and testing<br />
python main.py train \<in_vitro\in_vivo\> \<path to training sequences file\>  \<path to training RNAplfold file\> \<path to training annotation file\><br />
python main.py predict \<in_vitro\in_vivo\> \<path to testing sequences file\>  \<path to testing RNAplfold file\> \<path to testing annotation file\><br />

- Create a training directory containing three files: training sequences file, training RNAplfold file, training annotation file.
- Create a testing directory containing three files: testing sequences file, testing RNAplfold file, testing annotation file.

Outputs:
- Under the current working directory, a new directory, named “saved_models”, will be created containing the trained models.
- Under the directory which contains the test files, a new directory, named “outputs”, will be created containing the test sequences and their icSHAPE predictions.
