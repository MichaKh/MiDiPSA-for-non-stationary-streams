# MiDiPSA-for-non-stationary-streams

MiDiPSA (Microaggregation-based Differential Private Stream Anonymization) for continuously publishing non-stationary data. 
The algorithm satisfies k−anonymity, (c, l)−diversity and adhering to the conditions of ϵ−differential-privacy, combined with an unsupervised mechanism for detection of concept drift.
The algorithm evaluates the trade-off between privacy, measured by the disclosure risk, and utility, measured by the AUC of MOA stream classifiers trained on the anonymized streams (such as MajorityClass, HoeffdingAdaptiveTree, NaiveBayes, SGD and AdaptiveRandomForest).

The clustering process is illustrated below:
![picture](/MiDiPDSA_illustration_snapshot.png)

Requirements:
=============
1. Latest version of MOA installed (http://moa.cms.waikato.ac.nz). moa.jar file available in the directory of the project.
2. Python 2.7 installed. Anaconda is preferred, since it installs Numpy, Pandas and Scipy packages automatically.
3. Datasets (example is provided): data CSV file and a metadata file including list of all attributes and their type and range.

Instructions:
=============
* Run Program.py file with the following parameters:
    * ___DIR___ - directory of the datasets and metadata files.
	* ___stream_path___ - name the dataset file.
	* ___datatypes_path___ - name of the metadata file.
	* ___k___ - range of cluster size and k-anonymity parameter [k_min - k_max].
	* ___l___ - l-diversity parameter.
    * ___c___ - c parameter of recursive (c,l)-diversity (default 7).
    * ___eps___ - differential privacy parameter.
    * ___b___ - buffer size of each cluster (default 3k, where k is k-anonymity parameter).
    * ___delta___ - delay threshold parameter (default 10k).
    * ___dist_thr___ - distance threshold for choosing the nearest cluster in the clustering (should be tuned for each dataset).
    * ___cd_conf___ - confidence level of statistical test for detection of concept drift (default 0.1).
    * ___noise_thr___ - threshold for controlling noise addition to categorical attributes, as part of differential privacy mechanism.
* Log file is saved for the process in the project directory.
* Original data and its corresponding anonymization (with the classification performance) are saved in the project 'Output' directory.
* Report is produced as a CSV file in the project directory.
