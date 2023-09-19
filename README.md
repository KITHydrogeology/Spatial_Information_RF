[![License](by-nc-sa.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Instructions for installation are below

DOI of this repository:  

[![DOI](https://zenodo.org/badge/657178211.svg)](https://zenodo.org/badge/latestdoi/657178211)

# Incorporating spatial information for regionalization of environmental parameters in machine learning models
This repository enables you to perform the calculations shown in the manuscript: "Comin soon"

Contact: marc.ohmer@kit.edu

ORCIDs of authors:  
M. Ohmer: [0000-0002-2322-335X](https://orcid.org/0000-0002-2322-335X)  
F. Doll: [0009-0003-5455-7162](https://orcid.org/0009-0003-5455-7162)  
T. Liesch: [0000-0001-8648-5333](https://orcid.org/0000-0001-8648-5333)  

<img src="ga1.png" alt="Bildbeschreibung" width="500" height="400">

For a detailed description please refer to the publication. Please adapt all absolute loading/saving and software paths within the scripts to make them running, you need Python software for a successful application.

## Installation
To use the code and scripts in this repository, you'll need to install the required libraries and dependencies. You can do this by creating a virtual environment and using the `requirements.txt` file. Here are the steps:

1. Create a virtual environment (optional but recommended):
python -m venv spatialinfo
2. Activate the virtual environment:
source spatialinfo/bin/activate # On Unix/Linux
spatialinfo\Scripts\activate # On Windows
3. Install the required libraries from the `requirements.txt` file:  
pip install -r requirements.txt

## Usage

This repository includes the following main Python scripts:
- `RF.py`: The main script for conducting spatial analysis and prediction using Random Forest.
- `general_functions.py`: Contains general utility functions.
- `plotting_functions.py`: Contains functions for data visualization.
- `spatial_feature_functions.py`: Contains functions for extracting spatial features.

To run the analysis, you can use the `RF.py` script as follows:
python RF.py

Please refer to the script documentation and comments for more details on how to use them effectively.


## Contributions

Contributions to this project are welcome! If you find issues, have suggestions, or want to contribute new features, please open an issue or submit a pull request.


Feel free to contact us with any questions or feedback.





