# A Multi-Class Eye Disease Classification Model

Classifies retina images into 4 categories (normal, cataract, glaucoma, retina disease) using a CNN built from scratch, compared against a ResNet50 transfer learning model.

## Dataset
Source: `https://github.com/yiweichen04/retina_dataset`

Clone it into the data/folder:
`git clone https://github.com/yiweichen04/retina_dataset.git data/retina_dataset`

Normal - 300 images
Glaucoma, Retina and Cataract - ~100 images each

There is an obvious imbalance which will be solved by augmenting the smaller classes. 

## Setup
To Create Virtual Environment, run: `python -m venv venv`
Then: `venv\Scripts\Activate`

If PowerShell blocks the activation with a script execution policy error (common on Windows), run this once first, then retry activating: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Once activated, type into requirements.txt this:
```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
streamlit
```
Then run: `pip install -r requirements.txt`