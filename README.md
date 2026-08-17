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

Once activated, run: `pip install -r requirements.txt`

## Folder Structure
```
eye_disease_detection/
├── data/ # dataset (not tracked in git, see Dataset section above)
├── notebooks/ # exploration (unused so far)
├── src/
│ ├── data_loader.py # loads dataset, stratified split, preprocessing, augmentation
│ ├── model.py # from-scratch CNN architecture
│ ├── train.py # trains the from-scratch CNN, saves to models/
│ ├── train_resnet.py # ResNet50 transfer learning model, saves to models/
│ ├── evaluate.py # confusion matrix + classification report, from-scratch CNN
│ └── evaluate_resnet.py # confusion matrix + classification report, ResNet50
├── models/ # saved trained models (not tracked in git)
├── app/
│ └── app.py # Streamlit app (uses the ResNet50 model)
├── results/ # (for confusion matrix images / plots, if added)
├── docs/ # slides and writeup materials
├── requirements.txt
└── README.md
```

## Results Summary

| Model | Accuracy | Glaucoma Recall | Retina Disease Recall |
|---|---|---|---|
| From-scratch CNN | 58% | 25% | 13% |
| ResNet50 (transfer learning) | 63% | 38% | 20% |

ResNet50 improved overall accuracy and gave the clearest gains on our smallest, hardest classes — consistent with pretrained features helping most when per-class training data is limited.

## Team & Task Split
| Member | Area |
|---|---|
| Tobi | Data pipeline (split, preprocessing, augmentation) |
| Onono | From-scratch CNN (build, train, evaluate) |
| Tobi | ResNet50 transfer learning (build, train, evaluate) |
| Nifemi | Streamlit app |
| Onono and Nifemi| Slides & documentation |