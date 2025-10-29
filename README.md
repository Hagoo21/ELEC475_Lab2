ELEC 475 Lab 2 - Pet Nose Localization

Setup Instructions:

1. Install Dependencies
   Run: pip install -r requirements.txt

2. Download Model Weights
   Download the trained model weights from the link below and place the .pt files in the model_weights folder.
   
   Google Drive Link: https://drive.google.com/file/d/1YQbarrQDGiAwoM7BD3H8ulZCCyWd-xrx/view

3. Run Commands
   IMPORTANT: All commands must be run from the project root directory (ELEC475_Lab2).
   
   You will find the commands to test the code in the following txt files:
   - SnoutNet.txt (Base SnoutNet model)
   - SnoutNet-A.txt (AlexNet-based variant)
   - SnoutNet-V.txt (VGG16-based variant)
   - SnoutNet-Ensemble.txt (Ensemble model)

No Configuration Changes Needed:
The code uses relative paths, so you do NOT need to modify config.json.
Just make sure to run all commands from the project root directory.


Model Files:
- model.py contains the base SnoutNet CNN architecture
- snoutnet_alexnet.py contains the AlexNet-based variant
- snoutnet_vgg16.py contains the VGG16-based variant
- ensemble_model.py contains the ensemble model that averages all three models
