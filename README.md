# TinySiamese network
<p align="justify">
The TinySiamese neural network took on a new look and a new way of working which was different from the standard Siamese network. The difference first appeared in the input processing of the network. 
Instead of having images as input, the input was the output feature vector of a pre-trained feature extractor. In other words, all input images would be transformed into feature vectors using a feature extractor (here: ACGAN Discriminator; previously VGG16).
Then, the Tiny-Siamese encoded the features in a small set of layers and finally calculated the distance between two encoded feature vectors and generated similarity score. Using this score, the model was trained from scratch with the Adam optimization algorithm and binary cross-entropy loss function.
</p>

<p align="center">
  <img src="https://github.com/Islem-Jarraya/TinySiamese-network/assets/79153028/d3774b70-9163-4e1b-9b98-b45a046b1135" alt="Tiny-Siamese2">
</p>
<p align="center">Figure 1: The TinySiamese Network [1].</p>

## Instalation
Requires python 3

## Code structure
### Step 0:
  Run "GANTrain.py" to train an ACGAN on fingerprint classes and save the Discriminator to "model/GANFVC.h5" (feature extractor, 4096-d).
### Step 1:
  Run the "FeatureVectorsTrain.py" file to create the CSV files of features (csv_train.csv), labels (csv_trainLabels.csv), and paths (csv_trainPaths.csv).
### Step 2:
  Run the "FeatureVectorsTest.py" file to create the CSV files of features (csv_test.csv), labels (csv_testLabels.csv), and paths (csv_testPaths.csv).
  These two files use the trained ACGAN Discriminator for feature extraction. The model should be saved in the "model" folder as GANFVC.h5.
### Step 3:
  After completing Step 1 and Step 2, the CSV files will be created. We can then start the execution of the training code in the "TinySiameseTrain.py" file. Running this file will produce the trained TinySiamese model in the "model" folder.
### Step 4:
  The file "TinySiameseTest.py" contains the testing code for accuracy calculation.
  
## Example - Fingerprint classification(FVC)
<p align="justify">
FVC2002[3] and FVC2004[2] datasets include noisy images acquired by different live scan devices. The fingerprints of each dataset were categorized into four types: arch, right loop, left loop and whorl. The four sets of FVC2004 were merged into a single set of four classes to form a multi-sensor fingerprint dataset. The same procedure was used for FVC2002 using only three sets (DB1, DB2 and DB4). This example does not include all the data but only some images.
</p>

## References:
[1] Jarraya, I., Hamdani, T.M., Chabchoub, H. et al. TinySiamese network for biometric analysis. Int. J. Mach. Learn. & Cyber. 16, 7897–7911 (2025).

[2] MAIO, Dario, MALTONI, Davide, CAPPELLI, Raffaele, et al. FVC2004: Third fingerprint verification competition. In : International conference on biometric authentication. Berlin, Heidelberg : Springer Berlin Heidelberg, 2004. p. 1-7.

[3] MAIO, Dario, MALTONI, Davide, CAPPELLI, Raffaele, et al. FVC2000: Fingerprint verification competition. IEEE transactions on pattern analysis and machine intelligence, 2002, vol. 24, no 3, p. 402-412.
