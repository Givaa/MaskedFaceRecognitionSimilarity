# MaskedFaceRecognitionComparison

FVAB project based on Siamese Neural Network used for the comparison between masked and unmasked subjects.

## Why Masked Face Recognition?

With covid-19 and the introduction of anti-counterfeiting security measures such as masking, classical face recognition techniques have become obsolete. 
Masked Face Recognition aims to solve this problem by recognising the periocular area.
The study 'Masked Face Recognition: Human vs Machine' defines the difficulty of facial recognition, highlights the difficulty of recognising a masked subject and the need for the intervention of a human operator. 
Along the lines of this study, we chose to conduct an in-depth study of the datasets and carry out tests to put a value on the impact of the mask on the face.

## Celeb Masked Simulated

Celeb Masked Simulated is generated from the MS-CELEB-1M dataset, containing one million images of celebrities, from which 10,000 identities were extracted.
After extraction, Mask The Face was applied, 
generating 4 images with 4 different types of masks per identity.
Given the impossibility of careful classification due to a lack of sorting, as evidenced in the analysis of previous datasets, our dataset is divided into five classes by identity:

Unmasked: subject without mask
Mask type 1: subject having a white surgical mask
Mask type 2: subject having a black tissue mask
Mask type 3: subject wearing a blue FFP2 mask.
Mask type 4: subject wearing a red textile mask.

## Siamese Network

A Siamese network is a neural network consisting of two identical sub-networks that share the same weights with each other. 
It allows the contrast (distance) between two different inputs given to the two networks to be maximised. 
For the following project, it was used for image similarity techniques, i.e. to score the diversity of two images, given by the mask.
The network was constructed using TensorFlow and Keras.
To actually evaluate the difference between two images, the Euclidean distance was used as input layer.
Being a Siamese network, two input layers based on the Euclidean distance are used. 
A Lambda layer combines the results of the two layers using the Euclidean distance and sends the result to the final network. 
This performs the final evaluation and then outputs the similarity score given by the distance.

### Loss function
The loss function describes the efficiency of the model with respect to the expected result.
There are various types of loss or cost functions. In our neural network, the Constranctive loss has been implemented.
This loss function treats the output of the network as a positive example, calculates its distance to an example of the same class and contrasts it with the distance to negative examples.


## Authors

- [@Givaa](https://github.com/Givaa)
- [@IKinderBueno](https://github.com/lKinderBueno)