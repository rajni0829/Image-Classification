# Vgg16-Transfer-Learning

Model Used: VGG16 of 
Dataset : https://drive.google.com/drive/folders/15LTdFZUq9sCwhQa99uQqlchxTaQAHLqh?usp=sharing

Packages Used : Numpy, PIL, MatplotLib, Pandas, Keras, Tensorflow,

### Methodology   </br>
we will be training a convolutional neural network (CNN) that can identify objects in images. We’ll be using a part of Caltech 101 dataset which has images in 101 categories. Most categories only have 50 images which typically isn’t enough
for a neural network to learn to high accuracy. </br>
Therefore, instead of building and training a CNN from scratch, we’ll use a pre-built and pre-trained model (VGG16) applying transfer learning.
The approach starts with choosing a model which is pretrained on a large dataset,
which in our case is VGG16. For object recognition with a CNN, we freeze the early
convolutional layers of the network and only train the last few layers which makes a
prediction.<br>

Following is the general outline for transfer learning for object recognition:
1. Load in a VGG16 CNN model trained on a large dataset
2. Freeze parameters (weights) in model’s lower convolutional layers
3. Add custom classifier with several layers of trainable parameters to model
4. Train classifier layers on training data available for task
5. Fine-tune hyperparameters and unfreeze more layers as needed.

I. Dataset Preparation : The model has been trained and validated using the
Caltech101 object dataset. It is a publicly available dataset and can be used to
build classification models for objects belonging to 101 categories. The dataset has
been divided in the ratio 60:40 in which the training set consists of 5487 images and
the Validation Set consists of 3659 images.

II. Image Pre-processing : The following image pre-processing steps are performed on
the training images before we implement them in our deep learning algorithm.

III. Model Development : This phase involves modifying the VGG16 architecture by
adding a GlobalAveragePooling layer and a dense output layer. After adding this
layer, we add a final, dense layer, which has an activation function attached to it.
This takes all of the feature maps that it has collected, and then gives us the
prediction.

IV. Training Model : The model has been trained for 150 epochs with Adam optimizer
and Categorical Cross Entropy as loss function. ModelCheckpoint is the Callbacks
used in the training process. ModelCheckpoint allows the best version of the model
to be saved automatically.

### References

https://medium.com/swlh/reverse-image-search-using-resnet-50-f305d735385a <br>
https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a  <br>
https://stackoverflow.com/questions/50953127/in-sklearn-preprocessing-module-i-get-valueerror-found-array-with-0-features  <br>
https://github.com/nachi-hebbar/Transfer-Learning-Keras/blob/main/TransferLearning.ipynb  <br>
https://www.youtube.com/watch?v=lHM458ZsfkM   <br>
https://stackoverflow.com/questions/20176361/open-tar-gz-archives-in-python  <br>
