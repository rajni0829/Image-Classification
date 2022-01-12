# Vgg16-Transfer-Learning 

Model: VGG16 of convolutional neural network. <br>
Dataset : https://drive.google.com/drive/folders/15LTdFZUq9sCwhQa99uQqlchxTaQAHLqh?usp=sharing <br>
Packages Used : Numpy, PIL, MatplotLib, Pandas, Keras, Tensorflow <br><br>


### Methodology   </br>
I. **Dataset Preparation :** Dataset has been mounted at drive.The dataset is divided in the ratio 60:40 in which the Training Set consists of 251 images and the Validation Set consists of 127 images.

II. **Image Pre-processing :** VGG16 is used to train the model. Convolution base is built and for object recognition with a CNN, we freeze the early convolutional layers of the network and only train the last few layers which makes a prediction. Width and height of the Image is set equals to the default size used for VGG16. 

III. **Feature Extraction :** Dataset is augmented using ImageDataGenerator. This phase involves modifying the VGG16 architecture by adding a GlobalAveragePooling layer and a dense output layer. After adding this layer, we add a final, dense layer, which has an activation function attached to it. This takes all of the feature maps that it has collected, and then gives us the prediction.

IV. **Training Model :** The model has been trained for 150 epochs with Adam optimizer and Categorical Cross Entropy as loss function because multi-class classification model is used. ModelCheckpoint is the Callbacks used in the training process. ModelCheckpoint allows the best version of the model to be saved automatically.

V. **Making Prediction :** Image and its path is passed through predict function of model base by reshaping and Image with its prediction (labels) is displayed.


### Accuracy Of Model is 97.37672847509384 for the Given DataSet.  <br>

#### Transfer Learning
The Object is modelled using pre-built and pre-trained model (VGG16) and transfer learning is applied. The model has been trained and validated using 60 percent of the 6 Classes of the Caltech101 object dataset. The remaining part was used to test our accuracy using Transfer Learning.

### References
https://www.youtube.com/watch?v=LsdxvjLWkIY  <br>
https://www.youtube.com/watch?v=lHM458ZsfkM   <br>
https://medium.com/swlh/reverse-image-search-using-resnet-50-f305d735385a <br>
https://stackoverflow.com/questions/20176361/open-tar-gz-archives-in-python  <br>
https://github.com/nachi-hebbar/Transfer-Learning-Keras/blob/main/TransferLearning.ipynb  <br>
https://stackoverflow.com/questions/50953127/in-sklearn-preprocessing-module-i-get-valueerror-found-array-with-0-features  <br>
https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a  <br>
 
