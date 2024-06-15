Face Mask Detection
This project focuses on detecting whether a person is wearing a face mask or not using deep learning techniques. The dataset used for training and testing the model is the "Face Mask Dataset" from Kaggle.

Dataset
The dataset can be downloaded from Kaggle. It contains images of people with and without face masks.

Installation
To run this project, you'll need to install the necessary packages. Ensure you have the kaggle package installed to download the dataset.

Setup
1. Kaggle API
Make sure you have the kaggle.json file containing your Kaggle API credentials. Configure the path to this file to allow the script to access the Kaggle API.

2. Download the Dataset
Download the face mask dataset from Kaggle. Once downloaded, extract the dataset to access the images of people with and without masks.

Data Preparation
1. Load the Dataset
First, list all the images in the dataset directories for "with mask" and "without mask".

2. Create Labels
Create labels for the images: assign 1 for images with masks and 0 for images without masks.

3. Convert Images to Numpy Arrays
Resize the images to a uniform size and convert them to numpy arrays. This is necessary for feeding the images into the neural network.

4. Split the Data
Split the data into training and testing sets to evaluate the performance of the model. The training set is used to train the model, and the testing set is used to evaluate it.

Model
1. Build the Model
Construct a convolutional neural network (CNN) model. The model includes convolutional layers for feature extraction, pooling layers to reduce the spatial dimensions, and dense layers for classification.

2. Compile the Model
Compile the model by specifying the optimizer, loss function, and metrics. The optimizer helps in minimizing the loss, and the metrics help in evaluating the performance.

3. Train the Model
Train the model using the training data. During training, the model learns to differentiate between images with and without masks.

4. Evaluate the Model
Evaluate the model using the testing data to see how well it performs on unseen data. This step provides an accuracy metric indicating the model's performance.

Results
Plot the training and validation loss and accuracy to visualize the model's performance over epochs. This helps in understanding how well the model is learning and if it's overfitting or underfitting.

Predict
Make predictions on new images. The image to be predicted is preprocessed (resized and scaled) and then fed into the model. The model outputs the prediction indicating whether the person in the image is wearing a mask or not.

