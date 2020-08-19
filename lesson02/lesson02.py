# Web application framework
import streamlit as st
# deep learning library
import torch
# pretrained models
from torchvision import models, transforms, datasets
# handle image files
from PIL import Image
# decode json files
import json
# dealing with arrays?
import numpy as np
# one of the best libraries for handling data and analysis
import pandas as pd

# get around the file encoder warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# get labels for each of the categories recognized by the model
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
class_idx = json.load(open("imagenet_class_index.json"))
labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

'''
# Image Recognizer 2

Save images in lesson02/images folder and click on Predict Labels button.
The model will try to identify each image.
'''

# import pretrained ResNet model
model = models.resnet34(pretrained=True)

# modify the image to what the model expects
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# transform just to display images
transformST = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
])

# images directory
# directory should have subdirectory containing images.
# The subdirectory name denotes category which we'll see how to use next week
images_path = 'lesson02/images/'

# Create a progress bar
progressbar = st.progress(0.)

# create a button to run our model
if st.button('Predict Labels'):
    # ImageFolder dataset reads images from the folder and transforms them.
    dataset = datasets.ImageFolder(root=images_path, transform=transform)
    # a DataLoader is used to feed inputs (in this case images) to a model in batches.
    # We are using a batch size of 1, you can increase it.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Set the model in evaluation mode (as opposed to training mode)
    # as we are just predicting labels/categories not training the neural network.
    model.eval()

    # how many images to predict?
    numImages = len(dataset)

    # this will store labels for each image that we receive from the model
    predictions = []
    # iterate thru' each image
    for i, data in enumerate(dataloader):
        # data from dataloader is a tuple of the image and a label associated with it.
        # In our case we do not provide any label, so no useful for us.
        # This will be useful if we are training the network and giving it correct labels.
        inputs, _ = data
        # have the model predict the label for the image
        out = model(inputs)
        # save the prediction
        predictions.append(labels[out[0].argmax()])

        # update the progress bar
        progressbar.progress(i/numImages)

    # this dataset is just to display images
    dataset = datasets.ImageFolder(root=images_path, transform=transformST)
    # Creating a list of images from the dataset to display them on the browser
    images = [x[0] for x in dataset]
    # display the images with predicted labels
    st.image(images, width=150, caption=predictions)

    # make sure the progress bar is full.
    progressbar.progress(1.)

else:
    # this dataset is just to display images
    dataset = datasets.ImageFolder(root=images_path, transform=transformST)
    # Creating a list of images from the dataset to display them on the browser
    images = [x[0] for x in dataset]
    # display the images without labels (before predictions are made)
    st.image(images, width=150)

