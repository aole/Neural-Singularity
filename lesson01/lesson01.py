# Web application framework
import streamlit as st
# deep learning library
import torch
# pretrained models
from torchvision import models, transforms
# handle image files
from PIL import Image
# decode json files
import json
# one of the best libraries for handling data and analysis
import pandas as pd

# get around the file encoder warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# import pretrained ResNet model
model = models.resnet34(pretrained=True);

# get labels for each of the categories recognized by the model
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
class_idx = json.load(open("imagenet_class_index.json"))
labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

# modify the image to what the model expects
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

'''
# Image Recognizer

Click on browse files to load an image file.
The machine will then try to determine what is in the image.'
'''
f'Total Number of identified categories: {len(labels)}'

# streamlit widget to show open file dialog
file_name = st.file_uploader('', type=['jpeg','jpg','png','jfif'])
# if file selected
if file_name is not None:
    # open the image file
    img = Image.open(file_name)
    # show the image on the browser
    st.image(img, width=512)

    # modify the image to fit as an input to the model
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # run the model to predict what object is in the image
    model.eval()
    out = model(batch_t);
    _, index = torch.max(out, 1)
    # calculate how confident is the model about its prediction
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # show the predictions
    _, indices = torch.sort(out, descending=True)
    df = pd.DataFrame([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]], columns=['Label', 'Confidence'])
    df.index = df.index + 1; df
