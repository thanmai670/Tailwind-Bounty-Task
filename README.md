# Streamlit Application for Bug Report Assignment

In this section, I'll explain the Streamlit application designed to classify bug reports by assigning them to the correct developer. This application leverages a pre-trained BERT model, allowing users to input text and get predictions directly through a web interface.

## Code Explanation

### Importing Libraries

```python
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
```
-streamlit (st): Used to create and control the web application interface.
-BertTokenizer, BertForSequenceClassification: From the transformers library, these are used to tokenize input text and load the pre-trained BERT model.
-torch: This library is utilized to handle tensors and perform model inference.
-pickle: Used for loading the serialized Python object (label encoder).

### Model and Tokenizer Loadingmodel_path = './saved_model'

```python
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()  

# Set the model to evaluation mode
The model and tokenizer are loaded from a pre-saved directory. This directory contains the trained BERT model which has been fine-tuned for our specific task.
The model is set to evaluation mode, which is necessary when the model is used for inference (predicting), as it disables certain layers like dropout.

### Loading the Label Encoder
```python
label_encoder_path = './saved_model/label_encoder.pkl'
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

-The label encoder, saved in a pickle file, is loaded. 
-This encoder translates numerical class IDs back to developer names, which are understandable and useful for end users.

### Prediction Function
```python
def predict(text, model, tokenizer, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_developer = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_developer

-This function handles the process of making predictions:
-Tokenization: Converts the input text into a format suitable for the BERT model.
-Model Inference: Runs the model on the tokenized inputs and calculates the logits (model outputs before activation function).
-Developer Prediction: Finds the class with the highest logit value and translates this class ID back to the corresponding developer's name.

By integrating a sophisticated machine learning model like BERT and providing a user-friendly web interface, the application aids in efficient bug management and accelerates the development workflow.

-Kind regards,
Thanmai
