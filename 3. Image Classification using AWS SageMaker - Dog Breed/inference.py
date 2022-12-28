# Tutorial: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html

import os
import sys
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import requests
import json

from PIL import Image

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def net():
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def model_fn(model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model.to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading model ...")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('Model loaded')
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    print(f'Body CONTENT-TYPE: {content_type}')
    print(f'Body TYPE: {type(request_body)}')
    
    if content_type == JPEG_CONTENT_TYPE: 
        print('Loaded JPEG content')
        return Image.open(io.BytesIO(request_body))
    
    if content_type == JSON_CONTENT_TYPE:
        print('Loaded JSON content')
        print(f'Request body: {request_body}')
        request = json.loads(request_body)
        print(f'JSON object: {request}')
        url = request['url']
        imageContent = requests.get(url).content
        return Image.open(io.BytesIO(imageContent))
    
    raise Exception('Unsupported content type in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction

def output_fn(predictions, content_type):
    print(f'Postprocess CONTENT-TYPE: {content_type}')
    assert content_type == JSON_CONTENT_TYPE
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)