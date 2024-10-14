import gradio as gr

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import torchvision

import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()
        self.layers = {
            '0' : 'conv1_1',
            '5' : 'conv2_1',
            '10' : 'conv3_1',
            '19' : 'conv4_1',
            '21' : 'conv4_2', #content representation
            '28' : 'conv5_1'
        }
    
    def forward(self,x):
        features = {}
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in list(self.layers.keys()):
                features[self.layers[str(layer_num)]] = x
        return features

def load_image(image_path, max_size=400, shape=None):
    if "http" in image_path:
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image

def gram_matrix(tensor):
    b,d,h,w = tensor.size()
    tensor = tensor.view(d,h*w)
    gram = torch.mm(tensor,tensor.t())
    return gram

total_steps = 3000
learning_rate = 0.003
style_weights = {
    'conv1_1' : 1.0,
    'conv2_1' : 0.75,
    'conv3_1' : 0.2,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2
}
alpha = 1
beta = 1e6

def get_style_transfer(content_image, style_image):
    content = load_image(content_image).to(device)
    style = load_image(style_image, shape=content.shape[-2:]).to(device)
    
    model = VGG().to(device).eval()
    generated = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([generated], lr=learning_rate)
    
    style_features = model(style)
    content_features = model(content)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    for step in range(total_steps):
        generated_features = model(generated)
        content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        style_loss = 0
        for layer in style_weights:
            generated_gram = gram_matrix(generated_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((generated_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (generated_features[layer].nelement())
        
        total_loss = alpha * content_loss + beta * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step [{step}/{total_steps}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")
    
    final_img = generated.clone().squeeze()
    final_img = final_img.detach().cpu().numpy().transpose(1, 2, 0)
    final_img = np.clip(final_img, 0, 1)
    
    return final_img

demo = gr.Interface(
    fn=get_style_transfer,
    inputs=[gr.Image(type="numpy"), gr.Image(type="numpy")],
    outputs=gr.Image(type="numpy"),
)

demo.launch()
