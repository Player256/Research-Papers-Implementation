import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
from torch import optim
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_vgg_model():
    model_weights = torchvision.models.VGG19_Weights.DEFAULT
    model = torchvision.models.vgg19(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model = model.features
    return model

def preprocess(img):
    image = Image.fromarray(img).convert('RGB')
    imsize = 196
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(dim=0)
    return image

def deprocess(image):
    image = image.clone()
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.cpu().detach().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image

def get_features(image, model):
    features = {}
    layers = {
        '0': 'layer_1',
        '5': 'layer_2',
        '10': 'layer_3',
        '19': 'layer_4',
        '28': 'layer_5'
    }
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(image):
    b, c, h, w = image.size()
    image = image.view(c, h * w)
    gram = torch.mm(image, image.t())
    return gram

def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def style_loss(target_features, style_grams):
    loss = 0
    for layer in target_features:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        loss += layer_style_loss
    return loss

def total_loss(content_loss, style_loss, alpha, beta):
    return alpha * content_loss + beta * style_loss

def predict(content_image, style_image):
    model = create_vgg_model().to(device).eval()
    content_img = preprocess(content_image).to(device)
    style_img = preprocess(style_image).to(device)
    target_img = content_img.clone().requires_grad_(True)
    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    style_gram = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    optimizer = optim.Adam([target_img], lr=0.06)
    alpha_param = 1
    beta_param = 1e2
    epochs = 60
    for i in range(epochs):
        target_features = get_features(target_img, model)
        c_loss = content_loss(target_features['layer_4'], content_features['layer_4'])
        s_loss = style_loss(target_features, style_gram)
        t_loss = total_loss(c_loss, s_loss, alpha_param, beta_param)
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
    results = deprocess(target_img)
    return Image.fromarray((results * 255).astype(np.uint8))

title = "Neural Style Transfer 🎨"

demo = gr.Interface(fn=predict,
                    inputs=['image', 'image'],
                    outputs=gr.Image(),
                    title=title)

demo.launch(debug=False, share=False)
