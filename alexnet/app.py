import gradio as gr
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from model import AlexNet
import gdown

model = AlexNet()
# Google Drive link
url = 'https://drive.google.com/file/d/1D-qsL277cdvKwf9IXL4lcNSFYiX6SWCi/view?usp=drive_link'
output = 'alexnet_model.pth'
gdown.download(url, output, quiet=False)

# Load the model
model.load_state_dict(torch.load(output))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# Load CIFAR-100 labels
cifar100 = CIFAR100(root='./data', download=True)
labels = cifar100.classes


def predict(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return labels[predicted.item()]

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    examples=[
        "airplane.jpeg", 
        "bed.jpeg", 
        "boat.jpeg", 
        "fish.jpeg", 
        "dog.jpeg", 
        "maple.jpeg", 
        "person.jpeg", 
        "shark.jpeg"
    ]
)

if __name__ == "__main__":
    iface.launch()