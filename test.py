import torch
import os
from model import BiSeNet
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def evaluate(image_path, cp, input_size=(512, 512)):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)

    # Load model
    try:
        net.load_state_dict(torch.load(cp, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"Model checkpoint not found at {cp}. Please check the path.")
        return None
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None

    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(image_path)
        image = img.resize(input_size, Image.BILINEAR)  # Resize image as needed
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        return parsing

if __name__ == "__main__":
    # Change the image path and checkpoint path as necessary
    evaluate(image_path='./imgs/116.jpg', cp='cp/79999_iter.pth')
