import gradio as gr
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

class DigitCNNFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

ROOT = Path(__file__).parent
path_to_model_pth = ROOT / "mnist_digitcnn_final_best.pth"
model = DigitCNNFinal()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(path_to_model_pth, map_location=device))
model.eval()
model.to(device)

transform = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor()
])


def predict(editor_data):
    if editor_data is None or "composite" not in editor_data or editor_data["composite"] is None:
        return None, None
        
    img = editor_data["composite"]     
    img = ImageOps.invert(img.convert("L"))
    
    transformed = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(transformed)
        pred_class = preds.argmax(1).item()

    transformed_img = transformed.squeeze().cpu()
    img_min, img_max = transformed_img.min(), transformed_img.max()
    if img_max > img_min:
        transformed_img = (transformed_img - img_min) / (img_max - img_min)
        
    transformed_img = T.ToPILImage()(transformed_img)

    return transformed_img, pred_class

demo = gr.Interface(
    fn=predict,
    inputs=gr.ImageEditor(
        type="pil", 
        image_mode="L", 
        sources=[],    
        canvas_size=(400, 400),
        brush=gr.Brush(default_size=5, colors=["#000000"], color_mode="fixed")
    ),
    outputs=[
        gr.Image(label="Transformed Image (What the model sees)"),
        gr.Number(label="Prediction", precision=0)
    ],
    title="Digit Classifier Demo",
    description="Draw a number and see how the model interprets it."
)

demo.launch()