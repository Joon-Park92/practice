import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax
from PIL import Image
import numpy as np

# Setup an inference pipeline with a pre-trained model
model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
model.eval()

# Define the inference pipeline
pipe = Sequential(
    transforms.Resize([256, 256]),
    transforms.CenterCrop([224, 224]),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    model,
    Softmax(1),
)

# Save inference artifact using torch.script
scripted = torch.jit.script(pipe)
scripted.save("inference_artifact.pt")

# NOTE: normally we would uplaod it to the artifact store
# Load inference artifact
pipe_reload = torch.jit.load("inference_artifact.pt")
img = Image.open("dog.jpg")
img.load()

data = transforms.ToTensor()(np.asarray(img, dtype="unit8").copy()).unsqueeze(
    0
)

with torch.no_grad():
    logits = pipe_reload(data).detach()

proba = logits[0]
