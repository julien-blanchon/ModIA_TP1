import argparse
from typing import Literal

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])

def recognize_digit(image: Image) -> Literal["colorized.png"]:
    """Predict function for gradio.

    Returns
    -------
    str
        "colorized.png"
    """
    image_torch: torch.Tensor = source_process(image).unsqueeze(0)  # add a batch dimension
    with torch.no_grad():
      prediction = model(image_torch.to(device))[0]
    save_image(prediction, "colorized.png", normalize=True)
    return "colorized.png"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, default = "./unet.pth", help="State dict")
    args = parser.parse_args()

    state_dict_path = args.state_dict_path
    state_dict = torch.load(state_dict_path)

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    gr.Interface(
      fn=recognize_digit,
      inputs=gr.Image(type="pil", image_mode="L"),
      outputs="image",
      description="Select an image").launch(debug=True, share=False)
