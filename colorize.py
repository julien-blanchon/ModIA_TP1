"""Colorize."""

import argparse  # to parse script arguments

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid  #to generate image grids, will be used in tensorboard
from tqdm import tqdm  #used to generate progress bar during training

from data_utils import get_colorized_dataset_loader  # dataloarder
from unet import UNet

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
        net: UNet,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        epochs: int = 5,
        writer: SummaryWriter | None = None) -> float:
    """Train the Unet with the given optimizer and dataloader.

    Parameters
    ----------
    net : UNet
        The Unet Network
    optimizer : torch.optim.Optimizer
        The Optimizer
    loader : DataLoader
        The Dataloader
    epochs : int, optional
        The epochs, by default 5
    writer : SummaryWriter | None, optional
        The writer, by default None

    Returns
    -------
    float
        The loss
    """
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = []
        mean_loss = torch.inf
        outputs = []
        for x, y in (t := tqdm(loader)): # x: black and white image, y: colored image
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss = sum(running_loss) / len(running_loss)
            t.set_description(f"training loss: {mean_loss}")
        if writer is not None:
            #Logging loss in tensorboard
            writer.add_scalar("training loss", mean_loss, epoch)
            # Logging a sample of inputs in tensorboard
            input_grid = make_grid(x[:16].detach().cpu())
            writer.add_image("Input", input_grid, epoch)
            # Logging a sample of predicted outputs in tensorboard
            colorized_grid = make_grid(outputs[:16].detach().cpu())
            writer.add_image("Predicted", colorized_grid, epoch)
            # Logging a sample of ground truth in tensorboard
            original_grid = make_grid(y[:16].detach().cpu())
            writer.add_image("Ground truth", original_grid, epoch)
    return mean_loss



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default = "Colorize", help="experiment name")
    parser.add_argument("--data_path", type=str, default = "./data/landscapes", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default = 32, help="batch size")
    parser.add_argument("--epochs", type=int, default = 5, help="number of epochs")
    parser.add_argument("--lr", type=float, default = 0.001, help="learning rate")

    args = parser.parse_args()
    exp_name = args.exp_name
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    unet = UNet().to(device)
    loader = get_colorized_dataset_loader(
        path=data_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6
    )


    optimizer = optim.Adam(unet.parameters(), lr=lr)
    writer = SummaryWriter(f"runs/{exp_name}")
    train(unet, optimizer, loader, epochs=epochs, writer=writer)
    # writer.add_graph(unet, input_to_model=next(iter(loader))[0].to(device))

    # Save model weights
    torch.save(unet.state_dict(), "unet.pth")
