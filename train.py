import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def load_data(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    training_data_transforms = transforms.Compose(
        [
            transforms.RandomRotation(35),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    validate_data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    training_dataset = datasets.ImageFolder(
        train_dir, transform=training_data_transforms
    )
    validate_dataset = datasets.ImageFolder(
        valid_dir, transform=validate_data_transforms
    )

    training_data_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
    validate_data_loader = DataLoader(validate_dataset, batch_size=16, shuffle=False)

    image_datasets = {
        "train": training_dataset,
        "valid": validate_dataset,
    }

    image_dataloaders = {
        "train": training_data_loader,
        "valid": validate_data_loader,
    }

    return (image_datasets, image_dataloaders)


def get_classifier(
    input_size=25088,
    hidden_units=1000,
    output_size=102,  # Because of 102 classes of flowers from our datasets
    dropout=0.2,
):
    layers = []

    layers.append(nn.Linear(input_size, hidden_units))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_units, output_size))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def save_checkpoint(arch, epochs, model, optimiser, save_dir):

    # extract the number of hidden units in each layer
    hidden_layers = []
    for layer in model.classifier:
        if isinstance(layer, torch.nn.Linear):
            hidden_layers.append(layer.out_features)
    
    print(hidden_layers)

    checkpoint = {
        "arch": arch,
        "epochs": epochs,
        "input_size": model.classifier[0].in_features,
        "output_size": 102,
        "hidden_layers": hidden_layers[:-1],
        "m_state_dict": model.state_dict(),
        "o_state_dict": optimiser.state_dict(),
        "class_to_idx": model.class_to_idx,
    }

    torch.save(checkpoint, save_dir + "checkpoint.pth")


def train_network(arch, epochs, learn_rate, hidden_units, has_gpu, data_dir, save_dir):
    device = torch.device("cuda" if has_gpu else "cpu")

    if arch == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif arch == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.classifier = get_classifier(hidden_units=hidden_units)
    optimiser = optim.SGD(model.classifier.parameters(), lr=learn_rate)
    criterion = nn.NLLLoss()

    model.to(device)

    (image_datasets, image_dataloaders) = load_data(data_dir)

    for e in range(epochs):
        running_loss = 0
        model.train()
        torch.cuda.empty_cache()
        for images, labels in image_dataloaders["train"]:
            optimiser.zero_grad()

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            train_loss = criterion(output, labels)
            train_loss.backward()

            optimiser.step()
            running_loss += train_loss.item()

        # Evaluate model after each epoch
        model.eval()
        eval_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in image_dataloaders["valid"]:
                images = images.to(device)
                labels = labels.to(device)

                output = model.forward(images)
                validate_loss = criterion(output, labels)

                eval_loss += validate_loss.item()

                prob = torch.exp(output)
                top_p, top_class = prob.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                output = output.cpu()
                images = images.cpu()
                labels = labels.cpu()

        print(
            f"Epoch {e+1}/{epochs} | "
            f"Training loss: {running_loss/len(image_dataloaders['train']):.3f} | "
            f"Validate loss: {eval_loss/len(image_dataloaders['valid']):.3f} | "
            f"Total Accuracy: {accuracy/len(image_dataloaders['valid']):.3f}"
        )

    model.class_to_idx = image_datasets["train"].class_to_idx

    save_checkpoint(arch, epochs, model, optimiser, save_dir)


def get_args():
    parser = argparse.ArgumentParser(
        description="Training routine for the 102 flowers image classifier A.I."
    )

    # Flags Group
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg",
        help="Specify an architecture for the NN, defaults to VGG16",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.003,
        help="Specify a learning rate for the NN to use, defaults to 0.003",
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Specify the amount of hidden units on the hidden layer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Specify the number of epoch cycles to train the NN",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Specify path to save PyTorch checkpoint.pth file",
    )

    # if --gpu is called args.gpu is True, if --no-gpu is caled args.gpu is False
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Nvidia CUDA to speed up predictions",
    )

    # Params Group
    parser.add_argument(
        "data_dir",
        type=str,
        default="flowers",
        help="Specify path to the dataset directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    train_network(
        arch=args.arch,
        epochs=args.epochs,
        learn_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        has_gpu=args.gpu,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
