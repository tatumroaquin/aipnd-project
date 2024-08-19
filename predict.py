import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import models, transforms


def process_image(image_path):
    image = Image.open(image_path)

    if image.size[0] < image.size[1]:
        image.thumbnail((256, image.size[1] * 256 // image.size[0]))
    else:
        image.thumbnail((image.size[0] * 256 // image.size[1], 256))

    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    scaled_image = np.array(image) / 255.0

    normalised_mean = np.array([0.485, 0.456, 0.406])
    normalised_stdev = np.array([0.229, 0.224, 0.225])

    scaled_image = (scaled_image - normalised_mean) / normalised_stdev

    scaled_image = scaled_image.transpose((2, 0, 1))

    return scaled_image


def get_prediction(model, image_path, has_gpu, top_k):
    device = torch.device("cuda" if has_gpu else "cpu")

    model.eval()
    model.to(device)

    image = process_image(image_path)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.forward(image)

        prob = torch.exp(outputs)
        top_ps, top_idx = prob.topk(top_k, dim=1)

        top_ps = top_ps.cpu().squeeze().numpy()
        top_idx = top_idx.cpu().squeeze().numpy()

        if top_k == 1:
            top_ps = [top_ps]
            top_idx = [top_idx]

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[int(idx)] for idx in top_idx]

    return top_ps, top_classes


# def get_classifier(
#     input_size=25088,
#     hidden_layers=[512, 256],
#     output_size=102,
#     dropout=0.2,
# ):
#     layers = []

#     # first layer
#     layers.append(nn.Linear(input_size, hidden_layers[0]))

#     # append each subsequent layers after that
#     for i in range(len(hidden_layers) - 1):
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(p=dropout))
#         layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

#     layers.append(nn.ReLU())
#     layers.append(nn.Dropout(p=dropout))
#     layers.append(nn.Linear(hidden_layers[-1], output_size))

#     layers.append(nn.LogSoftmax(dim=1))

#     return nn.Sequential(*layers)


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


# Default args flag defaults to checkpoint.pth
def load_checkpoint(pthfile):
    checkpoint = torch.load(pthfile, weights_only=False)

    if checkpoint["arch"] == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif checkpoint["arch"] == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    if checkpoint["arch"] == "vgg":
        model.classifier = get_classifier(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            hidden_units=checkpoint["hidden_layers"][0],
            # dropout = 0.2
        )
    elif checkpoint["arch"] == "resnet":
        model.fc = get_classifier(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            hidden_units=checkpoint["hidden_layers"][0],
            # dropout = 0.2
        )

    model.load_state_dict(checkpoint["m_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    # === Only Necessary for Training === #
    # optimiser = optim.SGD(model.classifier.parameters(), lr=0.003)
    # optimiser.load_state_dict(checkpoint["o_state_dict"])

    # return (optimiser, model)

    return model


def get_args():
    # Initialise Parser
    parser = argparse.ArgumentParser(
        description="Image classifier AI for 102 Flowers Database"
    )

    # Flags Group
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,  # flag ommitted print only one prediction
        help="Get Top-K most likely classes for an image",
    )

    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="JSON file which maps category numbers to class names",
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
        "imgfile",
        type=str,
        help="Specifies the image file which the network attempts to classify",
    )

    parser.add_argument(
        "pthfile",
        type=str,
        default="checkpoint.pth",
        help="PyTorch checkpoint file which contains pretrained weights",
    )

    return parser.parse_args()


def load_json(jsonfile):
    try:
        with open(jsonfile, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON ERR: The file {jsonfile} was not found.")
    except json.JSONDecodeError:
        print(f"JSON ERR: The file {jsonfile} could not be decoded as JSON.")


if __name__ == "__main__":
    args = get_args()

    model = load_checkpoint(args.pthfile)

    # Clear the CUDA cache if PC has a GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.category_names:
        categories = load_json(args.category_names)

    # Probabilities and Classes
    top_ps, top_classes = get_prediction(
        model=model, image_path=args.imgfile, top_k=args.top_k, has_gpu=args.gpu
    )

    for p, c in zip(top_ps, top_classes):
        class_name = categories[c]
        print(f"Probability: {p:.3f} | Classified As: {class_name}")

    # if args.category_names:
    #     cat_names = load_json(args.category_names)
