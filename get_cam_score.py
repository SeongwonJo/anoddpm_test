import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet

from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import densenet_1ch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--image_path',
        type=str,
        default='./examples/',
        help='Input image folder path')

    parser.add_argument(
        '-p',
        '--pt_path',
        type=str,
        default='./cxr_densenet_n.pt',
        help='Input pt path')

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='densenet121',
        help='Select model')

    arguments = parser.parse_args()

    return arguments


def make_cam_score(arguments, method, target_img):
    use_cuda = True if torch.cuda.is_available() else False

    if use_cuda:
        print('\nUsing GPU for acceleration')
    else:
        print('\nUsing CPU for computation')

    methods = \
        {"gradcam": GradCAM,
         "gradcam++": GradCAMPlusPlus,
}

    models = {
        "resnet50": resnet.resnet50(num_classes=2),
        "resnet101": resnet.resnet101(num_classes=2),
        "resnet152": resnet.resnet152(num_classes=2),
        "densenet121": densenet_1ch.densenet121(num_classes=2),
        "densenet201": densenet_1ch.densenet201(num_classes=2),
        "densenet121_2048": densenet_1ch.densenet121_2048(num_classes=2),

    }

    model = models[arguments["model"]]

    # model = resnet.resnet50(pretrained=True)
    # model = torch.load("../Test_bench/cxr_densenet.pt")

    if arguments["model"][0:6] == "resnet":
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    # load pt
    pt = torch.load(arguments["pt_path"])
    model.load_state_dict(pt['model_state_dict'])

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # print(model)

    target_layers = []
    if arguments["model"][0:6] == "resnet":
        target_layers = [model.layer4]
    elif arguments["model"][0:8] == "densenet":
        target_layers = [model.features.denseblock4]

    # img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # img = cv2.imread(target_img, 0)
    img = target_img.clone().detach()
    print("\n get image to cam score", img.shape)
    img = img / 255
    preprocessing = Compose([
        # ToTensor(),
        Normalize(mean=[0.5, ], std=[0.5, ])
    ])
    input_tensor = preprocessing(img)

    print("\n after preprocess",input_tensor.shape)

    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    # targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None,)

        # Here grayscale_cam has only one image in the batch
        # grayscale_cam = grayscale_cam[0, :]
        # print("\n", grayscale_cam)
        # print(grayscale_cam.max())
        # print(grayscale_cam.shape)
        result = torch.tensor(grayscale_cam).unsqueeze(1)
        print("\n result", result.shape)
        # print(result.shape)

    return result


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both

    'pt_path' example : "./cxr_densenet_n.pt"

    """

    args = get_args()

    make_cam_score(args, target_img=args.image_path)
