import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib.lines import Line2D

from models import *
from misc_functions import *

import matplotlib
matplotlib.use("TkAgg")

def plot_grad_flow(named_params):
    ave_grads = []
    layers = []
    for n, p in named_params:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth = 1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin = 0, xmax = len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("avg gradient")
    plt.title("grad flow")
    plt.grid(True)
    plt.show()

def plot_grad_flow_v2(named_params):
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_params:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha = 0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top = 0.02)
    plt.xlabel("layers")
    plt.ylabel("ave grads")
    plt.title("grad flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ["max-gradient", "mean-gradient", "zero-gradient"])
    plt.show()

class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        layer_outputs = []
        output = []

        conv_output = None
        x = x.float()

        for i, (mdef, module) in enumerate(zip(self.model.module_defs, self.model.module_list)):
            mtype = mdef["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)
            elif mtype == "route":
                layers = [int(x) for x in mdef["layers"].split(",")]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == "shortcut":
                x = x + layer_outputs[int(mdef["from"])]
            elif mtype == "yolo":
                x = module(x, (416, 416))
                output.append(x)
            layer_outputs.append(x if i in self.model.routs else [])

            if i == self.target_layer:
                x.register_hook(self.save_gradient)
                #x.register_hook(lambda grad: print(grad))
                conv_output = x

        return conv_output, x[1]

    def forward_pass(self, x):
        # forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # flatten

        # forward pass on the classifier
        return conv_output, x

class GradCam():
    """
    produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.to("cuda:0").eval()

        # define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        # target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # zero grads
        self.model.zero_grad()

        # backward pass with specified target
        model_output.backward(gradient = one_hot_output, retain_graph = True)
        plot_grad_flow(self.model.named_parameters())
        plot_grad_flow_v2(self.model.named_parameters())

        # get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # get convolution outputs
        target = conv_output.cpu().data.numpy()[0]
        # get weights from gradients
        weights = np.mean(guided_gradients, axis = (1, 2))  # take averages for each gradient

        # create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype = np.float32)

        # multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
        cam = np.uint8(cam * 255)  # scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                   input_image.shape[3]), Image.ANTIALIAS))/255

        return cam

if __name__ == '__main__':
    original_image, prep_img, target_class, file_name_to_export = params_for_yolo()

    model = Darknet("cfg/yolov3_transfer.cfg", 416)
    model.load_state_dict(torch.load("weights/yoloweights.pt", map_location = "cuda:0")["model"])

    prep_img = torch.from_numpy(prep_img).to("cuda:0")

    if prep_img.ndimension() == 3:
        prep_img = prep_img.unsqueeze(0)

    grad_cam = GradCam(model, target_layer = 42)
    cam = grad_cam.generate_cam(prep_img, target_class)
    original_image = Image.fromarray(original_image.astype("uint8"))
    save_class_activation_images(original_image, cam, file_name_to_export)

    print("GradCAM completed")
