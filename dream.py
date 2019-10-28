#!/usr/bin/env python3

import os
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.nn.functional import interpolate

logging.basicConfig(level=logging.INFO,
                    format='\n%(message)s\n')
logging.getLogger().setLevel(logging.INFO)

class DeepDream(torch.nn.Module):

    def __init__(self, img_name, model, layer, octaves, img_size, jitter, use_mean_loss):
        super(DeepDream, self).__init__()
        self.img_name = img_name
        self.img_size = img_size
        self.octaves = octaves
        self.jitter = jitter
        self.use_mean_loss = use_mean_loss
        self.octave_img_size = int(self.img_size / self.octaves)
        if model == 'vgg':
          self.net = models.vgg19(pretrained=True).eval()
          self.net.features[int(layer)].register_forward_hook(self.hook)
        elif model == 'googlenet':
          self.net = models.googlenet(pretrained=True).eval()
          getattr(self.net, layer).register_forward_hook(self.hook)
        for p in self.net.parameters():
            p.requires_grad = False
        if img_name:
            img = Image.open(self.img_name)
            self.img = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor()
              ])(img).unsqueeze_(0).requires_grad_(False)
        else:
            self.img = torch.tensor(
                torch.rand(1, 3, self.img_size, self.img_size)
                ).requires_grad_(False)
        self.octave_img = interpolate(self.img, size=int(self.octave_img_size)).clone().detach().requires_grad_(True)

    def update_octave(self):
        # Downsample original image
        downscaled_img = interpolate(self.img, size=(self.octave_img_size, self.octave_img_size)).clone().detach().requires_grad_(False)
        # Update octave img size to next octave
        self.octave_img_size += int(self.img_size / self.octaves)
        # Upscale the resized original img
        downscaled_upscaled_img = interpolate(downscaled_img, size=(self.octave_img_size,self.octave_img_size)).clone().detach().requires_grad_(False)
        # Downsample original image to new octave size
        downscaled_img_new_octave = interpolate(self.img, size=(self.octave_img_size,self.octave_img_size)).clone().detach().requires_grad_(False)
        # Estimate lost detail
        lost_detail = downscaled_img_new_octave - downscaled_upscaled_img
        # Upscale octave image (detail is lost in this interpolation)
        resized_octave_img = interpolate(self.octave_img, size=
              (self.octave_img_size,self.octave_img_size))
        # Re-add lost details
        self.octave_img = (resized_octave_img.add_(lost_detail)).clone().detach().requires_grad_(True)
        return

    def hook(self, module, input, output):
        self.feature_output = output
        return

    def forward(self):
        if self.jitter:
            # Apply image jitter
            self.j_x, self.j_y = np.random.randint(0, self.jitter, 2)
            self.octave_img = torch.roll(self.octave_img, shifts=(self.j_x, self.j_y), dims=(2,3)).clone().detach().requires_grad_(True)
        _ = self.net(self.octave_img)
        if self.use_mean_loss:
            loss = self.feature_output.mean()
        else:
            loss = self.feature_output.pow(2).sum().sqrt()
        return loss

    def backward(self, loss):
        loss.backward()
        return

def get_layers(model, single_layer):
    layers = {
            'vgg': range(0,37,3)[1:],
            'googlenet': [
                        'conv1',
                        'conv2',
                        'conv3',
                        'inception3a',
                        'inception3b',
                        'inception4a',
                        'inception4b',
                        'inception4c',
                        'inception4d',
                        'inception4e',
                        'inception5a',
                        'inception5b',
                        ]
                }
    if single_layer:
        return [single_layer]
    return layers[model]


def main(**kwargs):
    img_name = kwargs.get('img_name')
    model = kwargs.get('model')
    steps = kwargs.get('steps')
    octaves = kwargs.get('octaves')
    single_layer = kwargs.get('single_layer')
    img_size = kwargs.get('img_size')
    jitter = kwargs.get('jitter')
    use_mean_loss = kwargs.get('use_mean_loss')

    learning_rate = 1e-2

    # Check that image size can be divided evenly by octaves
    while img_size % octaves > 0:
        octaves -= 1
        if octaves == 1:
            break

    results = torch.Tensor().requires_grad_(False)
    logging.info(f'Starting to process image {img_name}...')
    for layer in get_layers(model, single_layer):
        logging.info(f'Processing layer {layer} ...')
        dd = DeepDream(img_name, model, layer, octaves, img_size, jitter, use_mean_loss)
        for octave in tqdm(range(octaves-1)):
            for step in range(steps):
                loss = dd.forward()
                dd.backward(loss)
                with torch.no_grad():
                    norm_grad = dd.octave_img.grad / (dd.octave_img.grad.abs().mean() + 1e-12)
                    dd.octave_img += learning_rate  * norm_grad
                    dd.octave_img.grad.zero_()
                    if dd.jitter:
                        # Unapply image jitter
                        neg_jitter = (-dd.j_x, -dd.j_y)
                        dd.octave_img = torch.roll(dd.octave_img, shifts=neg_jitter, dims=(2,3)).clone().detach().requires_grad_(True)
            with torch.no_grad():
                oct_img = dd.update_octave()
        if layer is single_layer or not single_layer:
            results = torch.cat((results, dd.octave_img))
    fname = f'{os.path.splitext(img_name)[0]}_{model}_{steps}_{octaves}_results.jpg'
    save_image(results, fname, nrow=4)
    logging.info(f'Done! New image saved at {fname}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_name", type=str,
                        default=None, help="Input image filename. If empty, run on random noise")
    parser.add_argument("-m", "--model", type=str, choices=['googlenet', 'vgg'],
                        default="googlenet", help="Model to use. Default: googlenet")
    parser.add_argument("-g", "--steps", type=int,
                        default=10, help="Gradient descent steps per octave. Default: 10")
    parser.add_argument("-o", "--octaves", type=int,
                        default=4, help="Number of octaves. Default: 4")
    parser.add_argument("-s", "--img_size", type=int,
                        default=1600, help="Size of output image. Default: 1600")
    parser.add_argument("-j", "--jitter", type=int,
                        default=10, help="Max pixel jitter. If 0, no jitter is applied. Default: 10")
    parser.add_argument("-l", "--single_layer", type=str,
                        default=None, help="Specify single layer to run model for. If empty, \
run for all layers (googlenet) or once every 3 layers (vgg) and save results as an image grid. Default: None")
    parser.add_argument("-y", "--use_mean_loss", type=bool,
                        default=False, help="If True, use  \
mean of layer activations as loss function. If False, use L2 norm of layer activations. Default: False")
    args = parser.parse_args()

    kwargs = dict(
        img_name = args.img_name,
        model = args.model,
        steps = args.steps,
        octaves = args.octaves,
        single_layer = args.single_layer,
        img_size = args.img_size,
        jitter = args.jitter,
        use_mean_loss = args.use_mean_loss
    )

    if args.img_name and not os.path.exists(args.img_name):
        raise Exception(f'Image {args.img_name} not found. Please enter a valid image path')

    if args.single_layer and args.single_layer not in [str(l) for l in get_layers(args.model, None)]:
        raise Exception(f'Invalid layer in --single_layer. Please choose one of the following: \n \
        {get_layers(args.model, None)}')

    main(**kwargs)