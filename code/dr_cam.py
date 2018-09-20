#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import os
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

# if model has LSTM
# torch.backends.cudnn.enabled = False

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + 255.0 * raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

def calc_cam(output_path, image_path, model, transform, arch, topk, cuda):
    results = []
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        # Add your model
    }.get(arch)

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # Model
    model.to(device)
    model.eval()

    # Image
    pil_image = Image.open(image_path)
    #raw_image = cv2.imread(image_path)[..., ::-1]
    #raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    raw_image = transform(pil_image)
    image = raw_image.unsqueeze(0)
    
    raw_image = raw_image.numpy().transpose(1,2,0)
    
    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image.to(device))

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['target_layer'])
        
        result_name = '{}_gcam_{}.png'.format(idx[i].cpu().numpy(), arch)
        save_gradcam(os.path.join(output_path, result_name), output, raw_image)
        results.append(os.path.join(output_path, result_name))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))

    # =========================================================================
    print('Vanilla Backpropagation')
    # =========================================================================
    bp = BackPropagation(model=model)
    probs, idx = bp.forward(image.to(device))

    for i in range(0, topk):
        bp.backward(idx=idx[i])
        output = bp.generate()
        
        result_name = '{}_bp_{}.png'.format(idx[i].cpu().numpy(), arch)
        save_gradient(os.path.join(output_path, result_name), output)
        results.append(os.path.join(output_path, result_name))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))

#    # =========================================================================
#    print('Deconvolution')
#    # =========================================================================
#    deconv = Deconvolution(model=copy.deepcopy(model))  # TODO: remove hook func in advance
#    probs, idx = deconv.forward(image.to(device))
#
#    for i in range(0, topk):
#        deconv.backward(idx=idx[i])
#        output = deconv.generate()
#
#        result_name = '{}_deconv_{}.png'.format(idx[i].cpu().numpy(), arch)
#        save_gradient(os.path.join(output_path, result_name), output)
#        results.append(os.path.join(output_path, result_name))
#        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))

    # =========================================================================
    print('Guided Backpropagation/Guided Grad-CAM')
    # =========================================================================
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image.to(device))

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=CONFIG['target_layer'])

        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        h, w, _ = feature.shape
        region = cv2.resize(region, (w, h))[..., np.newaxis]
        output = feature * region

        result_name = '{}_gbp_{}.png'.format(idx[i].cpu().numpy(), arch)
        save_gradient(os.path.join(output_path, result_name), feature)
        results.append(os.path.join(output_path, result_name))
        result_name = '{}_ggcam_{}.png'.format(idx[i].cpu().numpy(), arch)
        save_gradient(os.path.join(output_path, result_name), output)
        results.append(os.path.join(output_path, result_name))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    return (results, probs)


if __name__ == '__main__':
    from torchvision import models
    import pandas as pd
    scale = 299
    val_transforms = transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(scale),
            transforms.ToTensor()])
    base_image_dir = os.path.join('..')
    output_dir = os.path.join(base_image_dir, 'output')
    train_image_dir = os.path.join(base_image_dir, 'train')
    
    retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(train_image_dir,
                                                             '{}.jpeg'.format(x)))
    
    model_best = torch.load(os.path.join(output_dir, "best_dr"))
    results, y = calc_cam(output_dir, retina_df['path'][0], model_best, val_transforms, 'inception_v3', 3, True)
    print(y)
    import matplotlib.pyplot as plt
    from PIL import Image
    fig = plt.figure(3, figsize=(20, 12))
    for i in range(len(results)):
        pil_im = Image.open(results[i])
        plt.subplot(3,5,i+1)
        plt.imshow(np.asarray(pil_im))
        plt.title(results[i])
    plt.show()