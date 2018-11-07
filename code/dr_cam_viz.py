#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import time
import os
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from diabeticretinopathy.code.grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

import matplotlib.pyplot as plt
# if model has LSTM
# torch.backends.cudnn.enabled = False

def find_gradient(data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    return np.uint8(data)


def find_gradcam(gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + 255.0*raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    #RGB_gcam = cv2.cvtColor(np.uint8(gcam), cv2.COLOR_BGR2RGB)
    return np.uint8(gcam)


def gradcam(model, device, raw_image, image, CONFIG, topk):
    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    y, ind = gcam.forward(image.to(device))
    results = []
#    for i in range(0, topk):
#        gcam.backward(idx=idx[i])
#        output = gcam.generate(target_layer=CONFIG['target_layer'])
#        
#        results.append(find_gradcam(output, raw_image))
#        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    probs = y.cpu().data.numpy()
    idx = ind.cpu().data.numpy()
    del gcam, y, ind
    torch.cuda.empty_cache()
    return (results, probs, idx)

def vanilla_backprop(model, device, raw_image, image, CONFIG, topk):
    # =========================================================================
    print('Vanilla Backpropagation')
    # =========================================================================
    bp = BackPropagation(model=model)
    probs, idx = bp.forward(image.to(device))
    results = []
    for i in range(0, topk):
        bp.backward(idx=idx[i])
        output = bp.generate()
                
        results.append(find_gradient(output))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    return (results, probs, idx)


def deconv(model, device, raw_image, image, CONFIG, topk):
    # =========================================================================
    print('Deconvolution')
    # =========================================================================
    deconv = Deconvolution(model=model)  # TODO: remove hook func in advance
    probs, idx = deconv.forward(image.to(device))

    for i in range(0, topk):
        deconv.backward(idx=idx[i])
        output = deconv.generate()

        results.append(find_gradient(output))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    return (results, probs, idx)


def guided_backprop(model, device, raw_image, image, CONFIG, topk):
    # =========================================================================
    print('Guided Backpropagation')
    # =========================================================================
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image.to(device))

    for i in range(0, topk):
        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        results.append(find_gradient(feature))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    return (results, probs, idx)


def guided_gradcam(model, device, raw_image, image, CONFIG, topk):
    # =========================================================================
    print('Guided Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
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
        
        results.append(find_gradient(output))
        print('[{:.5f}] {}'.format(probs[i], idx[i].cpu().numpy()))
    return (results, probs, idx)


def calc_cam(image_path, model, transform, arch, topk, cam_type='gradcam', cuda=False):
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
    raw_image = transform(pil_image)
    image = raw_image.unsqueeze(0)
    raw_image = raw_image.numpy().transpose(1,2,0)
    
    cam_method = {'gradcam' : gradcam,
                'vanilla' : vanilla_backprop,
                'deconv' : deconv,
                'guided-bp' : guided_backprop,
                'guided-gradcam' : guided_gradcam,
                }
    #import pdb; pdb.set_trace()
    results, y, idx = cam_method[cam_type.lower()](model, device, raw_image, image, CONFIG, topk)
    probs = y.cpu().data.numpy().tolist()
    idx = idx.cpu().data.numpy().tolist()
    del image, raw_image, y
    torch.cuda.empty_cache()
    return (results, probs, idx)


def find_pred_one(image_folder, image_name):
    start_time = time.time()
    from diabeticretinopathy.code.retrain import get_transformations
    _, val_transforms = get_transformations()
    print(image_folder, image_name)
    image_path = os.path.join(image_folder, image_name)
    save_name = os.path.splitext(image_name)[0]
    #import pdb; pdb.set_trace()
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dr_model = DRSeverePred.getInstance(os.path.join(base_dir,"best_dr"))
    results, probs, idx = dr_model.calc_cam(image_path, val_transforms, 'inception_v3', 5, 'gradcam')

    

    for i in range(len(results)):
        #import pdb; pdb.set_trace()
        heatmap_name = 'analysis/' + save_name + '_' + str(idx[i]) + '.jpeg'
        heatmap_path = os.path.join(image_folder, heatmap_name)
        if not os.path.exists(os.path.dirname(heatmap_path)):
            os.makedirs(os.path.dirname(heatmap_path))
        if not os.path.isfile(heatmap_path):
            cv2.imwrite(heatmap_path, results[i])
    
    del results, idx, dr_model
    torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    print('Time: {:.1f}'.format(elapsed_time))
    return ('analysis/' + save_name, probs)


class DRSeverePred:
    # Here will be the instance stored.
    __instance = None
    __CONFIG = {
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
    }
    __device = 0
    __model = None
    __gcam = None
    
    @staticmethod
    def getInstance(model_name = None):
        """ Static access method. """
        if DRSeverePred.__instance == None:
            DRSeverePred(model_name)
        return DRSeverePred.__instance 

    def __init__(self, model_name = None):
        """ Virtually private constructor. """
        if DRSeverePred.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DRSeverePred.__instance = self
            DRSeverePred.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print('Running on the GPU:', torch.cuda.get_device_name(current_device))
        else:
            print('Running on the CPU')
        # Model
        DRSeverePred.__model = torch.load(model_name)
        DRSeverePred.__model.to(DRSeverePred.__device)
        DRSeverePred.__model.eval()
        DRSeverePred.__gcam = GradCAM(model=DRSeverePred.__model)
        

    def calc_cam(self, image_path, transform, arch, topk, cam_type='gradcam'):
        #results = []
        myconfig = DRSeverePred.__CONFIG.get(arch)
        
        # Image
        pil_image = Image.open(image_path)
        raw_image = transform(pil_image)
        image = raw_image.unsqueeze(0)
        raw_image = raw_image.numpy().transpose(1,2,0)
        #import pdb; pdb.set_trace()
        cam_method = {'gradcam' : gradcam,
                    'vanilla' : vanilla_backprop,
                    'deconv' : deconv,
                    'guided-bp' : guided_backprop,
                    'guided-gradcam' : guided_gradcam,
                    }
        #import pdb; pdb.set_trace()
        # =========================================================================
        print('Grad-CAM')
        # =========================================================================
        probs, idx = DRSeverePred.__gcam.forward(image.to(DRSeverePred.__device))
        probs = probs.cpu().data.numpy()
        idx = idx.cpu().data.numpy()
        results = []
        for i in range(0, topk):
            DRSeverePred.__gcam.backward(idx=idx[i])
            output = DRSeverePred.__gcam.generate(target_layer=myconfig['target_layer'])
            
            results.append(find_gradcam(output, raw_image))
            print('[{:.5f}] {}'.format(probs[i], idx[i]))
        
#        results, probs, idx = cam_method[cam_type.lower()](DRSeverePred.__model, DRSeverePred.__device, raw_image, image, myconfig, topk)
#        probs = DRSeverePred.__model(image.to(DRSeverePred.__device))
#        probs = probs.cpu().data.numpy()
#        probs = np.exp(probs)
#        probs = probs / probs.max()
#        idx = np.array([0,1,2,3,4])
        del image, raw_image
        torch.cuda.empty_cache()
        return (results, probs, idx)



if __name__ == '__main__':
    from torchvision import models
    import pandas as pd
    scale = 299
    from retrain import get_transformations
    _, val_transforms = get_transformations()
    base_image_dir = os.path.join('..')
    output_dir = os.path.join(base_image_dir, 'output')
    train_image_dir = os.path.join(base_image_dir, 'train')
    
    retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(train_image_dir,
                                                             '{}.jpeg'.format(x)))
    
    dr_model = DRSeverePred.getInstance(os.path.join(output_dir, "best_dr"))
    image_name = retina_df['path'][0]
    results, y, idx = dr_model.calc_cam(image_name, val_transforms, 'inception_v3', 3, 'gradcam')
    
    print(y)
    
    fig = plt.figure(4, figsize=(20, 4))
    pil_im = Image.open(image_name)
    plt.subplot(1,4,1)
    plt.imshow(np.asarray(pil_im))
    plt.title('original')
    for i in range(len(results)):
        plt.subplot(1,4,i+2)
        RGB_gcam = cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_gcam)
        plt.title(i)
    plt.show()