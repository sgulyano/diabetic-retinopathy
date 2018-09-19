# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 08:52:09 2018

@author: sguly
"""

import time

import os
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    if use_gpu:
        model.cuda()

    for epoch in range(num_epochs):
        start_time = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                #import pdb; pdb.set_trace()
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]
                
            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "current_state")

        elapsed_time = time.time() - start_time
        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
              'valid loss: {:.4f} acc: {:.4f}   [Time: {:.1f}]'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc, 
                valid_epoch_loss, valid_epoch_acc, elapsed_time))
    
    elapsed_time = time.time() - since
    print('Best val Acc: {:4f}, Total elapsed time: {:.1f}'.format(best_acc, elapsed_time))

    model.load_state_dict(best_model_wts)
    return model


def eval_model(model, dataloader, dataset_size, criteria):
    since = time.time()
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model.cuda()
    
    model.train(False)

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        
        _, preds = torch.max(outputs.data, 1)
        loss = criteria(outputs, labels)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
    
    
    test_epoch_loss = running_loss / dataset_size
    test_epoch_acc = running_corrects / dataset_size
    elapsed_time = time.time() - since
    print('Test loss: {:.4f} acc: {:.4f}, Total elapsed time: {:.1f} '.format(
            test_epoch_loss, test_epoch_acc, elapsed_time))
    return (test_epoch_loss, test_epoch_acc)
