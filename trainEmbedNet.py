#!/usr/bin/python
# -*- coding: utf-8 -*-
# trainEmbedNet.py

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Face Recognition Training");

## Data loader
parser.add_argument('--batch_size',         type=int, default=100,	help='Batch size, defined as the number of classes per batch');
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch');
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of data loader threads');

## Training details
parser.add_argument('--test_interval',  type=int,   default=2,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=10,     help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="adaface",  help='Loss function to use');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='Optimizer');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Initial learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument('--margin',         type=float, default=0.4,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=64,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=9500,   help='Number of classes in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights, otherwise initialise with random weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and evaluation data
parser.add_argument('--train_path',     type=str,   default="data/train",   help='Absolute path to the train set');
parser.add_argument('--train_ext',      type=str,   default="jpg",  help='Training files extension');
parser.add_argument('--test_path',      type=str,   default="data/val",     help='Absolute path to the test set');
parser.add_argument('--test_list',      type=str,   default="data/val_pairs.csv",   help='Evaluation list');

## Model definition
parser.add_argument('--model',          type=str,   default="ImprovedResNet", help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true',   help='Eval only')
parser.add_argument('--output',         type=str,   default="",     help='Save a log of output to this file name');

## Training
parser.add_argument('--gpu',            type=int,   default=9,      help='GPU index');


args = parser.parse_args();

    
## ===== ===== ===== ===== ===== ===== ===== =====
## Script to compute EER
## ===== ===== ===== ===== ===== ===== ===== =====

# def compute_eer(all_labels,all_scores):

#     # compute receiver operating characteristic (ROC) for binary classification
#     # (write your code here)
#     fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
#     # calculate false negative rate (FNR)
#     # (write your code here)
#     fnr = 1 - tpr
#     # calculate equal error rate (EER). The EER is the error rate at which FNR is equal to FPR.
#     # (write your code here)
#     eer_threshold = thresholds[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
#     EER = fpr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
#     return EER
def compute_eer(all_labels, all_scores):
    print(f"Debug: Total pairs: {len(all_labels)}")
    print(f"Debug: Genuine pairs: {sum(all_labels)}")
    print(f"Debug: Impostor pairs: {len(all_labels) - sum(all_labels)}")
    
    # Check score distribution
    genuine_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    impostor_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
    
    print(f"Debug: Genuine scores - min: {min(genuine_scores):.4f}, max: {max(genuine_scores):.4f}, mean: {numpy.mean(genuine_scores):.4f}")
    print(f"Debug: Impostor scores - min: {min(impostor_scores):.4f}, max: {max(impostor_scores):.4f}, mean: {numpy.mean(impostor_scores):.4f}")
    
    # Check if scores are constant
    if len(set(all_scores)) == 1:
        print("WARNING: All scores are identical!")
        return 0.5  # Worst-case EER
    
    # Original EER computation
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR ≈ FNR
    eer_index = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer_threshold = thresholds[eer_index]
    EER = fpr[eer_index]
    
    print(f"Debug: EER threshold: {eer_threshold:.4f}")
    print(f"Debug: FPR at EER: {fpr[eer_index]:.4f}")
    print(f"Debug: FNR at EER: {fnr[eer_index]:.4f}")
    
    return EER

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path+"/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s', 
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Load models
    model = EmbedNet(**vars(args)).cuda();

    ep          = 1
        ## ===== AUTO-DETECT AUGMENTATION BASED ON DATASET =====
    if 'train1' in args.train_path:
        # Use minimal augmentation for large Caucasian dataset
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Using MINIMAL augmentation for train1 (large dataset)")
    else:
        # Use enhanced augmentation for small Korean dataset  
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
        print("Using ENHANCED augmentation for train2 (small dataset)")
    
    ## Test transform (same for both)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ## ===== END AUGMENTATION =====
    
    
    # from augmentation import get_enhanced_transforms, get_minimal_transforms
    # train_transform, test_transform = get_enhanced_transforms()
    
    # ## Input transformations for training (you can change if you like)
    # train_transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Resize(256),
    #      transforms.RandomCrop([224,224]),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # ## Input transformations for evaluation (you can change if you like)
    # test_transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Resize(256),
    #      transforms.CenterCrop([224,224]),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **vars(args));
    trainer     = ModelTrainer(model, **vars(args))

    ## Load model weights
    modelfiles = glob.glob(f'{args.save_path}/epoch0*.model')
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print(f"Model {modelfiles[-1]} loaded from previous state!");
        ep = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print(f"Model {args.initial_model} loaded!");

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,ep):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in model.__E__.parameters())
    print(f'Total model parameters: {pytorch_total_params:,}')
    
    ## Evaluation code 
    if args.eval == True:

        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
        
        EER = compute_eer(lab, sc)

        print(f'EER {EER*100:.2f}%')

        if args.output != '':
            with open(args.output,'w') as f:
                for ii in range(len(sc)):
                    f.write(f'{sc[ii]:.4f},{lab[ii]:d},{trials[ii]}\n')

        quit();

    ## Log arguments
    logger.info(f'{args}')

    ## Core training script
    for ep in range(ep,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        logger.info(f"Epoch {ep:04d} started with LR {max(clr):.5f} ");
        loss = trainer.train_network(trainLoader);
        logger.info(f"Epoch {ep:04d} completed with TLOSS {loss:.5f}");

        if ep % args.test_interval == 0:
            
            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
            EER = compute_eer(lab, sc)

            logger.info(f"Epoch {ep:04d}, Val EER {EER*100:.2f}%");
            trainer.saveParameters(args.save_path+f"/epoch{ep:04d}.model");

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    os.environ["CUDA_VISIBLE_DEVICES"]=f'{args.gpu}'
            
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)


if __name__ == '__main__':
    main()