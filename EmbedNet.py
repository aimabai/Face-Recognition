#!/usr/bin/python
#-*- coding: utf-8 -*-
#EmbedNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import os

class EmbedNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerClass, **kwargs):
        super(EmbedNet, self).__init__();

        ## __E__ is the embedding model
        EmbedNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        # self.__E__ = EmbedNetModel(**kwargs);
        backbone_kwargs = {'nOut': kwargs.get('nOut', 512),'pretrained': kwargs.get('pretrained', True),  # if you add this arg
    # add other backbone-specific args here if any
        }

        self.__E__ = EmbedNetModel(**backbone_kwargs)
        
        ## __C__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__C__ = LossFunction(**kwargs);

        ## Number of examples per identity per batch
        self.nPerClass = nPerClass

    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__E__.forward(data)

        if label == None:
            return outp

        else:
            outp    = outp.reshape(self.nPerClass,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss = self.__C__.forward(outp,label)
            return nloss


class ModelTrainer(object):

    def __init__(self, embed_model, optimizer, scheduler, **kwargs):

        self.__model__  = embed_model

        ## Optimizer (e.g. Adam or SGD)
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        ## Learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        loss    = 0;

        with tqdm(loader, unit="batch") as tepoch:
        
            for data, label in tepoch:

                tepoch.total = tepoch.__len__()

                data    = data.transpose(1,0)

                ## Reset gradients
                # (write your code here)
                optimizer = self.__optimizer__
                optimizer.zero_grad()
                ## Forward pass and compute loss
                nloss = self.__model__(data.cuda(), label.cuda())
                ## Backward pass
                # (write your code here)
                nloss.backward()
                ## Optimizer step
                # (write your code here)
                optimizer.step()

                ## Keep cumulative statistics
                loss    += nloss.item();
                counter += 1;

                # Print statistics to progress bar
                tepoch.set_postfix(loss=loss/counter)

                if self.lr_step == 'iteration': self.__scheduler__.step()

            if self.lr_step == 'epoch': self.__scheduler__.step()
        
        return (loss/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        
        self.__model__.eval()
        
        # Debug: Check a few embeddings first
        self._debug_embeddings(test_path, transform, test_list)
        
        feats = {}

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, transform=transform, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        print('Generating embeddings')

        ## Extract features for every image
        for data in tqdm(test_loader):
            inp1 = data[0][0].cuda()
            ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        print('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):
            data = line.strip().split(',')

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            ## Find cosine similarity score
            score = F.cosine_similarity(ref_feat, com_feat)

            all_scores.append(score.item())
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + "," + data[2])

        return (all_scores, all_labels, all_trials)

    def _debug_embeddings(self, test_path, transform, test_list):
        """Debug function to check embedding quality"""
        print("=== DEBUG: Checking Embeddings ===")
        
        # Read a few lines from test_list to get sample files
        with open(test_list) as f:
            lines = f.readlines()[:10]  # First 10 pairs
        
        sample_files = set()
        for line in lines:
            parts = line.strip().split(',')
            sample_files.add(parts[1])
            sample_files.add(parts[2])
        
        sample_files = list(sample_files)[:5]  # Take first 5 unique files
        
        with torch.no_grad():
            sample_embeddings = []
            for f in sample_files:
                try:
                    img = Image.open(os.path.join(test_path, f))
                    img_tensor = transform(img).unsqueeze(0).cuda()
                    emb = self.__model__(img_tensor).detach().cpu()
                    sample_embeddings.append(emb)
                    print(f"Debug: {f} - embedding norm: {torch.norm(emb):.4f}")
                except Exception as e:
                    print(f"Debug: Error loading {f}: {e}")
                    continue
            
            if len(sample_embeddings) >= 2:
                # Check if embeddings are identical
                for i in range(1, len(sample_embeddings)):
                    similarity = F.cosine_similarity(sample_embeddings[0], sample_embeddings[i])
                    print(f"Debug: Similarity between sample 0 and {i}: {similarity.item():.4f}")
        
        print("=== DEBUG: End Embedding Check ===")


    # def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        
    #     self.__model__.eval();   
            
    #     feats       = {}

    #     ## Read all lines
    #     with open(test_list) as f:
    #         lines = f.readlines()

    #     ## Get a list of unique file names
    #     files = sum([x.strip().split(',')[-2:] for x in lines],[])
    #     setfiles = list(set(files))
    #     setfiles.sort()
             
    #     ## Define test data loader
    #     test_dataset = test_dataset_loader(setfiles, test_path, transform=transform, num_eval=num_eval, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=nDataLoaderThread,
    #         drop_last=False,
    #     )

    #     print('Generating embeddings')

    #     ## Extract features for every image
    #     for data in tqdm(test_loader):
    #         inp1                = data[0][0].cuda()
    #         ref_feat            = self.__model__(inp1).detach().cpu()
    #         feats[data[1][0]]   = ref_feat

    #     all_scores = [];
    #     all_labels = [];
    #     all_trials = []

    #     print('Computing similarities')

    #     ## Read files and compute all scores
    #     for line in tqdm(lines):

    #         data = line.strip().split(',');

    #         ref_feat = feats[data[1]]
    #         com_feat = feats[data[2]]

    #         ## Find cosine similarity score
    #         score = F.cosine_similarity(ref_feat, com_feat);  #write your code here

    #         all_scores.append(score.item());  
    #         all_labels.append(int(data[0]));
    #         all_trials.append(data[1] + "," + data[2])

    #     return (all_scores, all_labels, all_trials)


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                if name not in self_state:
                    print(f'{origname} is not in the model.');
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}");
                continue;

            self_state[name].copy_(param);

