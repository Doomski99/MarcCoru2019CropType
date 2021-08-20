import sys
sys.path.append("../src")
sys.path.append("../src/models")

from models.duplo import DuPLO
import torch
from train import prepare_dataset
from argparse import Namespace
from tqdm.notebook import tqdm
import numpy as np
import sklearn.metrics
import pandas as pd
import os
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from train_duplo import metrics

model_root = "../models/duplo"

def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)

TUM_dataset = Namespace(
    dataset = "BavarianCrops",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    scheme="blocks",
    test_on = "test",
    train_on = "trainvalid",
    samplet = 70
)

GAF_dataset = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    features = "optical",
    scheme="blocks",
    test_on="test",
    train_on="train",
    samplet = 23
)

def setup(dataset, mode, dataroot="../data", store = '/tmp/'):
    
    if mode == "12classes":
        classmapping = os.path.join(dataroot,"BavarianCrops",'classmapping12.csv')
    elif mode == "23classes":
        classmapping = os.path.join(dataroot,"BavarianCrops",'classmapping23.csv')
    
    args = Namespace(batchsize=256,
                 classmapping=classmapping,
                 dataroot=dataroot, dataset=dataset,
                 model='duplo',mode=None,
                 seed=0, store=store, workers=0)

    if dataset == "BavarianCrops":
        args = merge([args,TUM_dataset])
        exp = "isprs_tum_duplo"
    elif dataset == "GAFv2":
        args = merge([args,GAF_dataset])
        exp = "isprs_gaf_duplo"
        
    traindataloader, testdataloader = prepare_dataset(args)
        
    input_dim = traindataloader.dataset.datasets[0].ndims
    nclasses = len(traindataloader.dataset.datasets[0].classes)

    device = torch.device("cuda")
    model = DuPLO(input_dim=input_dim, nclasses=nclasses, sequencelength=args.samplet, dropout=0.4)
    model.load(f"{model_root}/{mode}/{exp}/model.pth")

    model.to(device)

    
    return testdataloader, model

def evaluate(model, dataloader):
    stats = list()
    model.cuda()
    model.eval()
    y_preds = list()
    ys = list()
    ids = list()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            X,y,id = batch
            print(np.shape(X))
            logsoftmax , *_ = model.forward(X.transpose(1,2).cuda())
            y_pred = logsoftmax.argmax(dim=1)
            ys.append(y.cpu().detach().numpy())
            y_preds.append(y_pred.cpu().detach().numpy())
            ids.append(id)
    model.cpu()
    return np.hstack(y_preds), np.vstack(ys)[:,0], np.hstack(ids)

testdataloader, model = setup("BavarianCrops","23classes")
y_pred, y, ids = evaluate(model, testdataloader)
print(sklearn.metrics.classification_report(y,y_pred))
metrics(y,y_pred)