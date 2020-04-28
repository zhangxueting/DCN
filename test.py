import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import config
import scipy as sp
import scipy.stats
from task_generator import TaskGenerator,split_fine_grained_dataset,mini_imagenet_folder
from network import DCN,EmbeddingSENet

parser = argparse.ArgumentParser(description="RelationNet2/DCN for Few-Shot Learning")
parser.add_argument("--way",type = int, default = 5)  # num_class
parser.add_argument("--shot",type = int, default = 5) # num_support_per_class
parser.add_argument("--query",type = int, default = 15) # num_query_per_class
parser.add_argument("--embedding_class",type =int, default = 80) # num_class for embedding pre-training
parser.add_argument("--test_episode",type=int,default=600)
parser.add_argument("--model_episode",type=int,default=100000) # model saved at 100000 episode
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("--dataset",type=str,default="miniimagenet") # tieredimagenet,cub,car,aircraft
parser.add_argument("--valid_set",type=int,default=1) # 1: use valid set for training,  0: not use valid set
parser.add_argument("--variational",type=int,default=1) # 1: variational version 0: standard version
parser.add_argument("--multi_try",type=int,default=1) # multi try for variatioal version's test
parser.add_argument("--loss", type=str, default='Entropy')
parser.add_argument("--weight_or_not", type=str,default='weight') # to distinct "weight" or "noweight"
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def test(dcn,task_generator):
    print ('test process: ')
    accuracies = []

    with torch.no_grad():
        dcn.eval()

        for test_episode in range(args.test_episode):
            total_rewards = 0
            # init dataset
            support_x,support_y,query_x,query_y = task_generator.sample_task(args.way,args.shot,args.query,type="meta_test")
            # calculate features
            support_x = support_x.to(device)

            query_x = query_x.to(device)


            if args.variational == 1:
                query_predict_ys = 0
                for i in range(args.multi_try):
                    score1,score2,score3,score4 = dcn(support_x,query_x)
                    score1 = score1.view(-1,args.way)
                    score2 = score2.view(-1,args.way)
                    score3 = score3.view(-1,args.way)
                    score4 = score4.view(-1,args.way)
                    query_predict_ys += score1 + score2 + score3 + score4

                query_predict_y = query_predict_ys/args.multi_try
            else:
                score1,score2,score3,score4 = dcn(support_x,query_x)
                score1 = score1.view(-1,args.way)
                score2 = score2.view(-1,args.way)
                score3 = score3.view(-1,args.way)
                score4 = score4.view(-1,args.way)
                query_predict_y = score1 + score2 + score3 + score4

            _, predict_labels = torch.max(query_predict_y.data, 1) # find the label of max similarity score

            total_rewards = predict_labels.eq(query_y.to(device)).sum().item()

            accuracy = total_rewards/1.0/args.query/args.way
            accuracies.append(accuracy)
            print("episode:",test_episode,"acc:",accuracy)

    test_accuracy,h = mean_confidence_interval(accuracies)

    print("test accuracy:",test_accuracy,"h:",h)

def main():

    # Step 1: init data folders
    print("init dataset")

    torch.cuda.manual_seed_all(1)

    if args.dataset == 'miniimagenet':
        if args.valid_set == 1:
            metatrain_folder,metatest_folder = mini_imagenet_folder(config.miniimagenet_trainvalfolder,
                                                                config.miniimagenet_testfolder)
        else:
            metatrain_folder,metatest_folder = mini_imagenet_folder(config.miniimagenet_trainfolder,
                                                                config.miniimagenet_testfolder)
        task_generator = TaskGenerator(metatrain_folder,metatest_folder)

    elif args.dataset == 'tieredimagenet':
        pass

    elif args.dataset == 'cub':
        pass

    elif args.dataset == 'car':
        pass

    elif args.dataset == 'aircraft':
        pass

    # step 2: init neural networks
    print ('init neural networks')
    dcn = DCN(args.way,args.shot,args.query,args.embedding_class,with_variation=bool(args.variational),weight_or_not=args.weight_or_not,loss = args.loss)
    dcn.embedding = nn.DataParallel(dcn.embedding,device_ids=[args.gpu,args.gpu+1])
    dcn.relation = nn.DataParallel(dcn.relation,device_ids=[args.gpu,args.gpu+1])
    dcn.load_state_dict(torch.load("../models/VDRN-"+str(args.model_episode)+"-"+str(args.embedding_class) + "-" + args.dataset + "-" + args.loss +"-var"+ str(args.variational) + "-shot"+ str(args.shot)+ "-"+ str(args.weight_or_not) + ".pkl",map_location={'cuda:':'cuda:'+str(args.gpu)}))
#     dcn.load_state_dict(torch.load("../models/VDRN-"+str(args.model_episode)+"-"+str(args.embedding_class) + "-" + args.dataset + "-"+ str(args.variational) + "-shot"+ str(args.shot)+ ".pkl",map_location={'cuda:':'cuda:'+str(args.gpu)}))
    print("load model ok!")
    dcn.to(device)

    test(dcn,task_generator)

if __name__ == '__main__':
    main()


