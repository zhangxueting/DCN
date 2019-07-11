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
from task_generator import TaskGenerator,split_fine_grained_dataset,mini_imagenet_folder
from network import DCN,EmbeddingSENet

parser = argparse.ArgumentParser(description="Variational Dense Relation Network for Few-Shot Learning")
parser.add_argument("--way",type = int, default = 5)  # num_class
parser.add_argument("--shot",type = int, default = 1) # num_support_per_class
parser.add_argument("--query",type = int, default = 5) # num_query_per_class
parser.add_argument("--embedding_class",type =int, default = 80) # num_class for embedding pre-training
parser.add_argument("--relation_episode",type = int, default= 100000)
parser.add_argument("--relation_learning_rate", type = float, default = 0.1)
parser.add_argument("--embedding_episode", type = int, default = 200)
parser.add_argument("--embedding_learning_rate", type = float, default = 0.1)
parser.add_argument("--embedding_batch_size",type=int,default = 256)
parser.add_argument("--embedding_train_num",type=int,default=590) # for miniimagenet, 600 images per class. 590 for train,10 for test
parser.add_argument("--embedding_test_num",type=int,default=10)
parser.add_argument("--gpu",type=int, default=0)
parser.add_argument("--dataset",type=str,default="miniimagenet") # tieredimagenet,cub,car,aircraft
parser.add_argument("--valid_set",type=int,default=1) # 1: use valid set for training,  0: not use valid set
parser.add_argument("--variational",type=int,default=1) # 1: variational version 0: standard version
parser.add_argument("--train_embedding",type=int,default=0) # 1: train 0:not train
parser.add_argument("--conti_train",type=int,default=0) # continue to train relation from last save model
parser.add_argument("--loss", type=str, default='COT') # BCE,CE,COT
parser.add_argument("--weight_or_not", str='weight') # to distinct "weight" or "noweight"
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class ComplementEntropy(nn.Module):

    def __init__(self):
        super(ComplementEntropy, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = yHat.shape[1]
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda(device=Px_log.get_device())
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss
    
def embedding_train(dcn,task_generator):
    embedding = dcn.embedding
    optim = torch.optim.SGD(embedding.parameters(), lr=args.embedding_learning_rate, momentum=0.9, weight_decay=1e-4)
    schedule = StepLR(optim, step_size = 60, gamma=0.2)

    # step 3: train process
    print('embedding training:')
    train_dataloader, test_dataloader = task_generator.get_classifier_dataset(args.embedding_batch_size,
                                                                              args.embedding_class,
                                                                              args.embedding_train_num,
                                                                              args.embedding_test_num)

    best_accuracy = 0.0

    for train_episode in range(args.embedding_episode):
        schedule.step(train_episode)

        # init dataset
        # calculate features
        total_rewards = 0
        total_samples = 0
        embedding.train()
        for image,label in train_dataloader:

            image = image.to(device)
            label = label.to(device)

            predict,_,std_mean,_ = embedding(image)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predict,label)

            total_loss = loss - 0.05*torch.mean(std_mean) # for standard version,std_mean = 0.0

            embedding.zero_grad()
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), 0.5)
            total_loss.backward()
            optim.step()

            _, predict_labels = torch.max(predict.data, 1)
            total_rewards += predict_labels.eq(label).sum().item()
            total_samples += label.size(0)

        train_accuracy = total_rewards/total_samples


        # test:
        with torch.no_grad():
            total_rewards = 0
            total_samples = 0
            embedding.eval()
            for image,label in test_dataloader:

                image = image.to(device)
                label = label.to(device)

                predict,_,_,_ = embedding(image)

                _, predict_labels = torch.max(predict.data, 1) # find the label of max similarity score

                total_rewards += predict_labels.eq(label).sum().item()
                total_samples += label.size(0)

            test_accuracy = total_rewards/total_samples

        print("train episode ", train_episode,": loss = ", loss.item(),"test acc:",test_accuracy,"train acc:",train_accuracy,"std_mean:",torch.mean(std_mean).item())

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            print("save embedding network")
            if not os.path.exists("../models/"):
                os.makedirs("../models/")
            torch.save(embedding.state_dict(),"../models/Embedding-"+str(args.embedding_class) + "-" + args.dataset + "-" + str(args.variational) +  ".pkl")


def relation_train(dcn,task_generator):
    if args.conti_train == 0:
        dcn.embedding.load_state_dict(torch.load("../models/Embedding-"+str(args.embedding_class) + "-" + args.dataset + "-" + str(args.variational) + ".pkl",map_location={'cuda:':'cuda:'+str(args.gpu)}))

        print("load embedding ok")
    else:
        dcn.load_state_dict(torch.load("../models/VDRN-"+str(args.conti_train)+"-"+str(args.embedding_class) + "-" + args.dataset +"-var"+ str(args.variational) + "-shot"+ str(args.shot)+ "-" + str(args.weight_or_not) + ".pkl",map_location={'cuda:':'cuda:'+str(args.gpu)}))
        print("load model ok!")

    optim = torch.optim.SGD(dcn.relation.parameters(),lr=args.relation_learning_rate,momentum=0.9, weight_decay=1e-4)
    entropy_optimizer = torch.optim.SGD(dcn.relation.parameters(), lr=args.relation_learning_rate, momentum=0.9, weight_decay=1e-4)
    schedule = StepLR(optim, step_size = 25000, gamma=0.2)

    # step 3: train process
    print ('relation training: ')
    total_rewards = 0

        
    for train_episode in range(args.conti_train,args.relation_episode):
        dcn.train()
        schedule.step(train_episode)

        # init dataset
        support_x,support_y,query_x,query_y = task_generator.sample_task(args.way,args.shot,args.query)
        support_x = support_x.to(device)
        query_x = query_x.to(device)

        query_predict_y0,query_predict_y1,query_predict_y2,query_predict_y3 = dcn(support_x,query_x)
        
        if args.loss == 'BCE':
            criterion = nn.BCELoss(reduction=None)
        
            one_hot_labels = torch.zeros(args.query*args.way,args.way).scatter_(1, query_y.view(-1,1),1).view(-1,1)
            one_hot_labels = one_hot_labels.to(device)
            loss0 = criterion(query_predict_y0, one_hot_labels)
            loss1 = criterion(query_predict_y1, one_hot_labels)
            loss2 = criterion(query_predict_y2, one_hot_labels)
            loss3 = criterion(query_predict_y3, one_hot_labels)
            loss = loss0+loss1+loss2+loss3

            
        elif args.loss == 'CE':
            criterion = nn.CrossEntropyLoss()
            target_labels = query_y.view(-1)
            batch_size = target_labels.shape[0]
            target_labels = target_labels.to(device)
            # print('target labels : ', target_labels)
            loss0 = criterion(query_predict_y0.view(batch_size, -1), target_labels)
            loss1 = criterion(query_predict_y1.view(batch_size, -1), target_labels)
            loss2 = criterion(query_predict_y2.view(batch_size, -1), target_labels)
            loss3 = criterion(query_predict_y3.view(batch_size, -1), target_labels)
            loss = loss0+loss1+loss2+loss3
            
        elif args.loss == 'COT':
            
            target_labels = query_y.view(-1)
            batch_size = target_labels.shape[0]
            target_labels = target_labels.to(device)
            criterion = nn.CrossEntropyLoss()
            
            
            entropy_loss0 = criterion(query_predict_y0.view(batch_size, -1), target_labels)
            entropy_loss1 = criterion(query_predict_y1.view(batch_size, -1), target_labels)
            entropy_loss2 = criterion(query_predict_y2.view(batch_size, -1), target_labels)
            entropy_loss3 = criterion(query_predict_y3.view(batch_size, -1), target_labels)
            entropy_loss = entropy_loss0+entropy_loss1+entropy_loss2+entropy_loss3
            entropy_optimizer.zero_grad()
            entropy_loss.backward()
            entropy_optimizer.step()
            
            query_predict_y0,query_predict_y1,query_predict_y2,query_predict_y3 = dcn(support_x,query_x)
            target_labels = query_y.view(-1)
            batch_size = target_labels.shape[0]
            target_labels = target_labels.to(device)
            complement_criterion = ComplementEntropy()
            loss0 = complement_criterion(query_predict_y0.view(batch_size, -1), target_labels)
            loss1 = complement_criterion(query_predict_y1.view(batch_size, -1), target_labels)         
            loss2 = complement_criterion(query_predict_y2.view(batch_size, -1), target_labels)
            loss3 = complement_criterion(query_predict_y3.view(batch_size, -1), target_labels)
            loss = loss0 + loss1 + loss2 + loss3
            
            
#         elif args.loss == 'COT':
#             criterion = ComplementEntropy()
#             target_labels = query_y.view(-1)
#             batch_size = target_labels.shape[0]
#             target_labels = target_labels.to(device)
#             #one_hot_labels = torch.zeros(args.query*args.way, args.way).scatter_(1, query_y.view(-1, 1), 1)
#             #one_hot_labels = one_hot_labels.to(device)
#             #batch_size = args.query*args.way
#             loss0 = criterion(query_predict_y0.view(batch_size, -1), target_labels)
#             loss1 = criterion(query_predict_y1.view(batch_size, -1), target_labels)
#             loss2 = criterion(query_predict_y2.view(batch_size, -1), target_labels)
#             loss3 = criterion(query_predict_y3.view(batch_size, -1), target_labels)

        else:
            print('Error loss')
            
        # training
        
        
        dcn.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dcn.parameters(), 0.5)
        optim.step()


        query_predict_y3 = query_predict_y3.view(-1,args.way)
        _, predict_labels = torch.max(query_predict_y3.data, 1) # find the label of max similarity score
        total_rewards += predict_labels.eq(query_y.to(device)).sum().item()


        if (train_episode+1) % 100 == 0:
            accuracy = total_rewards/1.0/(args.way*args.query)/100
            total_rewards = 0
            print("train episode ", train_episode+1, ": loss = ", loss.item()," accuracy = ",accuracy)

        if (train_episode+1) % 10000 == 0:
            # save networks
            torch.save(dcn.state_dict(),"../models/VDRN-"+str(train_episode+1)+"-"+str(args.embedding_class) + "-" + args.dataset + "-" + args.loss +"-var"+ str(args.variational) + "-shot"+ str(args.shot) + "-" +str(args.weight_or_not) + ".pkl")

            print("save networks for episode:",train_episode)

def main():

    # Step 1: init data folders
    print("init dataset")

    if args.dataset == 'miniimagenet':
        if args.valid_set == 1:
            metatrain_folder,metatest_folder = mini_imagenet_folder(config.miniimagenet_trainvalfolder,
                                                                config.miniimagenet_testfolder)
        else:
            metatrain_folder,metatest_folder = mini_imagenet_folder(config.miniimagenet_trainfolder,
                                                                config.miniimagenet_valfolder)
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
    dcn.to(device)

    if args.train_embedding:
        embedding_train(dcn,task_generator)
        torch.cuda.empty_cache()
    relation_train(dcn,task_generator)

if __name__ == '__main__':
    main()


