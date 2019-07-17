# code is based on https://github.com/katerakelly/pytorch-maml
import torch
import random
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torchvision as tv
from PIL import Image
import numpy as np

# -----------------------------------
# Method: split_fine_grained_dataset
# Description: split fine grained dataset like CUB,Stanford Car and FGVC-aircraft into
#              two folders: metatrain_folders and metatest_folders
# -----------------------------------
def split_fine_grained_dataset(dataset_path,num_train,num_test):

    data_folder = [os.path.join(dataset_path, class_name) \
                for class_name in os.listdir(dataset_path) \
                if os.path.isdir(os.path.join(dataset_path, class_name)) \
                ]
    random.seed(1)
    random.shuffle(data_folder)

    metatrain_folder = data_folder[:num_train]
    metatest_folder = data_folder[num_train:num_train+num_test]

    return metatrain_folder,metatest_folder

# -----------------------------
# Method: mini_imagenet_folders
# Description: Since we already have split miniimagenet folders, here we only have to
#              shuffle them and return metatrain_folders and metatest_folders
# -----------------------------
def mini_imagenet_folder(train_folder,test_folder):

    metatrain_folder = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folder = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folder)
    random.shuffle(metatest_folder)

    return metatrain_folder,metatest_folder

# -----------------------
# Class: FewShotTask
# Description: Generate a C way K shot Few-Shot Learning task from data_folders (metatrain_folders or metatset_folders)
#              including data_roots and data_labels
# -----------------------
class FewShotTask(object):

    def __init__(self, data_folder, num_class, num_train, num_test,type="meta_train"):

        self.data_folder = data_folder
        self.num_class = num_class
        self.num_train = num_train
        self.num_test = num_test
        self.type=type

        class_folders = random.sample(self.data_folder,self.num_class)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:num_train]
            self.test_roots += samples[c][num_train:num_train+num_test]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

# -------------------------
# Class: FewShotDataset
# Description: Generate a pytorch dataset for training based on the input task.
# -------------------------
class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

# -------------------------
# Class: TaskGenerator
# Description: 1) to sample task (a ready-to-use task dataset) for training
#              2) generate a classification dataset for embedding pretraining
# -------------------------
class TaskGenerator():

    def __init__(self,metatrain_folder,metatest_folder,classifier_root):
        super(TaskGenerator, self).__init__()
        self.metatrain_folder = metatrain_folder
        self.metatest_folder = metatest_folder
        self.classifier_root = classifier_root

    def sample_task(self,num_class,num_support,num_query,type="meta_train"):

        if type == "meta_train":
            task = FewShotTask(self.metatrain_folder,num_class,num_support,num_query,type=type)
        else:
            task = FewShotTask(self.metatest_folder,num_class,num_support,num_query,type=type)
        support_dataloader = self.get_data_loader(task,split="train",shuffle=False)
        query_dataloader = self.get_data_loader(task,split="test",shuffle=True)
        # sample datas
        support_x,support_y = support_dataloader.__iter__().next() #sample once to obtain all data
        query_x,query_y = query_dataloader.__iter__().next()

        return support_x,support_y,query_x,query_y

    def get_data_loader(self,task,split='train',shuffle = False):

        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                    ])

        if task.type == "meta_train":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transform
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transform
            ])

        dataset = FewShotDataset(task,split=split,transform=transform)

        if split == 'train':
            batch_size = task.num_train*task.num_class
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4,
                                          pin_memory=torch.cuda.is_available())
        else:
            batch_size = task.num_test*task.num_class
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                          pin_memory=torch.cuda.is_available())

        return loader

    def get_classifier_dataset(self,batch_size):

        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                    ])

        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transform
            ])


        train_dataset = tv.datasets.ImageFolder(self.classifier_root,transform=train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          pin_memory=torch.cuda.is_available())


        return train_dataloader
