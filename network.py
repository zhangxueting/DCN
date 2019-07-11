import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# SENet's Module
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,with_variation=False):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if with_variation:
            out_planes = planes+1
        else:
            out_planes = planes
        self.conv2 = conv3x3(planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.se = SEModule(out_planes,reduction=16)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, with_variation=False):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if with_variation:
            out_planes = planes * self.expansion+1
        else:
            out_planes = planes * self.expansion
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(out_planes,reduction=16)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out

# -----------------------
# Class: EmbeddingSENet
# Description: A SENet based embedding network for feature extraction.
#              if with_variation = True, a variational SENet would be constructed.
# -----------------------
class EmbeddingSENet(nn.Module):

    def __init__(self, block, layers, num_class=80, with_variation=True):
        self.inplanes = 64
        super(EmbeddingSENet, self).__init__()
        self.with_variation = with_variation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_class)
        self.expansion = block.expansion


    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes,planes,with_variation=self.with_variation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        variational_features = []

        if self.with_variation: # variational version
            feature1 = self.layer1(x) # [expansion*64+1,56,56]
            split_size = [self.expansion*64,1]
            feature1_mean,feature1_std = torch.split(feature1,split_size,dim=1)
            feature1_std = torch.sigmoid(feature1_std)
            feature1_std_ext = feature1_std.repeat(1,split_size[0],1,1)
            feature1 = feature1_mean + feature1_std_ext*torch.randn(feature1_mean.size(),device=feature1.get_device())

            feature2 = self.layer2(feature1) #[expansion*128+1,28,28]
            split_size = [self.expansion*128,1]
            feature2_mean,feature2_std = torch.split(feature2,split_size,dim=1)
            feature2_std = torch.sigmoid(feature2_std)
            feature2_std_ext = feature2_std.repeat(1,split_size[0],1,1)
            feature2 = feature2_mean + feature2_std_ext*torch.randn(feature2_mean.size(),device=feature2.get_device())

            feature3 = self.layer3(feature2) #[expansion*256+1,14,14]
            split_size = [self.expansion*256,1]
            feature3_mean,feature3_std = torch.split(feature3,split_size,dim=1)
            feature3_std = torch.sigmoid(feature3_std)
            feature3_std_ext = feature3_std.repeat(1,split_size[0],1,1)
            feature3 = feature3_mean + feature3_std_ext*torch.randn(feature3_mean.size(),device=feature3.get_device())

            feature4 = self.layer4(feature3) #[expansion*512+1,7,7]
            split_size = [self.expansion*512,1]
            feature4_mean,feature4_std = torch.split(feature4,split_size,dim=1)
            feature4_std = torch.sigmoid(feature4_std)
            feature4_std_ext = feature4_std.repeat(1,split_size[0],1,1)
            feature4 = feature4_mean + feature4_std_ext*torch.randn(feature4_mean.size(),device=feature4.get_device())
            x = self.avgpool(feature4)

            variational_features = [feature1_mean,feature1_std_ext,
                                    feature2_mean,feature2_std_ext,
                                    feature3_mean,feature3_std_ext,
                                    feature4_mean,feature4_std_ext]

            feature1_std = feature1_std.view(feature1_std.size(0),-1)
            feature2_std = feature2_std.view(feature2_std.size(0),-1)
            feature3_std = feature3_std.view(feature3_std.size(0),-1)
            feature4_std = feature4_std.view(feature4_std.size(0),-1)

            std_mean = (torch.mean(feature1_std,1) + torch.mean(feature2_std,1) + torch.mean(feature3_std,1) + torch.mean(feature4_std,1))/4.0

        else: #standard version
            feature1 = self.layer1(x) # [expansion*64,56,56]
            feature2 = self.layer2(feature1) #[expansion*128,28,28]
            feature3 = self.layer3(feature2) #[expansion*256,14,14]
            feature4 = self.layer4(feature3) #[expansion*512,7,7]
            std_mean = torch.zeros(feature1.size(0),1,device = feature1.get_device())
            x = self.avgpool(feature4)


        x = x.view(x.size(0), -1)
        x = self.fc(x)



        return x,[feature1,feature2,feature3,feature4],std_mean,variational_features

# -------------------------
# Class: RelationSENet
# Description: Dense Relation Module based on SENet. Here we have 4 relation modules.
# -------------------------
class RelationSENet(nn.Module):
    def __init__(self, block, layers, num_class,weight_or_not="weight",loss="CE"):
        super(RelationSENet, self).__init__()
        self.relation1 = self._make_layer(block,64*2,128,layers[0],stride=2)
        self.relation2 = self._make_layer(block,128*3,256,layers[1],stride=2)
        self.relation3 = self._make_layer(block,256*3,512,layers[2],stride=2)
        self.relation4 = self._make_layer(block,512*3,512,layers[3],stride=1)

        self.avgpool1 = nn.AvgPool2d(28)
        self.fc1 = nn.Linear(128 * block.expansion,1)
        self.fc_w1 = nn.Linear(128 * block.expansion, 1)

        self.avgpool2 = nn.AvgPool2d(14)
        self.fc2 = nn.Linear(256 * block.expansion,1)
        self.fc_w2 = nn.Linear(256 * block.expansion, 1)

        self.avgpool3 = nn.AvgPool2d(7)
        self.fc3 = nn.Linear(512 * block.expansion,1)
        self.fc_w3 = nn.Linear(512 * block.expansion, 1)
        
        self.avgpool4 = nn.AvgPool2d(7)
        self.fc4 = nn.Linear(512 * block.expansion,1)
        self.fc_w4 = nn.Linear(512 * block.expansion, 1)

        self.num_class = num_class
        self.weight_or_not = weight_or_not
        self.loss = loss

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,support_x_features,query_x_features):

        pairs1 = torch.cat((support_x_features[0],query_x_features[0]),1)
        similarity_feature1 = self.relation1(pairs1) #[expansion*128,28,28]

        pairs2 = torch.cat((support_x_features[1],similarity_feature1,query_x_features[1]),1)
        similarity_feature2 = self.relation2(pairs2) #[expansion*256,14,14]

        pairs3 = torch.cat((support_x_features[2],similarity_feature2,query_x_features[2]),1)
        similarity_feature3 = self.relation3(pairs3) #[expansion*512,7,7]

        pairs4 = torch.cat((support_x_features[3],similarity_feature3,query_x_features[3]),1)
        similarity_feature4 = self.relation4(pairs4) #[expansion*512,7,7]

        similarity_feature1 = self.avgpool1(similarity_feature1)
        similarity_feature1 = similarity_feature1.view(similarity_feature1.size(0), -1)
        # score1 = torch.sigmoid(self.fc1(similarity_feature1))
        score1 = self.fc1(similarity_feature1)
        w1 = torch.sigmoid(self.fc_w1(similarity_feature1))

        similarity_feature2 = self.avgpool2(similarity_feature2)
        similarity_feature2 = similarity_feature2.view(similarity_feature2.size(0), -1)
        # score2 = torch.sigmoid(self.fc2(similarity_feature2))
        score2 = self.fc2(similarity_feature2)
        w2 = torch.sigmoid(self.fc_w2(similarity_feature2))
        

        similarity_feature3 = self.avgpool3(similarity_feature3)
        similarity_feature3 = similarity_feature3.view(similarity_feature3.size(0), -1)
        # score3 = torch.sigmoid(self.fc3(similarity_feature3))
        score3 = self.fc3(similarity_feature3)
        w3 = torch.sigmoid(self.fc_w3(similarity_feature3))

        similarity_feature4 = self.avgpool4(similarity_feature4)
        similarity_feature4 = similarity_feature4.view(similarity_feature4.size(0), -1)
        # score4 = torch.sigmoid(self.fc4(similarity_feature4))
        score4 = self.fc4(similarity_feature4)
        w4 = torch.sigmoid(self.fc_w4(similarity_feature4))

        if self.loss == "BCE":
            score1 = torch.sigmoid(score1)
            score2 = torch.sigmoid(score2)
            score3 = torch.sigmoid(score3)
            score4 = torch.sigmoid(score4)

        if self.weight_or_not == "weight":
            score1 = w1 * score1
            score2 = w2 * score2
            score3 = w3 * score3
            score4 = w4 * score4
        
        return score1,score2,score3,score4

# ----------------------
# Class: DCN
# Description: the main class to construct a variational/standard dense relation network
#              for 1 shot or k shot classification.
# ----------------------

class DCN(nn.Module):
    def __init__(self,num_class,num_support,num_query,num_embedding_class,with_variation=True,weight_or_not="weight",loss="CE"):
        super(DCN, self).__init__()

        self.num_class = num_class
        self.num_support = num_support
        self.num_query = num_query
        self.with_variation = with_variation
        self.weight_or_not = weight_or_not
        self.loss = loss

        self.embedding = EmbeddingSENet(SEBasicBlock,[3, 4, 6, 3],num_embedding_class,with_variation)
        self.relation = RelationSENet(SEBasicBlock,[2 , 2, 2, 2],num_class,self.weight_or_not,self.loss)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, support_x, query_x):

        if self.num_support > 1 and not self.with_variation: # k shot setting
            _,pre_support_x_features,_,_ = self.embedding(support_x) #[25,16,xx,xx]
            support_x_features = []
            for support_x_feature in pre_support_x_features:
                b,c,h,w = support_x_feature.size()
                support_x_feature = support_x_feature.view(self.num_class,self.num_support,c,h,w)
                support_x_feature = torch.mean(support_x_feature,1).squeeze(1) #[5,c,h,w]
                support_x_feature = support_x_feature.unsqueeze(0).repeat(self.num_class*self.num_query,1,1,1,1).view(-1,c,h,w)
                support_x_features.append(support_x_feature)

        elif self.num_support > 1 and self.with_variation:
            _,_,_,variational_features = self.embedding(support_x)
            pro_features = []
            for feature in variational_features:
                b,c,h,w = feature.size()
                feature = feature.view(self.num_class,self.num_support,c,h,w)
                feature = torch.mean(feature,1).squeeze(1) #[5,c,h,w]
                feature = feature.unsqueeze(0).repeat(self.num_class*self.num_query,1,1,1,1).view(-1,c,h,w)
                pro_features.append(feature)
            support_x_features = []
            for i in range(4):
                support_x_feature = pro_features[2*i] + pro_features[2*i+1]*torch.randn(pro_features[2*i].size(),device=pro_features[2*i].get_device())
                support_x_features.append(support_x_feature)

        else: #  1 shot setting
            support_x_ext = support_x.unsqueeze(0).repeat(self.num_class*self.num_query,1,1,1,1).view(-1,3,224,224)
            _,support_x_features,_,_ = self.embedding(support_x_ext)

        query_x_ext = query_x.unsqueeze(0).repeat(self.num_class,1,1,1,1)
        query_x_ext = torch.transpose(query_x_ext,0,1).contiguous().view(-1,3,224,224)
        _,query_x_features,_,_ = self.embedding(query_x_ext)

        score1,score2,score3,score4 = self.relation(support_x_features,query_x_features)

        return score1,score2,score3,score4



