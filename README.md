# RelationNet2: Deep Comparison Columns for Few-Shot Learning
PyTorch code for IJCNN 2020 paper: [RelationNet2: Deep Comparison Columns for few shot learning](https://arxiv.org/pdf/1811.07100v2.pdf) 

## Requirements
Python 2.7

Pytorch 0.3

## Data
For mini-Imagenet experiments, please download [mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and put it in ./datas/mini-Imagenet and run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on [maml](https://github.com/cbfinn/maml)).

For tiered-Imagenet experiments, please download [tiered-Imagenet](https://github.com/yaoyao-liu/tiered-imagenet-tools).

## Train
mini-Imagenet 5 way 1 shot:

```
python miniimagenet_train_one_shot.py -w 5 -s 1 -b 15
```

mini-Imagenet 5 way 5 shot:

```
python miniimagenet_train_few_shot.py -w 5 -s 5 -b 10
```

tiered-Imagenet 5 way 1 shot:

```
python tieredimagenet_train_one_shot.py -w 5 -s 1 -b 15
```

tiered-Imagenet 5 way 5 shot:

```
python tieredimagenet_train_few_shot.py -w 5 -s 5 -b 10
```

you can change -b parameter based on your GPU memory. Currently It will load my trained model, if you want to train from scratch, you can delete models by yourself.

## Test

mini-Imagenet 5way 1 shot:

```
python miniimagenet_test_one_shot.py -w 5 -s 1
```

Other experiments' testings are similar.


## Citing

If you use this code in your research, please use the following BibTeX entry.

```
@inproceedings{Xueting2020,
  title={RelationNet2: Deep Comparison Columns for Few-Shot Learning},
  author={Zhang, Xueting and Qiang, Yuting and Sung, Flood and Yang, Yongxin and Hospedales, Timothy M},
  booktitle={Proceedings of International Joint Conference on Neural Networks (IJCNN)},
  year={2020}
}
```

## Reference

[MAML](https://github.com/cbfinn/maml)

[MAML-pytorch](https://github.com/katerakelly/pytorch-maml)

[Prototypical Network](https://github.com/cyvius96/prototypical-network-pytorch)

