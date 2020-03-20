# Trainer node for EfficientNet

- EfficientNet: https://arxiv.org/abs/1905.11946
- Base code forked from https://github.com/narumiruna/efficientnet-pytorch


## Prerequisites

- Ubuntu
- Python 3
  - torch 1.0.1
  - torchvision 0.2.2.post3
  - mlconfig
  - tqdm
  - future
  - flask-restful
  - requests
  - httplib2
  - validators
  - pyYAML

## Usage

### Train

```shell
$ python train.py -c /path/to/config
```

### Evaluate

```shell
$ python evaluate.py --arch efficientnet_b0 -r /path/to/dataset
```

## Pretrained models

Source: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

| Model Name | Top-1 Accuracy |
| ------ | ------ |
| efficientnet_b0 | 76.52% |
| efficientnet_b1 | 77.80% |
| efficientnet_b2 | 78.83% |
| efficientnet_b3 | 80.19% |
| efficientnet_b4 | 82.27% |
| efficientnet_b5 | 83.11% |
| efficientnet_b6 |  |
| efficientnet_b7 |  |

## References

- https://arxiv.org/abs/1905.11946
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
