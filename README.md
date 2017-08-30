# PyTorch EDSR
Implementation of CVPR2017 Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution"(https://arxiv.org/pdf/1707.02921.pdf) in PyTorch

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]
               
optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                use cuda?
  --resume RESUME       path to latest checkpoint (default: none)
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  --clip CLIP           Clipping Gradients. Default=0.1
  --threads THREADS     number of threads for data loader to use
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 0

```


### Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]

PyTorch EDSR Test

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --image IMAGE  image name
  --scale SCALE  scale factor, Default: 4
```
We convert Set5 test set images to mat format using Matlab, for best PSNR performance, please use Matlab
An example of usage is shown as follows:
```
python test.py --model model/model_epoch_500.pth --image butterfly_GT --scale 4 --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-edsr/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.


### Todo
  - Evaluation
