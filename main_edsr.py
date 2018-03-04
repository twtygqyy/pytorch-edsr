import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from edsr import Net
from dataset import DatasetFromHdf5

# Training settings
parser = argparse.ArgumentParser(description="PyTorch EDSR")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 0")

def main():

    global opt, model 
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    train_set = DatasetFromHdf5("path_to_dataset.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = nn.L1Loss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay, betas = (0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()