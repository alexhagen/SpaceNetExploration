
import logging
import os
import shutil
import sys
from time import localtime, strftime
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.unet.unet import Unet
from models.unet.unet_baseline import UnetBaseline
from utils.data_transforms import ToTensor, ToBinaryTensor
from utils.dataset import SpaceNetDataset, SpaceNetDatasetBinary
from utils.logger import Logger
from utils.train_utils import AverageMeter, log_sample_img_gt, render
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Softmax, BatchNorm2d


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logging.info('Using PyTorch version %s.', torch.__version__)

TRAIN = {
    # hardware and framework parameters
    'use_gpu': True,
    'dtype': torch.float64,

    # paths to data splits
    'data_path_root': '/qfs/projects/sgdatasc/spacenet/', # common part of the path for data_path_train, data_path_val and data_path_test
    'data_path_train': 'Vegas_processed_train/annotations',
    'data_path_val': 'Vegas_processed_val/annotations',
    'data_path_test': 'Vegas_processed_test/annotations',

    # training and model parameters
    'evaluate_only': False,  # Only evaluate the model on the val set once
    'model_choice': 'unet_baseline',  # 'unet_baseline' or 'unet'
    'feature_scale': 1,  # parameter for the Unet

    'num_workers': 4,  # how many subprocesses to use for data loading
    'train_batch_size': 10,
    'val_batch_size': 10,
    'test_batch_size': 10,

    'starting_checkpoint_path': '',  # checkpoint .tar to train from, empty if training from scratch
    'loss_weights': [0.1, 0.8, 0.1],  # weight given to loss for pixels of background, building interior and building border classes
    'learning_rate': 0.5e-3,
    'print_every': 200,  # print every how many steps
    'total_epochs': 100,  # for the walkthrough, we are training for one epoch

    'experiment_name': 'unet_binary_weights', # using weights that emphasize the building interior pixels
}


# config for the run
evaluate_only = TRAIN['evaluate_only']

use_gpu = TRAIN['use_gpu']
dtype = TRAIN['dtype']

torch.backends.cudnn.benchmark = True  # enables benchmark mode in cudnn, see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

data_path_root = TRAIN['data_path_root']
data_path_train = os.path.join(data_path_root, TRAIN['data_path_train'])
data_path_val = os.path.join(data_path_root, TRAIN['data_path_val'])
data_path_test = os.path.join(data_path_root, TRAIN['data_path_test'])

model_choice = TRAIN['model_choice']
feature_scale = TRAIN['feature_scale']

num_workers = TRAIN['num_workers']  # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
train_batch_size = TRAIN['train_batch_size']
val_batch_size = TRAIN['val_batch_size']
test_batch_size = TRAIN['test_batch_size']

# checkpoint to used to initialize, empty if training from scratch
# starting_checkpoint_path = './checkpoints/unet_checkpoint_epoch9_2018-05-26-21-52-44.pth.tar'
starting_checkpoint_path = ''

# weights for computing the loss function; absolute values of the weights do not matter
# [background, interior of building, border of building]
loss_weights = torch.from_numpy(np.array(TRAIN['loss_weights']))
learning_rate = TRAIN['learning_rate']
print_every = TRAIN['print_every']
total_epochs = TRAIN['total_epochs']

experiment_name = TRAIN['experiment_name']

split_tags = ['trainval', 'test']  # compatibility with the SpaceNet image preparation code - do not change


# device configuration
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
logging.info('Using device: %s.', device)

# data sets and loaders
dset_train = SpaceNetDatasetBinary(data_path_train, split_tags, transform=T.Compose([ToBinaryTensor()]))
loader_train = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)  # shuffle True to reshuffle at every epoch

dset_val = SpaceNetDatasetBinary(data_path_val, split_tags, transform=T.Compose([ToBinaryTensor()]))
loader_val = DataLoader(dset_val, batch_size=val_batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True)  # also reshuffle val set because loss is recorded for the last batch

dset_test = SpaceNetDatasetBinary(data_path_test, split_tags, transform=T.Compose([ToBinaryTensor()]))
loader_test = DataLoader(dset_test, batch_size=test_batch_size, shuffle=True,
                         num_workers=num_workers, drop_last=True)

logging.info('Training set size: {}, validation set size: {}, test set size: {}'.format(
    len(dset_train), len(dset_val), len(dset_test)))


#def get_sample_images(which_set='train'):
#    # which_set could be 'train' or 'val'; loader should already have shuffled them; gets one batch
#    loader = loader_train if which_set == 'train' else loader_val
#    images = None
#    image_tensors = None
#    for batch in loader:
#        image_tensors = batch['image']
#        images = batch['image'].cpu().numpy()
#        break  # take the first shuffled batch
#    images_li = []
#    for b in range(0, images.shape[0]):
#        images_li.append(images[b, :, :, :])
#    return images_li, image_tensors
#
#sample_images_train, sample_images_train_tensors = get_sample_images(which_set='train')
#sample_images_val, sample_images_val_tensors = get_sample_images(which_set='val')


def visualize_result_on_samples(epoch, model, sample_images, logger, step, split='train'):
    model.eval()
    with torch.no_grad():
        sample_images = sample_images.to(device=device, dtype=dtype)
        scores = model(sample_images).cpu().numpy()
        images_li = []
        for i in range(scores.shape[0]):

            input = scores[i, :, :, :].squeeze()
            picture = render(input)
            if i == 0:
                truth = sample_images[i, :, :, :].cpu().numpy().squeeze()
                truth = np.moveaxis(truth, 0, -1)
                toprint = np.moveaxis(picture, 0, -1)
                io.imsave('img/prediction_epoch{}_step{}.png'.format(epoch, step), toprint)
                io.imsave('img/truth_epoch{}_step{}.png'.format(epoch, step), truth)
            images_li.append(picture)

        logger.image_summary('result_{}'.format(split), images_li, step)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def train(loader_train, model, criterion, optimizer, epoch, step, logger_train):
    mapes = []
    losses = []
    prog = tqdm#lambda x: x
    counter = 0
    for t, data in enumerate(prog(loader_train, total=100)):
        if counter >= 100:
            break
        # put model to training mode; we put it in eval mode in visualize_result_on_samples for every print_every
        model.train()
        step += 1
        counter += 1
        x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = data['target'].to(device=device, dtype=dtype)  # y is not a int value here; also an image
        # forward pass on this batch
        scores = model(x)

        loss = criterion(scores, y)
        losses.append(loss.detach().item())
        #mape = 100.0 * np.abs(scores.cpu().detach().numpy() - y.cpu().detach().numpy()) / (y.cpu().detach().numpy())
        num = (scores.detach() - y.detach()).abs()
        denom = ((y.detach() + scores.detach()) / 2.0)
        mape = (num/denom).mean().item()
        mapes.append(mape)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TensorBoard logging and print a line to stdout; note that the accuracy is wrt the current mini-batch only
        if step % print_every == 1:
            # 1. log scalar values (scalar summary)
            #_, preds = scores.max(1)
            #percs.append(y.detach().float().mean().item())
            #accuracy = ((y - preds).abs() <= 0.5).float().mean()
            #accuracies.append(accuracy.detach().item())

            #info = {'minibatch_loss': loss.item(), 'minibatch_accuracy': accuracy.item()}
            #for tag, value in info.items():
            #    logger_train.scalar_summary(tag, value, step + 1)

            #logging.info(
            #    'Epoch {}, step {}, loss is {:.2f}, accuracy is {:.2f}'.format(
            #        epoch, step, np.mean(losses), np.mean(accuracies)))

            # 2. log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                #logger_train.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                #logger_train.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

            # 3. log training images (image summary)
            #visualize_result_on_samples(epoch, model, sample_images_train_tensors, logger_train, step, split='train')
            #visualize_result_on_samples(epoch, model, sample_images_val_tensors, logger_train, step, split='val')
    logging.warning(
        'Epoch {}, loss is {:.2e}, mape is {:.2f}'.format(
            epoch, np.mean(losses), np.mean(mapes)))
    return step


def evaluate(loader, model, criterion):
    """Evaluate the model on dataset of the loader"""
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()  # put model to evaluation mode
    with torch.no_grad():
        for t, data in enumerate(loader):
            x = data['image'].to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = data['target'].to(device=device, dtype=dtype)
            scores = model(x)
            loss = criterion(scores, y)
            # DEBUG logging.info('Val loss = %.4f' % loss.item())

            #_, preds = scores.max(1)
            #accuracy = (y == preds).float().mean()

            #losses.update(loss.item(), x.size(0))
            #accuracies.update(accuracy.item(), 1)  # average already taken for accuracy for each pixel

    return losses.avg, accuracies.avg


def save_checkpoint(state, is_best, path='../checkpoints/binary_checkpoints.pth.tar', checkpoint_dir='../checkpoints'):
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(checkpoint_dir, 'binary_model_best.pth.tar'))

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class encoder(torch.nn.Module):
    def __init__(self, latent_size=1000, first_conv_size=32, img_size=256,
                 bs=4):
        super(encoder, self).__init__()
        self.latent_size = latent_size
        self.first_conv_size = first_conv_size
        self.img_size = img_size
        self.bs = bs
        self.ops = []
        cargs = dict(kernel_size=3, stride=1, padding=1)
        mpargs = dict(kernel_size=2, stride=2, padding=0)
        # Downsize convolve until dims < 10
        si = 3
        dim = self.img_size
        so = self.first_conv_size
        self.conv1 = Conv2d(si, so, **cargs)
        self.bn1 = BatchNorm2d(so)
        self.relu1 = ReLU()
        self.conv11 = Conv2d(so, so, **cargs)
        self.bn11 = BatchNorm2d(so)
        self.relu11 = ReLU()
        self.mp1 = MaxPool2d(**mpargs)
        si = so; so *= 2; dim /= 2
        self.conv2 = Conv2d(si, so, **cargs)
        self.bn2 = BatchNorm2d(so)
        self.relu2 = ReLU()
        self.conv21 = Conv2d(so, so, **cargs)
        self.bn21 = BatchNorm2d(so)
        self.relu21 = ReLU()
        self.mp2 = MaxPool2d(**mpargs)
        si = so; so *= 2; dim /= 2
        self.conv3 = Conv2d(si, so, **cargs)
        self.bn3 = BatchNorm2d(so)
        self.relu3 = ReLU()
        self.conv31 = Conv2d(so, so, **cargs)
        self.bn31 = BatchNorm2d(so)
        self.relu31 = ReLU()
        self.mp3 = MaxPool2d(**mpargs)
        si = so; so *= 2; dim /= 2
        self.conv4 = Conv2d(si, so, **cargs)
        self.bn4 = BatchNorm2d(so)
        self.relu4 = ReLU()
        self.conv41 = Conv2d(so, so, **cargs)
        self.bn41 = BatchNorm2d(so)
        self.relu41 = ReLU()
        self.mp4 = MaxPool2d(**mpargs)
        si = so; so *= 2; dim /= 2
        self.conv5 = Conv2d(si, so, **cargs)
        self.relu5 = ReLU()
        self.bn5 = BatchNorm2d(so)
        self.conv51 = Conv2d(so, so, **cargs)
        self.relu51 = ReLU()
        self.bn51 = BatchNorm2d(so)
        self.mp5 = MaxPool2d(**mpargs)
        si = so; so *= 2; dim /= 2
        self.conv6 = Conv2d(si, so, **cargs)
        self.relu6 = ReLU()
        self.bn6 = BatchNorm2d(so)
        self.conv61 = Conv2d(so, so, **cargs)
        self.relu61 = ReLU()
        self.bn61 = BatchNorm2d(so)
        #self.mp6 = MaxPool2d(**mpargs)
        #dim /= 2
        self.reshape = Reshape(bs, -1)
        self.linear1 = Linear(int(so * dim * dim), self.latent_size)
        self.relu10 = ReLU()
        self.linear2 = Linear(self.latent_size, 1)
        self.relu11 = ReLU()
        #self.softmax = Softmax()


    def forward(self, x):
        # block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.bn11(x)
        x = self.mp1(x)
        # block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv21(x)
        x = self.relu21(x)
        x = self.bn21(x)
        x = self.mp2(x)
        # block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv31(x)
        x = self.relu31(x)
        x = self.bn31(x)
        x = self.mp3(x)
        # block 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.conv41(x)
        x = self.relu41(x)
        x = self.bn41(x)
        x = self.mp4(x)
        # block 5
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.conv51(x)
        x = self.relu51(x)
        x = self.bn51(x)
        x = self.mp5(x)
        # block 6
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.conv61(x)
        x = self.relu61(x)
        x = self.bn61(x)
        #x = self.mp6(x)
        x = self.reshape(x)
        x = self.linear1(x)
        x = self.relu10(x)
        x = self.linear2(x)
        x = self.relu11(x)
        #x = self.softmax(x)
        return x


def main():
    num_classes = 3

    # create checkpoint dir
    checkpoint_dir = 'checkpoints/{}'.format(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger_train = Logger('logs/{}/train'.format(experiment_name))
    logger_val = Logger('logs/{}/val'.format(experiment_name))
    #log_sample_img_gt(sample_images_train, sample_images_val, logger_train, logger_val)
    logging.info('Logged ground truth image samples')

    model = encoder(bs=TRAIN['train_batch_size'])

    model = model.to(device=device, dtype=dtype)  # move the model parameters to CPU/GPU
    #model = nn.DataParallel(model, device_ids=[0, 1])

    criterion = nn.L1Loss().to(device=device, dtype=dtype) #nn.CrossEntropyLoss().to(device=device, dtype=dtype)

    optimizer = optim.Adam(model.parameters())

    # resume from a checkpoint if provided
    starting_epoch = 0
    best_acc = 0.0

    if os.path.isfile(starting_checkpoint_path):
        logging.info('Loading checkpoint from {0}'.format(starting_checkpoint_path))
        checkpoint = torch.load(starting_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
    else:
        logging.info('No valid checkpoint is provided. Start to train from scratch...')
        model.apply(weights_init)

    if evaluate_only:
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        print('Evaluated on val set, loss is {}, accuracy is {}'.format(val_loss, val_acc))
        return

    step = starting_epoch * len(dset_train)

    for epoch in range(starting_epoch, total_epochs):
        logging.info('Epoch {} of {}'.format(epoch, total_epochs))

        # train for one epoch
        step = train(loader_train, model, criterion, optimizer, epoch, step, logger_train)

        # evaluate on val set
        logging.info('Evaluating model on the val set at the end of epoch {}...'.format(epoch))
        val_loss, val_acc = evaluate(loader_val, model, criterion)
        logging.info('\nEpoch {}, val loss is {}, val accuracy is {}\n'.format(epoch, step, val_loss, val_acc))
        logger_val.scalar_summary('val_loss', val_loss, step + 1)
        logger_val.scalar_summary('val_acc', val_acc, step + 1)
        # log the val images too

        # record the best accuracy; save checkpoint for every epoch
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint_path = os.path.join(checkpoint_dir,
                                       'binary_checkpoint_epoch{}_{}.pth.tar'.format(epoch, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
        logging.info(
            'Saving to checkoutpoint file at {}. Is it the highest accuracy checkpoint so far: {}'.format(
                checkpoint_path, str(is_best)))
        save_checkpoint({
            'epoch': epoch + 1,  # saved checkpoints are numbered starting from 1
            'arch': model_choice,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, is_best, checkpoint_path, checkpoint_dir)


if __name__ == '__main__':
    main()
