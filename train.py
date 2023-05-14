import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import load_dataset
from model import load_model


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args, config):
    '''
    train function
    training and validation loop.
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataloader
    train_loader, val_loader = load_dataset(args)

    # load model
    model = load_model(args, config)
    model = model.to(device)

    # set averagemeter
    acc_list = AverageMeter()
    loss_list = AverageMeter()

    # train settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch_num = args.epoch

    # training loop
    for epoch in range(epoch_num):
        print('epoch: ' + str(epoch+1))

        for i, data in enumerate(tqdm(train_loader, leave=False)):

            inputs, labels = data

            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(outputs, labels, (1, 5))

        # print accuracy and loss
            acc_list.update(acc1.item())
            loss_list.update(loss.item())

        print('train accuracy: ' + str(acc_list.avg))
        print('train loss: ' + str(loss_list.avg))

        # validation loop
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(val_loader, leave=False):
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('val accuracy : %d %%' % (100 * correct / total))

    print('Finished Training')
