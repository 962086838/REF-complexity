import argparse
import os
import time
import bisect
from contextlib import contextmanager
from utils import random_seed, create_result_dir, Logger, TableLogger, AverageMeter, parse_function_call
from torch.nn.functional import cross_entropy
from model import *
from torch.optim import AdamW, SGD
import copy
from complexity_measures import get_all_measures

parser = argparse.ArgumentParser(description='Training SortNet')

parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--model', default='ResNet', type=str)
parser.add_argument('--loss', default='cross_entropy', type=str)
parser.add_argument('--metric', default='loss', type=str)
parser.add_argument('--repeat', default=1, type=int)

parser.add_argument('--epochs', default='150', type=str)
# corresponding to: eps_start, eps_end, p_start, p_end, total
parser.add_argument('--decays', default=None, type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--adam', action='store_true')

parser.add_argument('--gpu', default=-1, type=int, help='GPU id to use')

parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency')
parser.add_argument('--result-dir', default='result', type=str)
parser.add_argument('--filter-name', default='', type=str)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--visualize', action='store_true')


def cal_acc(outputs, targets):
    predicted = torch.max(outputs.data, 1)[1]
    return (predicted == targets).float().mean().item()


@contextmanager
def eval(model):
    state = [m.training for m in model.modules()]
    model.eval()
    yield
    for m, s in zip(model.modules(), state):
        m.train(s)


def train(net, loss_fun, epoch, train_loader, optimizer, schedule, logger, gpu, print_freq):
    batch_time, losses, accs = [AverageMeter() for _ in range(3)]
    start = time.time()
    epoch_start_time = start
    train_loader_len = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        lr = schedule(epoch, batch_idx)
        inputs = inputs.cuda(gpu, non_blocking=True)
        targets = targets.cuda(gpu, non_blocking=True)
        outputs = net(inputs)
        loss = loss_fun(outputs, targets)

        with torch.no_grad():
            losses.update(loss.data.item(), targets.size(0))
            accs.update(cal_acc(outputs.data, targets), targets.size(0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        if (batch_idx + 1) % print_freq == 0 and logger is not None:
            logger.print('Epoch: [{0}][{1}/{2}]   '
                         'time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                         'lr {lr:.4f}   '
                         'loss {loss.val:.4f} ({loss.avg:.4f})   '
                         'acc {acc.val:.4f} ({acc.avg:.4f})   '.format(
                epoch, batch_idx + 1, train_loader_len, batch_time=batch_time,
                lr=lr, loss=losses, acc=accs))
        start = time.time()

    loss, acc = losses.avg, accs.avg
    if logger is not None:
        logger.print('Epoch {0} training:  train loss {loss:.4f}   train acc {acc:.4f}   '
                     'lr {lr:.4f}   time {time:.2f}'.format(
            epoch, loss=loss, acc=acc, lr=lr, time=time.time() - epoch_start_time))
    return loss, acc


@torch.no_grad()
def test(net, loss_fun, epoch, test_loader, logger, test_logger, gpu, print_freq):
    batch_time, losses, accs = [AverageMeter() for _ in range(3)]
    start = time.time()
    epoch_start_time = start
    test_loader_len = len(test_loader)

    with eval(net):
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            outputs = net(inputs)
            losses.update(loss_fun(outputs, targets).item(), targets.size(0))
            accs.update(cal_acc(outputs, targets), targets.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if (batch_idx + 1) % print_freq == 0 and logger is not None:
                logger.print('Test: [{0}/{1}]   '
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                             'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                    batch_idx + 1, test_loader_len, batch_time=batch_time, acc=accs))

    loss, acc = losses.avg, accs.avg
    if test_logger is not None:
        test_logger.log({'epoch': epoch, 'loss': loss, 'acc': acc})
    if logger is not None:
        elapse = time.time() - epoch_start_time
        logger.print('Epoch %d testing:  ' % epoch + 'loss ' + f'{loss:.4f}' + '   acc ' + f'{acc:.4f}'
                     + '   time ' + f'{elapse:.2f}')
    return loss, acc


def create_schedule(args, batch_per_epoch, optimizer):
    epoch_tot = args.epochs
    if args.decays is not None:
        decays = [int(epoch) for epoch in args.decays.split(',')]
    else:
        decays = None
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def num_batches(epoch, minibatch=0):
        return epoch * batch_per_epoch + minibatch

    def cal_ratio(epoch, epoch_start, epoch_end, minibatch):
        if epoch_end <= epoch_start:
            return 1
        return min(max(num_batches(epoch - epoch_start, minibatch) / num_batches(epoch_end - epoch_start), 0), 1)

    def schedule(epoch, minibatch):
        if decays is None:
            ratio = cal_ratio(epoch, 0, epoch_tot, minibatch)
            for param_group, lr in zip(optimizer.param_groups, lrs):
                param_group['lr'] = 0.5 * lr * (1 + math.cos((ratio * math.pi)))
        else:
            index = bisect.bisect_right(decays, epoch)
            for param_group, lr in zip(optimizer.param_groups, lrs):
                param_group['lr'] = lr / (5 ** index)
        return optimizer.param_groups[0]['lr']

    return schedule


def init_data_and_model(args, gpu, noisy_generator):
    from dataset import load_data, input_dim
    random_seed(args.seed)

    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    else:
        raise NotImplementedError
    # num_classes = len(train_loader.dataset.classes)

    train_loader, train_no_noise_loader, test_loader = load_data(args.dataset, 'data/', args.batch_size,
                                                                 noisy_generator=noisy_generator,
                                                                 augmentation=args.aug,
                                                                 num_classes=num_classes)

    model_name, paras, kwargs = parse_function_call(args.model)
    if 'resnet' in model_name:
        model = globals()[model_name](*paras, input_dim=input_dim[args.dataset], num_classes=num_classes,
                                      dropout_rate=args.dropout_rate, **kwargs)
    else:
        model = VGG(model_name, num_classes)
    model = model.cuda(gpu)
    return train_loader, train_no_noise_loader, test_loader, model


def W_norm(param1, param2):
    dist = 0.0
    for ky in param1:
        if 'tracked' in ky:
            continue
        dist = torch.square(param1[ky] - param2[ky]).sum() + dist
        # print(ky)

    return dist ** 0.5


def main_worker(gpu, args, result_dir):
    extra_evaluation_metric_clean_train_loss = []
    extra_evaluation_metric_noisy_train_loss = []
    extra_evaluation_metric_clean_init_loss = []
    extra_evaluation_metric_noisy_init_loss = []

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)

    assert args.loss == 'cross_entropy'
    loss = cross_entropy
    train_loader, train_no_noise_loader, test_loader, model = init_data_and_model(args, gpu, None)

    logger = Logger(os.path.join(result_dir, 'log.txt'))
    for arg in vars(args):
        logger.print(arg, '=', getattr(args, arg))
    logger.print(model)
    logger.print('number of params: ', sum([p.numel() for p in model.parameters()]))

    if args.adam:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    args.epochs = int(args.epochs)
    schedule = create_schedule(args, len(train_loader), optimizer)

    if args.visualize:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(result_dir)
    else:
        writer = None

    train_logger = TableLogger(os.path.join(result_dir, 'train.log'), ['epoch', 'loss', 'acc'])
    test_logger = TableLogger(os.path.join(result_dir, 'test.log'), ['epoch', 'loss', 'acc'])
    init_loss, _ = test(model, loss, -1, train_no_noise_loader, logger, train_logger, gpu, args.print_freq)
    clean_ratio = None
    para_old = copy.deepcopy(model.state_dict())
    init_model = copy.deepcopy(model)

    for epoch in range(args.epochs):
        train(model, loss, epoch, train_loader, optimizer, schedule, logger, gpu, args.print_freq)
        train_loss, train_acc = test(model, loss, epoch, train_no_noise_loader, logger, train_logger, gpu,
                                     args.print_freq)
        if writer is not None:
            writer.add_scalar('curve/train loss', train_loss, epoch)
            writer.add_scalar('curve/train acc', train_acc, epoch)
    para_new = model.state_dict()
    distance_clean = W_norm(para_new, para_old)
    del para_new, para_old

    if args.metric == 'loss':
        clean_ratio = train_loss / init_loss
        extra_evaluation_metric_clean_train_loss.append(train_loss)
        extra_evaluation_metric_clean_init_loss.append(init_loss)
    else:
        clean_ratio = 1 - train_acc
    logger.print('Testing the final model')
    test_loss, test_acc = test(model, loss, args.epochs, test_loader, logger, test_logger, gpu, args.print_freq)
    train_measures = get_all_measures(model, init_model, train_no_noise_loader, train_acc, seed=args.seed)
    print(train_measures)

    noisy_generator = torch.Generator()
    noisy_ratio_list = []
    for repeat in range(args.repeat):
        train_loader, train_no_noise_loader, test_loader, model = init_data_and_model(args, gpu,
                                                                                      noisy_generator=noisy_generator)
        # para_old = model.state_dict()
        train_logger = TableLogger(os.path.join(result_dir, 'train_noise%d.log' % repeat), ['epoch', 'loss', 'acc'])
        test_logger = TableLogger(os.path.join(result_dir, 'test_noise%d.log' % repeat), ['epoch', 'loss', 'acc'])
        init_loss, _ = test(model, loss, -1, train_no_noise_loader, logger, train_logger, gpu, args.print_freq)
        if args.adam:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        schedule = create_schedule(args, len(train_loader), optimizer)
        for epoch in range(args.epochs):
            train(model, loss, epoch, train_loader, optimizer, schedule, logger, gpu, args.print_freq)
            train_loss, train_acc = test(model, loss, epoch, train_no_noise_loader, logger, train_logger, gpu,
                                         args.print_freq)
            if writer is not None:
                writer.add_scalar('curve noise%d/train loss' % repeat, train_loss, epoch)
                writer.add_scalar('curve noise%d/train acc' % repeat, train_acc, epoch)
            if args.metric == 'loss':
                noisy_ratio = train_loss / init_loss
                extra_evaluation_metric_noisy_train_loss.append(train_loss)
                extra_evaluation_metric_noisy_init_loss.append(init_loss)
            else:
                noisy_ratio = 1 - train_acc
        noisy_ratio_list.append(noisy_ratio)
    logger.print('clean distance: ', distance_clean)
    # logger.print('noisy ratios: ', distance_list)
    logger.print('clean ratio: ', clean_ratio)
    logger.print('train_measures', train_measures)
    logger.print('extra_evaluation_metric_clean_train_loss', extra_evaluation_metric_clean_train_loss)
    logger.print('extra_evaluation_metric_clean_init_loss', extra_evaluation_metric_clean_init_loss)
    logger.print('extra_evaluation_metric_noisy_train_loss', extra_evaluation_metric_noisy_train_loss)
    logger.print('extra_evaluation_metric_noisy_init_loss', extra_evaluation_metric_noisy_init_loss)
    logger.print('noisy ratio: ', noisy_ratio_list)
    if args.repeat == 0:
        logger.print('metirc: -1')
    else:
        logger.print('metirc: ', sum(noisy_ratio_list) / len(noisy_ratio_list) / clean_ratio)
    logger.print('clean test acc', test_acc)

    # torch.save({
    #     'state_dict': model.state_dict(),
    # }, os.path.join(result_dir, 'model.pth'))
    if writer is not None:
        writer.close()


def main(father_handle, **extra_argv):
    args = parser.parse_args()
    for key, val in extra_argv.items():
        setattr(args, key, val)
    result_dir = create_result_dir(args)
    if father_handle is not None:
        father_handle.put(result_dir)
    main_worker(args.gpu, args, result_dir)


if __name__ == '__main__':
    main(None)
