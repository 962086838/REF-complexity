import numpy as np
import torch
import os
import random
import re


def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_result_dir(args):
    result_dir = ''
    mp = {
        'dataset': '',
        'model': '',
        'loss': '',
        'epochs': 'epoch',
        'decays': None,
        'batch_size': 'bs',
        'start_epoch': None,
        'checkpoint': None,
        'gpu': None,
        'print_freq': None,
        'result_dir': None,
        'filter_name': '',
        'seed': '',
        'visualize': None,
    }
    for arg in vars(args):
        if arg in mp and mp[arg] is None:
            continue
        value = getattr(args, arg)
        if type(value) == bool:
            value = 'T' if value else 'F'
        if type(value) == list:
            value = str(value).replace(' ', '')
        name = mp.get(arg, arg)
        result_dir += name + str(value) + '_'
    return os.path.join(args.result_dir, result_dir)


def create_result_dir(args):
    result_dir = get_result_dir(args)
    id = 0

    result_dir_id = result_dir + '_%d' % id
    if os.path.exists(result_dir_id):
        assert 1==0, "already done"

    while True:
        result_dir_id = result_dir + '_%d'%id
        if not os.path.exists(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    return result_dir_id


class Logger(object):
    def __init__(self, dir):
        self.fp = open(dir, 'w')

    def __del__(self):
        self.fp.close()

    def print(self, *args, **kwargs):
        print(*args, file=self.fp, **kwargs)
        print(*args, **kwargs)


class TableLogger(object):
    def __init__(self, path, header):
        import csv
        self.fp = open(path, 'w')
        self.logger = csv.writer(self.fp, delimiter='\t')
        self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.fp.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.fp.flush()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_function_call(s):
    def convert_types(params, items):
        for key, value in items:
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    special = {'True': True, 'False': False, 'None': None}
                    try:
                        params[key] = special[value]
                    except KeyError:
                        pass

    s = re.split(r'[()]', s)
    if len(s) == 1:
        return s[0], [], {}
    name, params, _ = s
    params = re.split(r',\s*', params)
    args = [p for p in params if '=' not in p]
    kwargs = dict([p.split('=') for p in params if '=' in p])
    convert_types(args, enumerate(args))
    convert_types(kwargs, kwargs.items())
    return name, args, kwargs