import os
import subprocess as sp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
import numpy as np
import torch
import random


def check_gpu():
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 1000
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu



class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        now = datetime.now()
        display_now = str(now).split(' ')[1][:-3]
        self.init(os.path.expanduser('~/'+str(display_now)), 'tmp.log')
        self._logger.info('[' + display_now + ']' + ' ' + str_info)        

        
logger = Logger()  


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)
        
        

def prepare_logging(args):
    logger.init(args.logdir, args.log_name)

    ensure_dir(args.logdir)
    logger.info("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    logger.info("========================================")
    
    

class ResultWriter:
    def __init__(self, log_on, name):
        str_time = name + '_' + time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        shutil.rmtree(f"results/{str_time}", True)
        self.model_folder = f"results/{str_time}/save_models"
        self.tensorboard_folder = f"results/{str_time}/runs"
        self.output_folder = f"results/{str_time}/outputs/"
        if log_on:
            os.makedirs(self.model_folder, exist_ok=True)
            os.makedirs(self.tensorboard_folder, exist_ok=True)
            os.makedirs(self.output_folder, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_folder)
        self.log_on = log_on

    def add_scalar(self, path, scaler, step):
        if self.log_on:
            self.writer.add_scalar(path, scaler, step)

    def add_scalars(self, path, scaler_dict, step):
        if self.log_on:
            self.writer.add_scalars(path, scaler_dict, step)

    def flush(self):
        if self.log_on:
            self.writer.flush()

    def close(self):
        if self.log_on:
            self.writer.close()

    def add_output(self, fname, data):
        if self.log_on:
            with open(self.output_folder + fname + ".csv", "ab") as f:
                # f.write(b"\n")
                np.savetxt(f, data, fmt='%.4f', delimiter=',')

    def add_header(self, fname, header):
        if self.log_on:
            with open(self.output_folder + fname + ".csv", "w") as f:
                f.write(','.join(header) + '\n')

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.model_folder, 'checkpoint.pth'))
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decay_temperature(temp, _tem_decay, decay_ratio=None):
    tmp = temp
    if decay_ratio is None:
        temp *= _tem_decay
    else:
        temp *= decay_ratio
#     print("Change temperature from %.5f to %.5f" % (tmp, temp))
    return temp