import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse
from utils.utils import *
from datasets.CelebA.dataloader import CelebaGroupedDataset
from models.resent_dual_mask import resnet18
import random
import math

print = logger.info
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str)
parser.add_argument('--dataroot', default='datasets/CelebA/', type=str)
parser.add_argument('--checkpoint_path', default='out')
parser.add_argument('--recover', default=False, type=bool)
parser.add_argument('--reco_type', default='last_checkpoint', type=str)
parser.add_argument('--total_epoch', default=40, type=int)
parser.add_argument('--image_size', default=64, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--logdir', default='logs', type=str)
parser.add_argument('--log_on', default=True, type=bool, help='whether to log the results')
parser.add_argument('--log_name', default='psa', type=str)
opt = parser.parse_args()
prepare_logging(opt)


set_seed(110)

task_groups = [
    [2, 10, 13, 14, 18, 20, 25, 26, 39],
    [3, 15, 23, 1, 12],
    [4, 5, 8, 9, 11, 17, 28, 32, 33],
    [6, 21, 31, 36],
    [7, 27],
    [0, 16, 22, 24, 30],
    [19, 29],
    [34, 35, 37, 38],
]

num_tasks = len(task_groups)
num_classes = sum([len(elt) for elt in task_groups])

# Define device, model and optimiser
gpu = check_gpu()
device = 0
model = resnet18(task_groups).to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epoch = 0

# Metrics
metrics = ['LOSS', 'ACC', 'PREC', 'REC', 'FSCORE']
metrics += ['LOSS_{}'.format(c) for c in range(num_classes)]
metrics += ['ACC_{}'.format(c) for c in range(num_classes)]
metrics += ['PREC_{}'.format(c) for c in range(num_classes)]
metrics += ['REC_{}'.format(c) for c in range(num_classes)]
metrics += ['FSCORE_{}'.format(c) for c in range(num_classes)]

# Create datasets
train_set = CelebaGroupedDataset(opt.dataroot,
                                 task_groups,
                                 split='train',
                                 image_size=opt.image_size)
val_set = CelebaGroupedDataset(opt.dataroot,
                               task_groups,
                               split='val',
                               image_size=opt.image_size)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=6)
val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=6)

writer = ResultWriter(log_on=opt.log_on, name=opt.name)
eval_header = ['LOSS_Train', 'ACC_Train', 'PREC_Train', 'REC_Train', 'FSCORE_Train',
               'LOSS_Test', 'ACC_Test', 'PREC_Test', 'REC_Test', 'FSCORE_Test']
writer.add_header("out", eval_header)

# Set different values and meters
train_batches = len(train_loader)
val_batches = len(val_loader)
nb_iter = 0

# Iterations
def training_and_eval(epoch, model, optimizer, total_epoch = opt.total_epoch, temp_c = 1, temp_l = 1):
    epocht = 0
    while epocht < total_epoch:
        # Train loop
        model.train()
        train_dataset = iter(train_loader)
        train_losses = torch.zeros(num_classes, dtype=torch.float32)
        train_well_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_to_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_pred = torch.zeros(num_classes, dtype=torch.float32)
        train_accs = torch.zeros(num_classes, dtype=torch.float32)
        for i in range(train_batches):
            # Get data
            data, targets = train_dataset.next()
            data, targets = data.to(device), [elt.to(device) for elt in targets]

            for task in range(num_tasks):
                # Set task
                model.change_task(task)
                model.change_temp_l(temp_l)
                model.change_temp_c(temp_c)
                target = targets[task]

                # Forward
                logits = torch.sigmoid(model(data))
                preds = (logits >= 0.5).type(torch.float32)
                losses = torch.mean(criterion(logits, target), 0)
                loss = torch.mean(losses)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scoring
                with torch.no_grad():
                    train_losses[task_groups[task]] += losses.cpu() / train_batches
                    train_pred[task_groups[task]] += torch.sum(preds.cpu(), dim=0)
                    train_to_pred[task_groups[task]] += torch.sum(target.cpu(), dim=0)
                    train_well_pred[task_groups[task]] += torch.sum((preds * target).cpu(), dim=0)
                    train_accs[task_groups[task]] += torch.mean((preds == target).cpu().type(torch.float32),
                                                                axis=0) / train_batches

            # Avg scores
            train_precs = train_well_pred / (train_pred + 1e-7)
            train_recs = train_well_pred / (train_to_pred + 1e-7)
            train_fscores = 2 * train_precs * train_recs / (train_precs + train_recs + 1e-7)

            # Out line
        #         print('Epoch {}, iter {}/{}, Loss : {}'.format(epoch, i+1, train_batches, loss.item()), end='\r')

        print('temp_c : ' + str(temp_c) + ',' + str(model.temp_c))
        print('temp_l : ' + str(temp_l) + ',' + str(model.temp_l))
        print('remaining_weights_c, remaining_weights_l: ' + str(model.compute_remaining_weights()) + ", " + str(
            model.compute_layer_remaining_weights()))
        print('remaining_origin_weights, compute_overall_weights : ' + str(
            model.compute_remaining_origin_weights()) + "," + str(model.compute_overall_weights()))
        # print('compute_overall_weights:' + str(model.compute_overall_weights()))

        # Eval loop
        model.eval()
        with torch.no_grad():
            val_dataset = iter(val_loader)
            val_losses = torch.zeros(num_classes, dtype=torch.float32)
            val_well_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_to_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_pred = torch.zeros(num_classes, dtype=torch.float32)
            val_accs = torch.zeros(num_classes, dtype=torch.float32)
            for i in range(val_batches):
                #             print('Eval iter {}/{}'.format(i+1, val_batches), end='\r')

                # Get data
                data, targets = val_dataset.next()
                data, targets = data.to(device), [elt.to(device) for elt in targets]

                for task in range(num_tasks):
                    # Set task
                    model.change_task(task)
                    model.change_temp_l(temp_l)
                    model.change_temp_c(temp_c)
                    target = targets[task]

                    # Forward
                    logits = torch.sigmoid(model(data))
                    preds = (logits >= 0.5).type(torch.float32)
                    losses = torch.mean(criterion(logits, target), 0)

                    # Scoring
                    val_losses[task_groups[task]] += losses.cpu() / val_batches
                    val_pred[task_groups[task]] += torch.sum(preds.cpu(), dim=0)
                    val_to_pred[task_groups[task]] += torch.sum(target.cpu(), dim=0)
                    val_well_pred[task_groups[task]] += torch.sum((preds * target).cpu(), dim=0)
                    val_accs[task_groups[task]] += torch.mean((preds == target).cpu().type(torch.float32),
                                                              axis=0) / val_batches

            # Avg scores
            val_precs = val_well_pred / (val_pred + 1e-7)
            val_recs = val_well_pred / (val_to_pred + 1e-7)
            val_fscores = 2 * val_precs * val_recs / (val_precs + val_recs + 1e-7)

            # Out line
            print('EVAL EPOCH {}, Train Loss : {}, acc : {}, prec : {}, rec : {}, f : {}'.format(epoch, torch.sum(
                train_losses).item(), torch.mean(train_accs).item(), torch.mean(train_precs).item(), torch.mean(
                train_recs).item(), torch.mean(train_fscores).item()))
            print('EVAL EPOCH {}, Loss : {}, acc : {}, prec : {}, rec : {}, f : {}'.format(epoch,
                                                                                           torch.sum(val_losses).item(),
                                                                                           torch.mean(val_accs).item(),
                                                                                           torch.mean(val_precs).item(),
                                                                                           torch.mean(val_recs).item(),
                                                                                           torch.mean(
                                                                                               val_fscores).item()))
        epocht =  epocht + 1
    return torch.sum(train_losses).item(), torch.sum(val_losses).item()


# warmup
flag_stop = 0

model.train()
model.change_policy_c(False)
model.change_policy_l(False)
training_and_eval(0, model, optimizer, 2, temp_c = 1, temp_l = 1)
model.train()
best_loss = 100000
temp_c, temp_l = 1, 1
v1 = 0
val_loss_tmp = 0


model.change_policy_c(None)
model.change_policy_l(None)
model.change_gumbel_noise(True)

for i in range(opt.total_epoch):
    model.train()
    t, v = training_and_eval(i, model, optimizer, 1, temp_c, temp_l)
    model.train()

    temp_c = 8 ** (i + 1)
    if temp_c > 2000: temp_c = 2000

    # 1500 0.9 3000 0.8 4096 0.7 16384 0.6
    temp_l = 8 ** (i + 1)
    if temp_l > 2000: temp_l = 2000


writer.close()