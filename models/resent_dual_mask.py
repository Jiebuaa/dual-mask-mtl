import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mask import Mask
import numpy as np

class ConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, norm_layer=None, num_tasks=4):
        super(ConvLayer, self).__init__()
        modules = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)]
        if norm_layer is not None:
            modules.append(norm_layer(num_features=out_planes, track_running_stats=False))
        modules.append(Mask(out_planes, num_tasks))
        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)

    def get_weight(self):
        return self.conv_block[0].weight


def get_blocks(model):
    if isinstance(model, ConvLayer):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child)
    return blocks


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64, norm_layer=None, num_tasks=4, sigma=0.4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvLayer(inplanes, planes, kernel_size=3, padding=1, stride=stride, norm_layer=norm_layer,
                               num_tasks=num_tasks)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, padding=1, norm_layer=norm_layer, num_tasks=num_tasks)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.len = 3 if downsample is not None else 2
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, task_groups,
                 width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        self.active_task = 0
        self.num_tasks = len(task_groups)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.device = 0

        self.temp_c = 1
        self.temp_l = 1
        self.policy_c = None
        self.policy_l = None
        self.gumbel_noise = False

        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = ConvLayer(3, self.inplanes, kernel_size=7, stride=2, padding=3, norm_layer=norm_layer,
                               num_tasks=self.num_tasks)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()
        for task in range(len(task_groups)):
            self.fc.append(nn.Linear(512, len(task_groups[task])))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.reset_logits()

    def reset_logits(self):
        num_layers = 8
        for t_id in range(8):
            layer_logits = torch.zeros(num_layers)
            self.register_parameter('layer%d_logits' % (t_id + 1), nn.Parameter(layer_logits, requires_grad=True))

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ConvLayer(self.inplanes, planes, kernel_size=1, stride=stride, num_tasks=self.num_tasks)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, norm_layer, num_tasks=self.num_tasks))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                base_width=self.base_width, norm_layer=norm_layer,
                                num_tasks=self.num_tasks))

        return layers

    def change_task(self, task):
        def aux(m):
            if hasattr(m, 'active_task'):
                m.set_active_task(task)
        self.apply(aux)

    def set_active_task(self, active_task):
        self.active_task = active_task
        return self.active_task

    def change_temp_c(self, temp_c):
        def aux(m):
            if hasattr(m, 'temp_c'):
                m.set_temp_c(temp_c)
        self.apply(aux)

    def set_temp_c(self, temp_c):
        self.temp_c = temp_c
        return self.temp_c

    def change_temp_l(self, temp_l):
        def aux(m):
            if hasattr(m, 'temp_l'):
                m.set_temp_l(temp_l)
        self.apply(aux)

    def set_temp_l(self, temp_l):
        self.temp_l = temp_l
        return self.temp_l

    def set_policy_c(self, policy_c):
        self.policy_c = policy_c
        return self.policy_c

    def change_policy_c(self, policy_c):
        def aux(m):
            if hasattr(m, 'policy_c'):
                m.set_policy_c(policy_c)

        self.apply(aux)

    def set_policy_l(self, policy_l):
        self.policy_l = policy_l
        return self.policy_l

    def change_policy_l(self, policy_l):
        def aux(m):
            if hasattr(m, 'policy_l'):
                m.set_policy_l(policy_l)

        self.apply(aux)

    def saved_policy_c(self, epoch):
        mask = [[], [], []]
        for n, m in self.named_parameters():
            for i in range(8):
                if 'alphas_channel.%d' % (i) in n:
                    mask[i].append(m)
        np.save("results/policys" + str(epoch) + ".npy", mask)

    def sparisy_loss(self):
        masks = []
        for n, m in self.named_parameters():
            for i in range(8):
                if 'task%d_mask' % (i + 1) in n:
                    masks.append(m)
        entries_sum = (sum(m.sum() for m in masks) / sum(m.numel() for m in masks))
        return entries_sum

    def set_gumbel_noise(self, gumbel_noise):
        self.gumbel_noise = gumbel_noise
        return self.gumbel_noise

    def change_gumbel_noise(self, gumbel_noise):
        def aux(m):
            if hasattr(m, 'gumbel_noise'):
                m.set_gumbel_noise(gumbel_noise)

        self.apply(aux)

    def get_masks(self):
        mask = [[], [], [], [], [], [], [], []]
        masks = []
        for n, m in self.named_parameters():
            for i in range(8):
                if 'task%d_mask' % (i + 1) in n:
                    mask[i].append(m)
                    masks.append(m)
        return mask, masks

    def get_Layer_masks(self):
        masks = []
        for n, m in self.named_parameters():
            for i in range(8):
                if 'layer%d_mask' % (i + 1) in n:
                    masks.append(m)
        return masks

    def get_origin_masks(self):
        mask = [[], [], [], [], [], [], [], []]
        masks = []
        for n, m in self.named_parameters():
            for i in range(8):
                if 'origin_task%d_mask' % (i + 1) in n:
                    mask[i].append(m)
                    masks.append(m)
        return mask, masks

    def get_origin_Layer_masks(self):
        mask = [[], [], [], [], [], [], [], []]
        masks = []
        for n, m in self.named_parameters():
            for i in range(8):
                if 'origin_layer%d_mask' % (i + 1) in n:
                    mask[i].append(m)
                    masks.append(m)
        return mask, masks

    def saved_mask(self, name):
        mask, _ = self.get_masks()
        mask = [i.data.cpu().numpy() for v in mask for i in v]
        np.save("masks1227/mask" + name + ".npy", mask)
        return mask

    def saved_layer_mask(self, name):
        mask = self.get_Layer_masks()
        mask = [i.data.cpu().numpy() for v in mask for i in v]
        np.save("masks1227/layer_mask" + name + ".npy", mask)
        return mask

    def compute_remaining_weights(self):
        _, masks = self.get_masks()
        return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)

    def compute_layer_remaining_weights(self):
        masks = self.get_Layer_masks()
        return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)

    def compute_remaining_origin_weights(self):
        _, masks = self.get_origin_masks()
        return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)


    def compute_overall_weights(self):
        _, masks_layer = self.get_origin_Layer_masks()
        masks, masks1 = self.get_origin_masks()
        overall = 0
        for k in range(len(masks_layer)):
            j = 0
            for i in range(len(masks[k])):
                if i not in [0, 1, 4, 5, 8, 9, 14, 15]:
                    if float(masks_layer[k][int(j / 2)] == 0) == 1:
                        overall = overall + masks[k][int(j / 2)].numel()
                    else:
                        overall = overall + float((masks[k][i] == 0).sum())
                    j = j + 1
                else:
                    overall = overall + float((masks[k][i] == 0).sum())
        return 1 - overall / sum(m.numel() for m in masks1)

    def temp_l_sample_policy(self):
        scaling = 2
        if self.policy_l is None:
            eta = [0.1] * 8
            origin_mask = getattr(self, 'layer%d_logits' % (self.active_task + 1))
            mask = torch.sigmoid(self.temp_l * origin_mask)
            self.register_parameter('origin_layer%d_mask' % (self.active_task + 1),
                                    nn.Parameter(mask, requires_grad=True))
            if self.gumbel_noise == True and self.training == True:
                eps = torch.FloatTensor(origin_mask.shape).uniform_(1e-6, 1 - 1e-6).cuda(self.device)
                temp_mask = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + mask))
                temp_mask = scaling * (temp_mask - torch.tensor(eta[self.active_task]).cuda(self.device)) + torch.tensor(
                    eta[self.active_task]).cuda(self.device)
                mask = mask * temp_mask
            self.mask = F.hardtanh(mask, min_val=0, max_val=1)
            self.register_parameter('layer%d_mask' % (self.active_task + 1),
                                    nn.Parameter(self.mask, requires_grad=True))
        else:
            origin_mask = getattr(self, 'layer%d_logits' % (self.active_task + 1))
            mask = torch.sigmoid(origin_mask)
            self.mask = scaling * mask
            self.register_parameter('layer%d_mask' % (self.active_task + 1),
                                    nn.Parameter(self.mask, requires_grad=True))
            self.register_parameter('origin_layer%d_mask' % (self.active_task + 1),
                                    nn.Parameter(self.mask, requires_grad=True))
        return self.mask


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        policy = self.temp_l_sample_policy().to(self.device)

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i != 0:
                x = policy[self.active_task] * self.layer1[i](x) + x * (1 - policy[self.active_task])

        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i != 0:
                x = policy[self.active_task] * self.layer2[i](x) + x * (1 - policy[self.active_task])

        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i != 0:
                x = policy[self.active_task] * self.layer3[i](x) + x * (1 - policy[self.active_task])

        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i != 0:
                x = policy[self.active_task] * self.layer4[i](x) + x * (1 - policy[self.active_task])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc[self.active_task](x)

        return x


def resnet18(task_groups, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], task_groups, **kwargs)