import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask(nn.Module):
    def __init__(self, channel, task_count, device=0, active_task=0):
        super(Mask, self).__init__()
        self.task_count = task_count
        self.active_task = active_task
        self.device = device
        self.alphas_channel = nn.Parameter(torch.zeros((task_count, channel)),requires_grad=True)
        self.policy_c = None
        self.temp_c = 1
        self.sigmoid = nn.Sigmoid()
        self.gumbel_noise = False

    def set_active_task(self, active_task):
        self.active_task = active_task
        return self.active_task

    def set_gumbel_noise(self, gumbel_noise):
        self.gumbel_noise = gumbel_noise
        return  self.gumbel_noise

    def set_temp_c(self, temp_c):
        self.temp_c = temp_c
        return self.temp_c

    def set_policy_c(self, policy_c):
        self.policy_c = policy_c
        return self.policy_c

    def bernoulli_sampling(self, masks):
        eps = torch.FloatTensor(masks.shape).uniform_(1e-6, 1 - 1e-6).cuda(self.device)
        soft = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + masks))
        return soft

    def forward(self, x, scaling = 2):
        eta = [0.1] * 8
        if self.policy_c is None:
            origin_mask = self.alphas_channel[self.active_task]
            mask = torch.sigmoid(self.temp_c * origin_mask)
            self.register_parameter('origin_task%d_mask' % (self.active_task + 1), nn.Parameter(mask, requires_grad=True))
            if self.gumbel_noise == True and self.training == True:
                temp_mask = self.bernoulli_sampling(mask)
                temp_mask = scaling * (temp_mask - torch.tensor(eta[self.active_task]).cuda(self.device)) + torch.tensor(eta[self.active_task]).cuda(self.device)
                mask = mask * temp_mask
            self.mask = F.hardtanh(mask, min_val=0, max_val=1)
            self.register_parameter('task%d_mask' % (self.active_task + 1), nn.Parameter(self.mask, requires_grad=True))
            return x * self.mask.view(1, -1, 1, 1)
        elif self.policy_c == False:
            origin_mask = self.alphas_channel[self.active_task]
            mask = torch.sigmoid(origin_mask)
            self.mask = scaling * mask
            self.register_parameter('task%d_mask' % (self.active_task + 1), nn.Parameter(self.mask, requires_grad=True))
            self.register_parameter('origin_task%d_mask' % (self.active_task + 1), nn.Parameter(self.mask, requires_grad=True))
            return x * mask.view(1, -1, 1, 1)
        else:
            self.register_parameter('task%d_mask' % (self.active_task + 1), nn.Parameter(self.mask, requires_grad=True))
            return x * self.mask.view(1, -1, 1, 1)