import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import h5py
import os
import json
import cv2
from torchvision import models
from tqdm import tqdm

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
    
    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, requires_grad=True).to(input_.device)
            # if torch.cuda.is_available():
            #     prev_state = torch.zeros(state_size, requires_grad=True).cuda()
            # else:
            #     prev_state = torch.zeros(state_size, requires_grad=True)
        
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes

        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells
    
    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*self.n_layers
        
        input_ = x
        upd_hidden = []
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            input_ = upd_cell_hidden
        
        return upd_hidden

class VGG_feat(nn.Module):
    def __init__(self, arch):
        super(VGG_feat, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)

def process_imgs(model, img_dir, batch_size, device=None):
    fns = os.listdir(img_dir)
    if 'vgg_feat.hdf5' in fns:
        fns.remove('vgg_feat.hdf5')
    h5f = h5py.File('{0}/vgg_feat.hdf5'.format(img_dir), 'w')
    with torch.no_grad():
        for i in tqdm(range(0, len(fns), batch_size)):
            batch = []
            for j in range(i, min(i+batch_size, len(fns))):
                batch.append(cv2.cvtColor(cv2.imread('{0}/{1}'.format(img_dir, fns[j])), cv2.COLOR_BGR2RGB))
            if device:
                imgs = torch.from_numpy(np.stack(batch)).float().to(device)
            else:
                imgs = torch.from_numpy(np.stack(batch)).float()
            imgs = imgs.permute(0, 3, 1, 2) / 255.0
            imgs[:, 0, :, :] = (imgs[:, 0, :, :] - 0.485) / 0.229
            imgs[:, 1, :, :] = (imgs[:, 1, :, :] - 0.456) / 0.224
            imgs[:, 2, :, :] = (imgs[:, 2, :, :] - 0.406) / 0.225
            feats = model(imgs)
            feats = feats.cpu().numpy()
            for j in range(i, min(i+batch_size, len(fns))):
                h5f[fns[j].split('.')[0]] = feats[j-i, :, :, :]
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = models.vgg16(pretrained=True).features
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    for i in range(16):
        process_imgs(model, 'data/videos/pass_0_{0}'.format(i), 8, device)