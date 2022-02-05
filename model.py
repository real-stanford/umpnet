import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch import nn

import spherical_sampling
from module_utils import MLP
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Conv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DirModel(nn.Module):
    def __init__(self, num_directions, model_type):
        super().__init__()
        self.num_directions = num_directions
        self.model_type = model_type
        self.raw_directions = spherical_sampling.fibonacci(num_directions, co_ords='cart')

        image_feature_dim = 256
        action_feature_dim = 128
        output_dim = 1
        self.sgn_action_encoder = MLP(3, action_feature_dim, [action_feature_dim, action_feature_dim])
        self.mag_action_encoder = MLP(3, action_feature_dim, [action_feature_dim, action_feature_dim])

        if 'sgn' in model_type:
            self.sgn_image_encoder_1 = Conv(20, 32)
            self.sgn_image_encoder_2 = Down(32, 64)
            self.sgn_image_encoder_3 = Down(64, 128)
            self.sgn_image_encoder_4 = Down(128, 256)
            self.sgn_image_encoder_5 = Down(256, 512)
            self.sgn_image_encoder_6 = Down(512, 512)
            self.sgn_image_encoder_7 = Down(512, 512)
            self.sgn_image_feature_extractor = MLP(512*7*10, image_feature_dim, [image_feature_dim])
            self.sgn_decoder = MLP(image_feature_dim + action_feature_dim, 3 * output_dim, [1024, 1024, 1024])

        if 'mag' in model_type:
            num_channels = 20 if model_type == 'mag' else 10
            self.mag_image_encoder_1 = Conv(num_channels, 32)
            self.mag_image_encoder_2 = Down(32, 64)
            self.mag_image_encoder_3 = Down(64, 128)
            self.mag_image_encoder_4 = Down(128, 256)
            self.mag_image_encoder_5 = Down(256, 512)
            self.mag_image_encoder_6 = Down(512, 512)
            self.mag_image_encoder_7 = Down(512, 512)
            self.mag_image_feature_extractor = MLP(512*7*10, image_feature_dim, [image_feature_dim])
            self.mag_decoder = MLP(image_feature_dim + action_feature_dim, output_dim, [1024, 1024, 1024])

        # Initialize random weights
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Conv3d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm3d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()


    def forward(self, observation, directions=None):
        if 'sgn' in self.model_type:
            x0 = observation
            x1 = self.sgn_image_encoder_1(x0)
            x2 = self.sgn_image_encoder_2(x1)
            x3 = self.sgn_image_encoder_3(x2)
            x4 = self.sgn_image_encoder_4(x3)
            x5 = self.sgn_image_encoder_5(x4)
            x6 = self.sgn_image_encoder_6(x5)
            x7 = self.sgn_image_encoder_7(x6)
            embedding = x7.reshape([x7.size(0), -1])
            sgn_feature = self.sgn_image_feature_extractor(embedding)
        if 'mag' in self.model_type:
            x0 = observation if self.model_type == 'mag' else observation[:, :10]
            x1 = self.mag_image_encoder_1(x0)
            x2 = self.mag_image_encoder_2(x1)
            x3 = self.mag_image_encoder_3(x2)
            x4 = self.mag_image_encoder_4(x3)
            x5 = self.mag_image_encoder_5(x4)
            x6 = self.mag_image_encoder_6(x5)
            x7 = self.mag_image_encoder_7(x6)
            embedding = x7.reshape([x7.size(0), -1])
            mag_feature = self.mag_image_feature_extractor(embedding)
        batch_size = observation.size(0)

        if directions is None:
            directions = list()
            for _ in range(observation.size(0)):
                r_mat_T = R.from_euler('xyz', np.random.rand(3) * 360, degrees=True).as_matrix().T
                directions.append(self.raw_directions @ r_mat_T)
            directions = np.asarray(directions)
        else:
            if len(directions.shape) == 2:
                directions = directions[:, np.newaxis]
        num_directions = directions.shape[1]
        torch_directions = torch.from_numpy(directions.astype(np.float32)).to(observation.device)
        sgn_direction_features = [self.sgn_action_encoder(torch_directions[:, i]) for i in range(num_directions)]
        mag_direction_features = [self.mag_action_encoder(torch_directions[:, i]) for i in range(num_directions)]

        sgn_output, mag_output = None, None
        if 'sgn' in self.model_type:
            sgn_output = list()
            for i in range(num_directions):
                feature_input = torch.cat([sgn_feature, sgn_direction_features[i]], dim=1)
                sgn_output.append(self.sgn_decoder(feature_input))
            sgn_output = torch.stack(sgn_output, dim=1)

        if 'mag' in self.model_type:
            mag_output = list()
            for i in range(num_directions):
                feature_input = torch.cat([mag_feature, mag_direction_features[i]], dim=1)
                mag_output.append(self.mag_decoder(feature_input))
            mag_output = torch.stack(mag_output, dim=1).squeeze(2)

        output = sgn_output, mag_output, directions
        return output


class Model():
    def __init__(self, num_directions, model_type):
        self.num_directions = num_directions
        self.model_type = model_type

        self.pos_model = UNet(10, 2)
        self.dir_model = DirModel(num_directions, model_type)
    
    def get_direction_affordance(self, observations, model_type, torch_tensor=False, directions=None):
        """Get position affordance maps.

        Args:
            observations: list of dict
                - image: [W, H, 10]. dtype: float32
                - image_init: [W, H, 10]. dtype: float32
            model_type: 'sgn', 'mag', 'sgn_mag'
            torch_tensor: Whether the retuen value is torch tensor (default is numpy array). torch tensor is used for training.
        Return:
            affordance_maps: numpy array/torch tensor, [B, K, W, H]
            directions: list of direction vector
        """
        skip_id_list = list()
        scene_inputs = []
        for id, observation in enumerate(observations):
            if observation is None:
                skip_id_list.append(id)
                continue
            scene_inputs.append(np.concatenate([observation['image'].transpose([2, 0, 1]), observation['image_init'].transpose([2, 0, 1])], axis=0))

        scene_input_tensor = torch.from_numpy(np.stack(scene_inputs))
        sgn_output, mag_output, skipped_directions = self.dir_model.forward(scene_input_tensor.to(self.device_dir), directions=directions) # [B, K, W, H]
        if torch_tensor:
            assert len(skip_id_list) == 0
            return sgn_output, mag_output, None
        else:
            if model_type == 'sgn':
                affordance_maps = 1 - F.softmax(sgn_output, dim=2)[:, :, 1]
            elif model_type == 'mag':
                affordance_maps = mag_output
            elif model_type == 'sgn_mag':
                sgn = sgn_output.max(2)[1] - 1
                affordance_maps = sgn * F.relu(mag_output)

            skipped_affordance_maps = affordance_maps.data.cpu().numpy()

            affordance_maps = list()
            directions = list()
            cur = 0
            for id in range(len(skipped_affordance_maps)+len(skip_id_list)):
                if id in skip_id_list:
                    affordance_maps.append(None)
                    directions.append(None)
                else:
                    affordance_maps.append(skipped_affordance_maps[cur])
                    directions.append(skipped_directions[cur])
                    cur += 1

        return affordance_maps, directions
    

    def get_position_affordance(self, observations, torch_tensor=False):
        """Get position affordance maps.

        Args:
            observations: list of dict
                - image: [W, H, 10]. dtype: float32
            torch_tensor: Whether the retuen value is torch tensor (default is numpy array). torch tensor is used for training.
        Return:
            affordance_maps: numpy array/torch tensor, [B, K, W, H]
        """
        skip_id_list = list()
        scene_inputs = []
        for observation in observations:
            scene_inputs.append(observation['image'].transpose([2, 0, 1]))
        scene_input_tensor = torch.from_numpy(np.stack(scene_inputs))

        affordance_maps = self.pos_model.forward(scene_input_tensor.to(self.device_pos)) # [B, K, W, H]
        if not torch_tensor:
            affordance_maps = 1 - F.softmax(affordance_maps, dim=1)[:, 0]
            affordance_maps = affordance_maps.data.cpu().numpy()

        return affordance_maps

    def to(self, device_pos, device_dir):
        self.device_pos = device_pos
        self.device_dir = device_dir
        self.pos_model = self.pos_model.to(device_pos)
        self.dir_model = self.dir_model.to(device_dir)
        return self
    
    def eval(self):
        self.pos_model.eval()
        self.dir_model.eval()

    def train(self):
        self.pos_model.train()
        self.dir_model.train()
