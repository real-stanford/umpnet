import json
import os
import pickle

import numpy as np

mobility_path = 'mobility_dataset'
split_file = 'split-full.json'
split_meta = json.load(open(os.path.join(mobility_path, split_file), 'r'))


def fetch_mobility_object(category_type='train', instance_type = 'train', category_name=None, instance_id=None):
    if category_name is None:
        category_name = np.random.choice(list(split_meta[category_type].keys()))
    if instance_id is None:
        instance_id = np.random.choice(split_meta[category_type][category_name][instance_type])
    
    urdf_path = os.path.join(mobility_path, category_name, instance_id, 'mobility.urdf')
    with open(os.path.join(mobility_path, category_name, instance_id, 'object_meta_info.json'), 'r') as f:
        meta_info = json.load(f)
    orientation = meta_info['orientation']
    offset = [0, 0, meta_info['offset_z']]
    scale = meta_info['scale']
    return category_name, instance_id, urdf_path, orientation, offset, scale, meta_info['moveable_link']


def fetch_toy_object(object_id=None):
    if object_id is None:
        object_id = np.random.choice(['toy1', 'toy2'])
    urdf_path = os.path.join('assets', 'toys', object_id, 'toy.urdf')
    rotation = [0, 0, 0]
    offset_z = 0.2
    with open(os.path.join('assets', 'toys', object_id, 'init_state.pkl'), 'rb') as f:
        position_list = pickle.load(f)
    L = len(position_list)
    init_state = position_list[np.random.choice(L - L // 5) + L // 10]
    return urdf_path, rotation, offset_z, init_state