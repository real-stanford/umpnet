import argparse
import json
import os
import pickle

import numpy as np
import torch

import spherical_sampling
import train
from model import Model
from sim import PybulletSim


parser = argparse.ArgumentParser()

# global
parser.add_argument('--checkpoint', default='pretrained/umpnet.pth', type=str, help='path to the checkpoint')
parser.add_argument('--mode', default='manipulation', type=str, choices=['exploration', 'manipulation'], help='type of test mode')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')
parser.add_argument('--category', default='Refrigerator', type=str, help='testing category')

# model
parser.add_argument('--model_type', default='sgn_mag', type=str, choices=['sgn', 'mag', 'sgn_mag'], help='model_type')

# environment
parser.add_argument('--num_direction', default=64, type=int, help='number of directions')
parser.add_argument('--no_cem', action='store_true', help='without cem')
parser.add_argument('--action_distance', default=0.18, type=float, help='dragging distance in each interaction')

step_num_dict = {
    'Refrigerator': 12,
    'FoldingChair': 8,
    'Laptop': 12,
    'Stapler': 15,
    'TrashCan': 9,
    'Microwave': 8,
    'Toilet': 7,
    'Window': 6,
    'StorageFurniture': 9,
    'Switch': 7,
    'Kettle': 3,
    'Toy': 10,
    'Box': 10,
    'Phone': 12,
    'Dishwasher': 10,
    'Safe': 10,
    'Oven': 9,
    'WashingMachine': 9,
    'Table': 7,
    'KitchenPot': 3,
    'Bucket': 13,
    'Door': 10
}

def main():
    args = parser.parse_args()

    mobility_path = 'mobility_dataset'
    split_file = 'split-full.json'
    split_meta = json.load(open(os.path.join(mobility_path, split_file), 'r'))

    # Load model
    device = torch.device(f'cpu')
    model = Model(num_directions=args.num_direction, model_type=args.model_type)
    model = model.to(device, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.pos_model.load_state_dict(checkpoint['pos_state_dict'])
    print('==> pos model loaded')
    model.dir_model.load_state_dict(checkpoint['dir_state_dict'])
    print('==> dir model loaded')
    model.eval()
    torch.set_grad_enabled(False)

    run_test(args, model, args.category, 'test')

def run_test(args, model, category_name, instance_type):
    print(f'==> run test: {args.mode} - {category_name} - {instance_type}')

    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    max_step_num = step_num_dict[category_name]

    # test data info
    test_data_path = os.path.join('test_data', args.mode, category_name, instance_type)
    test_num = 100

    sim = PybulletSim(True, args.action_distance)
    
    while True:
        id = np.random.choice(test_num)
        scene_state = pickle.load(open(os.path.join(test_data_path, f'{id}.pkl'), 'rb'))
        observation = sim.reset(scene_state=scene_state)

        # position inference
        position_affordance = model.get_position_affordance([observation])[0]
        if args.mode == 'manipulation':
            pixel_dist = np.sum((observation['image'][:, :, :3] - observation['image_init'][:, :, :3]) ** 2, axis=2)
            diff_mask = (pixel_dist > 1e-5).astype(np.float)
            position_affordance *= diff_mask

        action, score = train.get_position_action(position_affordance, epsilon=0, image=observation['image'], prev_actions=list())
        observation, (reward, move_flag), done, info = sim.step([0, action[0], action[1]])

        # terminate immediately if the position is wrong
        if done:
            print('Wrong position prediction. Terminate immediately!')
            continue

        # pre-preparation
        reach_boundary, reach_init = False, False
        bad_actions = list()

        # direction inference
        for step in range(1, max_step_num + 1):
            direction_affordance, directions = model.get_direction_affordance([observation], model_type=args.model_type)
            direction_affordance = direction_affordance[0]
            directions = directions[0]

            # remove bad actions
            for bad_action in bad_actions:
                dist_map = np.sum((directions - bad_action) ** 2, axis=1)
                ban_idx_list = np.argsort(dist_map)[:1]
                for idx in ban_idx_list:
                    direction_affordance[idx] = 0

            # CEM
            if not args.no_cem:
                prob = np.exp(direction_affordance * 20) if args.mode == 'exploration' else np.exp(-direction_affordance * 20)
                prob /= np.sum(prob)
                new_direction_ids = np.random.choice(args.num_direction, args.num_direction, replace=True, p=prob)
                noise_candidates = spherical_sampling.fibonacci(1024, co_ords='cart')
                noise_id = np.random.choice(1024, args.num_direction)
                new_directions = np.zeros([args.num_direction, 3])
                for dir_id in range(args.num_direction):
                    vec = directions[new_direction_ids[dir_id]]
                    vec += noise_candidates[noise_id[dir_id]] / np.sqrt(args.num_direction) * 2
                    vec /= np.sqrt(np.sum(vec ** 2))
                    new_directions[dir_id] = vec
                new_direction_affordance, _ = model.get_direction_affordance([observation], model_type=args.model_type)
                new_direction_affordance = new_direction_affordance[0]
                directions = new_directions
                direction_affordance = new_direction_affordance

            action_direction='positive' if args.mode == 'exploration' else 'negative'
            action_index, score = train.get_direction_action(direction_affordance, None, 0, action_direction=action_direction)
            action = directions[action_index]
            observation, (reward, move_flag), (reach_init, reach_boundary), info = sim.step([1, action[0], action[1], action[2]])

            if args.mode == 'exploration' and reach_boundary:
                print("Boundary reached.")
                break
            if args.mode == 'manipulation'and reach_init:
                print("Target reached.")
                break

            # remove bad action
            if move_flag:
                bad_actions = list()
            else:
                bad_actions.append(action)

if __name__=='__main__':
    main()