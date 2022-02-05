import argparse
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import utils
from model import Model
from replay_buffer import ReplayBuffer
from sim import PybulletSim

parser = argparse.ArgumentParser()

# global
parser.add_argument('--exp', default='exp', type=str, help='name of experiment. The directory to save data is exp/[exp]')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')
parser.add_argument('--snapshot_gap', default=1, type=int, help='Frequence of saving the snapshot (e.g. visualization, model, optimizer)')
parser.add_argument('--num_visualization', default=None, type=int, help='numer of visualization sequences, None means num_envs')

# environment
parser.add_argument('--num_envs', default=16, type=int, help='number of envs, each env has a process')
parser.add_argument('--max_seq_len', default=4, type=int, help='number of steps for each sequence')
parser.add_argument('--increase_seq_len_start_epoch', default=1000, type=int, help='start to increase seq_len from (inclusive), -1 means not increase')
parser.add_argument('--increase_seq_len_gap', default=400, type=int, help='epohc gap to seq_len += 2')
parser.add_argument('--max_seq_len_limit', default=20, type=int, help='maximum seq_len')
parser.add_argument('--num_direction', default=64, type=int, help='number of directions')
parser.add_argument('--action_distance', default=0.18, type=float, help='dragging distance in each interaction')

# model
parser.add_argument('--model_type', default='sgn_mag', type=str, choices=['sgn', 'mag', 'sgn_mag'], help='model_type')

# training
parser.add_argument('--load_checkpoint', default=None, type=str, help='exp name or a directpry of ckpt (suffix is .pth). Load the the checkpoint (model, optimizer) from another training exp')
parser.add_argument('--load_model_type', default=[], type=str, nargs='+', help='pos or dir')
parser.add_argument('--learning_rate', default=8e-3, type=float, help='learning rate of the optimizer')
parser.add_argument('--learning_rate_decay', default=500, type=int, help='learning rate decay')
parser.add_argument('--epoch', default=5000, type=int, help='How many training epochs')
parser.add_argument('--pos_iter_per_epoch', default=8, type=int, help='numer of traininig iterations per epoch (pos)')
parser.add_argument('--dir_iter_per_epoch', default=8, type=int, help='numer of traininig iterations per epoch (dir)')
parser.add_argument('--pos_batch_size', default=16, type=int, help='batch size for position training')
parser.add_argument('--dir_batch_size', default=24, type=int, help='batch size for direction training')

# replay buffer
parser.add_argument('--load_replay_buffer', default=None, type=str, help='exp name. Load the replay buffer from another training exp')
parser.add_argument('--replay_buffer_size', default=6400, type=int, help='maximum size of replay buffer')

# policy
parser.add_argument('--position_min_epsilon', default=0.1, type=float, help='(position selection) minimal epsilon in data collection')
parser.add_argument('--position_decay_epoch', default=300, type=int, help='(position selection) how many epoches to decay from 1 to min_epsilon')
parser.add_argument('--position_start_epoch', default=500, type=int, help='(position selection) start epoch of training')

parser.add_argument('--direction_min_epsilon', default=0.2, type=float, help='(direction selection) minimal epsilon in data collection')
parser.add_argument('--direction_decay_epoch', default=500, type=int, help='(direction selection)how many epoches to decay from 1 to min_epsilon')
parser.add_argument('--direction_start_epoch', default=800, type=int, help='(direction selection) start epoch of training')


def main():
    args = parser.parse_args()

    # Set exp directory and tensorboard writer
    writer_dir = os.path.join('exp', args.exp)
    utils.mkdir(writer_dir)
    writer = SummaryWriter(writer_dir)

    # Save arguments
    str_list = []
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
        str_list.append('--{0}={1} \\'.format(key, getattr(args, key)))
    with open(os.path.join('exp', args.exp, 'args.txt'), 'w+') as f:
        f.write('\n'.join(str_list))

    # Set directory. e.g. replay buffer, visualization, model snapshot
    args.replay_buffer_dir = os.path.join('exp', args.exp, 'replay_buffer')
    args.visualization_dir = os.path.join('exp', args.exp, 'visualization')
    utils.mkdir(args.visualization_dir)
    args.model_dir = os.path.join('exp', args.exp, 'models')
    utils.mkdir(args.model_dir)

    # Reset random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialization of model, optimizer, replay buffer
    model = Model(num_directions=args.num_direction, model_type=args.model_type)
    pos_optimizer = torch.optim.Adam(model.pos_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    dir_optimizer = torch.optim.Adam(model.dir_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    pos_scheduler = torch.optim.lr_scheduler.StepLR(pos_optimizer, step_size=args.learning_rate_decay, gamma=0.5)
    dir_scheduler = torch.optim.lr_scheduler.StepLR(dir_optimizer, step_size=args.learning_rate_decay, gamma=0.5)
    replay_buffer = ReplayBuffer(args.replay_buffer_dir, args.replay_buffer_size)

    # Set device
    device_pos = torch.device(f'cuda:0')
    device_dir = torch.device(f'cuda:0')
    model = model.to(device_pos, device_dir)

    if args.load_replay_buffer is not None:
        print(f'==> Loading replay buffer from {args.load_replay_buffer}')
        replay_buffer.load(os.path.join('exp', args.load_replay_buffer, 'replay_buffer'))
        print(f'==> Loaded replay buffer from {args.load_replay_buffer} [size = {replay_buffer.length}]')

    if args.load_checkpoint is not None:
        print(f'==> Loading checkpoint from {args.load_checkpoint}')
        if args.load_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.load_checkpoint, map_location=device_pos)
        else:
            checkpoint = torch.load(os.path.join('exp', args.load_checkpoint, 'models', 'latest.pth'), map_location=device_pos)
        if 'pos' in args.load_model_type:
            model.pos_model.load_state_dict(checkpoint['pos_state_dict'])
            pos_optimizer.load_state_dict(checkpoint['pos_optimizer'])
            print('==> pos model loaded')
        if 'dir' in args.load_model_type:
            model.dir_model.load_state_dict(checkpoint['dir_state_dict'])
            dir_optimizer.load_state_dict(checkpoint['dir_optimizer'])
            print('==> dir model loaded')
        start_epoch = 0
        del checkpoint
        print(f'==> Loaded checkpoint from {args.load_checkpoint}')
    else:
        start_epoch = 0

    
    for g in pos_optimizer.param_groups:
        g['lr'] = args.learning_rate
    for g in dir_optimizer.param_groups:
        g['lr'] = args.learning_rate

    # launch processes for each env
    processes, conns = [], []
    ctx = mp.get_context('spawn')
    env_arguments = {
        'action_distance': args.action_distance,
    }
    for rank in range(args.num_envs):
        conn_main, conn_env = ctx.Pipe()
        p = ctx.Process(target=env_process, args=(rank, start_epoch + args.seed + rank, conn_env, env_arguments))
        p.daemon=True
        p.start()
        processes.append(p)
        conns.append(conn_main)

    # Initialize exit signal handler (for graceful exits)
    def save_and_exit(signal, frame):
        print('Warning: keyboard interrupt! Cleaning up...')
        for p in processes:
            p.terminate()
        replay_buffer.dump()
        writer.close()
        print('Finished. Now exiting gracefully.')
        sys.exit(0)
    signal.signal(signal.SIGINT, save_and_exit)

    for epoch in range(start_epoch, args.epoch):
        print(f'---------- epoch-{epoch + 1} ----------')
        timestamp = time.time()

        if args.increase_seq_len_start_epoch != -1 and args.max_seq_len < args.max_seq_len_limit and epoch >= args.increase_seq_len_start_epoch and epoch % args.increase_seq_len_gap == 0:
            args.max_seq_len += 2
        
        print('==> max_seq_len = ', args.max_seq_len)

        # Data collection
        data = collect_data(
            conns, model, args.model_type,
            max_seq_len=args.max_seq_len,
            position_epsilon=args.position_min_epsilon + max(0, (1 - (epoch - args.position_start_epoch) / args.position_decay_epoch) * (1 - args.position_min_epsilon)),
            direction_epsilon=args.direction_min_epsilon + max(0, (1 - (epoch - args.direction_start_epoch) / args.direction_decay_epoch) * (1 - args.direction_min_epsilon)),
            category_type='train',
            instance_type='train'
        )

        for d in data.values():
            replay_buffer.save_data(d)
            
        pos_move_list, pos_reward_list = list(), list()
        dir_move_list, dir_reward_list = list(), list()

        for key, val in data.items():
            if val['type'] == 0:
                pos_reward_list.append(val['reward'])
                pos_move_list.append(val['move_flag'])
            else:
                dir_move_list.append(int(val['move_flag']))
                dir_reward_list.append(abs(val['reward']))

        mean_pos_move = np.mean(pos_move_list) if len(pos_move_list) != 0 else 0
        mean_pos_reward = np.mean(pos_reward_list) if len(pos_reward_list) != 0 else 0
        mean_dir_move = np.mean(dir_move_list) if len(dir_move_list) != 0 else 0
        mean_dir_reward = np.mean(dir_reward_list) if len(dir_reward_list) != 0 else 0

        print(f'Data Collection. mean_pos_reward = {mean_pos_reward}, mean_dir_move = {mean_dir_move}, mean_dir_reward = {mean_dir_reward}')

        writer.add_scalar('Data Collection/Position-move accuracy', mean_pos_move, epoch + 1)
        writer.add_scalar('Data Collection/Position accuracy', mean_pos_reward, epoch + 1)
        writer.add_scalar('Data Collection/Direction-move accuracy', mean_dir_move, epoch + 1)
        writer.add_scalar('Data Collection/Direction-val Magnitude', mean_dir_reward, epoch + 1)

        time_data_collection = time.time() - timestamp

        # Replay buffer statistic
        type_data = np.array(replay_buffer.scalar_data['type'])
        move_flag_data = np.array(replay_buffer.scalar_data['move_flag'])
        pos_positive_num = np.sum(np.logical_and(type_data == 0, move_flag_data == True))
        pos_negative_num = np.sum(np.logical_and(type_data == 0, move_flag_data == False))
        dir_positive_num = np.sum(np.logical_and(type_data == 1, move_flag_data == True))
        dir_negative_num = np.sum(np.logical_and(type_data == 1, move_flag_data == False))
        print(f'Replay buffer size = {len(type_data)}, pos(p+n) = {pos_positive_num}+{pos_negative_num}, dir(p+n) = {dir_positive_num}+{dir_negative_num}')

        writer.add_scalar('Replay Buffer/Position-positive', pos_positive_num, epoch + 1)
        writer.add_scalar('Replay Buffer/Position-negative', pos_negative_num, epoch + 1)
        writer.add_scalar('Replay Buffer/Direction-positive', dir_positive_num, epoch + 1)
        writer.add_scalar('Replay Buffer/Direction-negative', dir_negative_num, epoch + 1)

        # Policy training
        iter_info= list()
        if epoch >= args.position_start_epoch:
            iter_info.append(('pos', args.pos_iter_per_epoch))
        if epoch >= args.direction_start_epoch:
            iter_info.append(('dir', args.dir_iter_per_epoch))

        if len(iter_info) == 0:
            print('skip training')
            continue

        loss_summary = dict()
        for train_model_type, num_iters in iter_info:
            for _ in range(num_iters):
                loss_info = train(model, replay_buffer, pos_optimizer, dir_optimizer, args.pos_batch_size, args.dir_batch_size, args.model_type, device_pos, device_dir, [train_model_type])
                for k in loss_info:
                    if not k in loss_summary:
                        loss_summary[k] = list()
                    loss_summary[k].append(loss_info[k])
        print_str = 'Training loss: '
        for k in loss_summary:
            loss_avg = np.mean(loss_summary[k])
            print_str += f' {k} = {loss_avg:.4f}'
            writer.add_scalar(f'Policy Training/Loss-{k}', loss_avg, epoch + 1)
        print(print_str)

        # Step scheduler
        pos_scheduler.step()
        dir_scheduler.step()

        if (epoch + 1) % args.snapshot_gap == 0:
            # Save model and optimizer
            save_state = {
                'pos_state_dict': model.pos_model.state_dict(),
                'dir_state_dict': model.dir_model.state_dict(),
                'pos_optimizer': pos_optimizer.state_dict(),
                'dir_optimizer': dir_optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(save_state, os.path.join(args.model_dir, 'latest.pth'))
            shutil.copyfile(
                os.path.join(args.model_dir, 'latest.pth'),
                os.path.join(args.model_dir, 'epoch_%06d.pth' % (epoch + 1))
            )

            # Save replay buffer
            replay_buffer.dump()

        # Print elapsed time for an epoch
        time_all = time.time() - timestamp
        time_training = time_all - time_data_collection
        print(f'Elapsed time = {time_all:.2f}: (collect) {time_data_collection:.2f} + (train) {time_training:.2f}')
    
        if (epoch + 1) % args.snapshot_gap == 0:
            # Visualization
            for (category_type, instance_type) in [('train', 'train'), ('train', 'test'), ('test', 'test')]:
                data = collect_data(
                    conns, model, args.model_type,
                    max_seq_len=args.max_seq_len,
                    position_epsilon=0,
                    direction_epsilon=0,
                    category_type=category_type,
                    instance_type=instance_type
                )

                vis_path = os.path.join(args.visualization_dir, 'epoch_%06d-cat_%s-ins_%s' % (epoch + 1, category_type, instance_type))
                visualization(data, args.num_envs, args.max_seq_len, args.num_visualization, vis_path, f'{epoch + 1}_{args.exp}')

    save_and_exit(None, None)


def get_position_action(affordance_map, epsilon, image, prev_actions):
    """Get position action based on affordance maps. (remove backgrund if rand() < 0.05)

    Returns:
        action: [w, h]s
        score: float
    """
    threshold = 0.1
    for prev_action in prev_actions:
        coord = image[prev_action[0], prev_action[1], :3]
        dist_map = np.sqrt(np.sum((image[:, :, :3] - coord) ** 2, axis=2))
        dist_mask = (dist_map > threshold).astype(np.float)
        affordance_map = affordance_map * dist_mask

    if np.random.rand() < epsilon or np.max(affordance_map) == 0:
        while True:
            idx = np.random.choice(affordance_map.size)
            action = np.array(np.unravel_index(idx, affordance_map.shape))
            z_value = image[action[0], action[1], 2]
            if z_value > 0.005 or np.random.rand() < 0.1:
                break
    else:
        idx = np.argmax(affordance_map)

    action = np.array(np.unravel_index(idx, affordance_map.shape))
    action = action.tolist()
    score = affordance_map[action[0], action[1]]

    return action, score


def get_direction_action(affordance_map, directions, epsilon, action_direction='positive', bad_actions=list()):
    """Get direction action based on affordance maps.

    Returns:
        action: int (index)
        score: float 
    """
    affordance_map_copy = affordance_map.copy()
    affordance_map = affordance_map_copy
    for bad_action in bad_actions:
        dist_map = np.sum((directions - bad_action) ** 2, axis=1)
        cloest_action_id = np.argmin(dist_map)
        affordance_map[cloest_action_id] = 0
    if np.random.rand() < epsilon:
        idx = np.random.choice(affordance_map.size)
    else:
        if action_direction == 'positive':
            idx = np.argmax(affordance_map) if np.max(affordance_map) > 0 else np.random.choice(affordance_map.size)
        elif action_direction == 'negative':
            idx = np.argmax(-affordance_map) if np.max(-affordance_map) > 0 else np.random.choice(affordance_map.size)
        elif action_direction == 'both':
            idx = np.argmax(np.abs(affordance_map))
    action = idx
    score = affordance_map[idx]

    return action, score


def env_process(rank, seed, conn, env_arguments):
    # set random
    np.random.seed(seed)
    
    env = PybulletSim(gui_enabled=False, **env_arguments)
    remain_num = 0
    bad_actions = list()

    while True:
        kwargs = conn.recv()
        if 'message' not in kwargs:
            raise ValueError(f'can not find \'message\'')

        if kwargs['message'] == 'reset':
            if remain_num == 0:
                observation = env.reset(**kwargs)
                scene_state = env.get_scene_state()
                remain_num = 2
                prev_position = list()
            else:
                observation = env.reset(scene_state=scene_state)
                remain_num -= 1
            bad_actions = list()
            conn.send(observation)
        elif kwargs['message'] == 'step-position':
            affordance_map = kwargs['affordance_map']
            action, score = get_position_action(affordance_map, kwargs['epsilon'], kwargs['image'], prev_position)
            prev_position.append(action)
            observation, reward, done, info = env.step([0, action[0], action[1]])
            conn.send((action, score, reward, observation, done, info))
        elif kwargs['message'] == 'step-direction':
            affordance_map = kwargs['affordance_map']
            action_id, score = get_direction_action(affordance_map, kwargs['directions'], kwargs['epsilon'], action_direction=kwargs['action_direction'], bad_actions=bad_actions)
            action = kwargs['directions'][action_id]
            observation, reward, done, info = env.step([1, action[0], action[1], action[2]])
            if reward[1]:
                bad_actions = list()
            else:
                bad_actions.append(action)
            conn.send((action, score, reward, observation, done, info))
        else:
            raise ValueError


def collect_data(conns, model, model_type, max_seq_len, position_epsilon, direction_epsilon, **kwargs):
    num_envs = len(conns)

    model.eval()
    torch.set_grad_enabled(False)

    data = dict()
    kwargs['message'] = 'reset'
    for conn in conns:
        conn.send(kwargs)
    observations = [conn.recv() for conn in conns]
    done_record = [False for _ in range(num_envs)]
    action_direction = ['positive' for _ in range(num_envs)]
    
    # position selection
    step = 0
    position_affordances = model.get_position_affordance(observations)
    for rank in range(num_envs):
        position_affordance = position_affordances[rank]
        conns[rank].send({
            'message': 'step-position',
            'affordance_map': position_affordance,
            'epsilon': position_epsilon,
            'image': observations[rank]['image']
        })
        data[(rank, step)] = observations[rank]
        data[(rank, step)]['affordance_map'] = position_affordance
    
    observations = list()
    for rank in range(num_envs):
        (action, score, (reward, move_flag), observation, done, info) = conns[rank].recv()
        observations.append(observation)
        data[(rank, step)]['type'] = 0
        data[(rank, step)]['action'] = action
        data[(rank, step)]['score'] = score
        data[(rank, step)]['reward'] = reward
        data[(rank, step)]['move_flag'] = move_flag
        data[(rank, step)]['next_image'] = observation['image']
        data[(rank, step)]['image_init'] = observation['image_init']
        data[(rank, step)]['pcd_init'] = observation['pcd_init']
        done_record[rank] = done_record[rank] or done

    # direction selection
    for step in range(1, max_seq_len + 1):
        if np.sum(done_record) == len(done_record):
            break
        direction_affordance_maps, directions = model.get_direction_affordance(observations, model_type)
        for rank in range(num_envs):
            if done_record[rank]:
                continue
            direction_affordance = direction_affordance_maps[rank]
            conns[rank].send({
                'message':'step-direction',
                'affordance_map': direction_affordance,
                'epsilon': direction_epsilon,
                'action_direction': action_direction[rank] if step > 1 else 'both',
                'directions': directions[rank]
            })
            data[(rank, step)] = observations[rank]
            data[(rank, step)]['affordance_map'] = direction_affordance
            data[(rank, step)]['directions'] = directions[rank]

        observations = list()
        for rank in range(num_envs):
            if done_record[rank]:
                observations.append(None)
                continue
            (action, score, (reward, move_flag), observation, (reach_init, reach_boundary), info) = conns[rank].recv()

            observations.append(observation)
            data[(rank, step)]['type'] = 1
            data[(rank, step)]['action'] = action
            data[(rank, step)]['score'] = score
            data[(rank, step)]['reward'] = reward
            data[(rank, step)]['move_flag'] = move_flag
            data[(rank, step)]['reach_init'] = reach_init
            data[(rank, step)]['reach_boundary'] = reach_boundary
            data[(rank, step)]['next_image'] = observation['image']
            data[(rank, step)]['action_direction'] = action_direction[rank]
            for k in info:
                data[(rank, step)][k] = info[k]

            if move_flag:
                data[(rank, 0)]['move_flag'] = True

            # analysis action direction (init)
            if action_direction[rank] == 'negative' and reach_init:
                done_record[rank] = True

            # analysis action direction (boundary)
            if action_direction[rank] == 'positive':
                if step == max_seq_len // 2 or reach_boundary:
                    action_direction[rank] = 'negative'
    return data


def train(model, replay_buffer, pos_optimizer, dir_optimizer, pos_batch_size, dir_batch_size, model_type, device_pos, device_dir, train_model_type):
    type_data = np.array(replay_buffer.scalar_data['type'])
    reward_data = np.array(replay_buffer.scalar_data['reward'])
    move_flag_data = np.array(replay_buffer.scalar_data['move_flag'])

    # add data randomly
    sample_inds = dict()
    if 'pos' in train_model_type:
        sample_inds['position'] = {
            'index': np.argwhere(type_data == 0)[:, 0],
            'positive_index': np.argwhere(np.logical_and(type_data == 0, move_flag_data == True))[:, 0],
            'negative_index': np.argwhere(np.logical_and(type_data == 0, move_flag_data == False))[:, 0],
            'iter': 1,
        }
    if 'dir' in train_model_type:
        sample_inds['direction'] = {
            'index': np.argwhere(type_data == 1)[:, 0],
            'positive_index': np.argwhere(np.logical_and(np.logical_and(type_data == 1, move_flag_data == True), reward_data > 0))[:, 0],
            'negative_index': np.argwhere(np.logical_and(np.logical_and(type_data == 1, move_flag_data == True), reward_data < 0))[:, 0],
            'static_index': np.argwhere(np.logical_and(type_data == 1, move_flag_data == False))[:, 0],
            'iter': 3,
        }
    loss_dict = {'pos': [], 'sgn': [], 'mag': []}
    for sample_type, sample_info in sample_inds.items():
        if len(sample_info['index']) == 0:
            print('[Warning] Data is not balanced')
            continue
        for _ in range(sample_info['iter']):
            if sample_type == 'position':
                replay_iter = list()
                replay_iter.append(np.random.choice(
                    sample_info['positive_index'],
                    min(len(sample_info['positive_index']), pos_batch_size // 2),
                    replace=False
                ))
                replay_iter.append(np.random.choice(
                    sample_info['negative_index'],
                    min(len(sample_info['negative_index']), pos_batch_size // 2),
                    replace=False
                ))
                replay_iter = np.concatenate(replay_iter, 0)
            else:
                replay_iter = list()
                replay_iter.append(np.random.choice(
                    sample_info['positive_index'],
                    min(len(sample_info['positive_index']), dir_batch_size // 3),
                    replace=False
                ))
                replay_iter.append(np.random.choice(
                    sample_info['negative_index'],
                    min(len(sample_info['negative_index']), dir_batch_size // 3),
                    replace=False
                ))
                replay_iter.append(np.random.choice(
                    sample_info['static_index'],
                    min(len(sample_info['static_index']), dir_batch_size // 3),
                    replace=False
                ))
                replay_iter = np.concatenate(replay_iter, 0)

            # fetch data from replay buffer
            observations, scalars = replay_buffer.fetch_data(replay_iter)

            actions = scalars['action']

            model.train()
            torch.set_grad_enabled(True)
            if sample_type == 'position':
                output_tensor = model.get_position_affordance(observations, torch_tensor=True)
                # Compute loss and gradients
                pos_optimizer.zero_grad()
                criterion = nn.CrossEntropyLoss()
                loss = criterion(
                    output_tensor[np.arange(actions.shape[0]), :, actions[:, 0], actions[:, 1]],
                    torch.from_numpy(np.array(scalars['move_flag'], dtype=int)).to(device_pos)
                )
                loss.backward()
                pos_optimizer.step()
                loss_dict['pos'].append(loss.item())
            elif sample_type == 'direction':
                sgn_output, mag_output, _ = model.get_direction_affordance(observations, model_type, torch_tensor=True, directions=actions)
                sgn_target = scalars['move_flag'].astype(int) * np.sign(scalars['reward']).astype(int) + 1
                mag_target = np.abs(scalars['reward']) if model_type == 'sgn_mag' else scalars['reward']
                # Compute loss and gradients
                dir_optimizer.zero_grad()
                loss = 0
                if 'sgn' in model_type:
                    criterion = nn.CrossEntropyLoss()
                    loss_sgn = criterion(
                        sgn_output[:, 0],
                        torch.from_numpy(sgn_target).to(device_dir)
                    )
                    loss += loss_sgn
                    loss_dict['sgn'].append(loss_sgn.item())
                if 'mag' in model_type:
                    criterion = nn.MSELoss()
                    loss_mag = criterion(
                        mag_output[:, 0],
                        torch.from_numpy(mag_target.astype(np.float32)).to(device_dir)
                    ) * 100
                    loss += loss_mag
                    loss_dict['mag'].append(loss_mag.item())

                loss.backward()
                dir_optimizer.step()
        
    loss_info = {}
    for k in loss_dict:
        if len(loss_dict[k]) > 0:
            loss_info[k] = np.mean(loss_dict[k])

    return loss_info


def visualization(vis_data, num_envs, max_seq_len, num_visualization, vis_path, title='visualization'):
    num_visualization = num_envs if num_visualization is None else min(num_visualization, num_envs)
    data = {}
    ids = list()
    cols = ['compare', 'color_image', 'next_image', 'affordance', 'pred', 'info']
    for rank in range(num_visualization):
        for step in range(max_seq_len + 1):
            if (rank, step) in vis_data:
                ids.append(f'{rank}_{step}')

    for (rank, step), sample_data in vis_data.items():
        if rank >= num_visualization:
            continue
        color_image = sample_data['image'][:, :, 3:6]
        next_color_image = sample_data['next_image'][:, :, 3:6]
        data[f'{rank}_{step}_color_image'] = color_image
        data[f'{rank}_{step}_next_image'] = next_color_image
        action = sample_data['action']
        affordance_map = sample_data['affordance_map']

        if sample_data['type'] == 0:
            data[f'{rank}_{step}_pred'] = [f"score: {sample_data['score']:.3f}", f"reward: {sample_data['move_flag']}, {sample_data['reward']:.3f}"]
            data[f'{rank}_{step}_info'] = [f"action: {action}"]
            data[f'{rank}_{step}_compare'] = color_image

            affordance_map -= np.min(affordance_map)
            affordance_map /= np.max(affordance_map)
            cmap = plt.get_cmap('jet')
            affordance_map = cmap(affordance_map)[..., :3]
            data[f'{rank}_{step}_affordance'] = affordance_map * 0.8 + color_image * 0.2
        else:
            data[f'{rank}_{step}_pred'] = [f"action_dir: {sample_data['action_direction']}", f"score: {sample_data['score']:.3f}", f"reward: {sample_data['move_flag']}, {sample_data['reward']:.3f}"]
            data[f'{rank}_{step}_info'] = [f"action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]", f"reach_init: {sample_data['reach_init']}", f"reach_boundary: {sample_data['reach_boundary']}"]
            compare_image = color_image / 2 + next_color_image / 2
            compare_image = utils.draw_action(
                image=compare_image,
                position_start=sample_data['position_start'],
                position_end=sample_data['position_end'],
                cam_intrinsics=sample_data['cam_intrinsics'],
                cam_view_matrix=sample_data['cam_view_matrix']
            )
            data[f'{rank}_{step}_compare'] = compare_image

            affordance_map /= np.max(np.abs(affordance_map))
            affordance_map = (affordance_map + 1) / 2
            cmap = plt.get_cmap('jet')
            affordance_map = cmap(affordance_map)[..., :3]
            affordance_image = color_image.copy()
            affordance_map = (affordance_map * 255).astype(np.uint8).astype(np.float)
            num_direction = len(sample_data['directions'])
            for direction_id in range(num_direction):
                affordance_image = utils.draw_action(
                    image=affordance_image,
                    position_start=sample_data['position_start'],
                    position_end=sample_data['position_start'] + sample_data['directions'][direction_id] * 0.4,
                    cam_intrinsics=sample_data['cam_intrinsics'],
                    cam_view_matrix=sample_data['cam_view_matrix'],
                    thickness=2,
                    tipLength=0.1,
                    color=tuple(affordance_map[direction_id])
                )
            data[f'{rank}_{step}_affordance'] = affordance_image

    utils.html_visualize(vis_path, data, ids, cols, title=title)


if __name__ == '__main__':
    main()