import torch
import numpy as np
from datetime import datetime
from ppo.envs import make_vec_envs
from animal.vision_module.vision_functions import make_animal_env
from ppo.model import (Policy, CNNBase, FixupCNNBase, ImpalaCNNBase,
                       StateCNNBase)

CNN = {'CNN': CNNBase, 'Impala': ImpalaCNNBase, 'Fixup': FixupCNNBase,
       'State': StateCNNBase}


def collect_data(target_dir, args, arenas, params, num_samples=1000, frames_episode=20):

    maker = make_animal_env(
        arenas, params, inference_mode=args.realtime,
        frame_skip=args.frame_skip, reduced_actions=args.reduced_actions,
        state=args.state, arenas_dir=args.arenas_dir)

    env = make_vec_envs(
        make=maker, num_processes=args.num_processes, device=device, log_dir=None,
        num_frame_stack=args.frame_stack, state_shape=None, num_state_stack=0)

    actor_critic = Policy(
        env.observation_space.shape, env.action_space,
        base=CNN[args.cnn],
        base_kwargs={'recurrent': args.recurrent_policy})

    if args.load_model:
        actor_critic.load_state_dict(
            torch.load(args.load_model, map_location=device))
    actor_critic.to(device)
    recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(1, 1).to(device)

    # Final storage
    obs_rollouts = np.zeros([num_samples, 3, 84, 84], dtype=np.uint8)
    pos_rollouts = np.zeros([num_samples, 3], dtype=np.float32)
    rot_rollouts = np.zeros([num_samples, 3], dtype=np.float32)
    norm_vel_rollouts = np.zeros([num_samples, 3], dtype=np.float32)

    # Temporary storage
    episode_obs = np.zeros(
        [args.num_processes, frames_episode, 3, 84, 84], dtype=np.uint8)
    episode_pos = np.zeros([args.num_processes, frames_episode, 3], dtype=np.float32)
    episode_rot = np.zeros([args.num_processes, frames_episode, 3], dtype=np.float32)
    episode_norm_vel = np.zeros([num_samples, frames_episode, 3], dtype=np.float32)

    global_step = 0
    steps = [0 for _ in range(args.num_processes)]
    obs = env.reset()
    print()
    start = datetime.now()

    while global_step < (num_samples // frames_episode):

        end = datetime.now()
        delta = end - start
        print('collected {}/{} data points in {} h {} mins and {} secs.'.format(
            global_step * frames_episode, num_samples, delta.seconds // 3600,
            ((delta.seconds // 60) % 60), delta.seconds % 60), end='\r')

        with torch.no_grad():
            _, actions, _, _, _ = actor_critic.act(
                obs, recurrent_hidden_states, masks,
                deterministic=args.det)

        # first wait for things to fall down
        for num_process, step in enumerate(steps):
            if step < 10:
                actions[num_process] = 0

        # Observation reward and next obs
        obs, _, dones, infos = env.step(actions)

        # after 10 steps start saving obs
        for num_process, step in enumerate(steps):
            if step >= 10:
                episode_obs[num_process, step - 10, :, :, :
                ] = obs[num_process].cpu().numpy()[-3:, :, :].astype(np.uint8)
                episode_pos[num_process, step - 10, :] = infos[num_process]['agent_position']
                episode_rot[num_process, step - 10, :] = infos[num_process]['agent_rotation']
                episode_norm_vel[num_process, step - 10, :] = infos[num_process]['agent_norm_vel']

        # if done and max step not reached -> ignore data
        for num_process, done in enumerate(dones):
            if done and steps[num_process] < (frames_episode - 1 + 10):
                steps[num_process] = 0

        # if max step reached -> copy to final storage
        for num_process, step in enumerate(steps):
            if step == (frames_episode - 1 + 10):
                idx = global_step * frames_episode
                obs_rollouts[idx:idx + frames_episode, :, :, :] = episode_obs[num_process, :, :, :, :].astype(np.uint8)
                pos_rollouts[idx:idx + frames_episode] = episode_pos[num_process, :, :].astype(np.float32)
                rot_rollouts[idx:idx + frames_episode] = episode_rot[num_process, :, :].astype(np.float32)
                norm_vel_rollouts[idx:idx + frames_episode] = episode_norm_vel[num_process, :, :].astype(np.float32)

                steps[num_process] = 0
                global_step += 1

                if global_step >= (num_samples // frames_episode):
                    break

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in dones]).to(device)

        for i in range(len(steps)):
            steps[i] += 1

    np.savez(target_dir,
             observations=np.array(obs_rollouts).astype(np.uint8),
             positions=pos_rollouts.astype(np.float32),
             rotations=rot_rollouts.astype(np.float32),
             norm_vel=norm_vel_rollouts.astype(np.float32),
             frames_per_episode=frames_episode)


if __name__ == '__main__':

    import random
    import argparse
    import numpy as np
    from animal.arenas.utils import (
        create_c1_arena,
        create_c1_arena_weird,
        create_c2_arena,
        create_c3_arena,
        create_c3_arena_basic,
        create_c4_arena,
        create_c5_arena,
        create_c6_arena,
        create_c6_arena_basic,
        create_c7_arena,
        create_maze,
        create_arena_choice,

        # skills
        create_arena_cross,
        create_arena_push1,
        create_arena_push2,
        create_arena_tunnel1,
        create_arena_tunnel2,
        create_arena_ramp1,
        create_arena_ramp2,
        create_arena_ramp3,
        create_arena_narrow_spaces_1,
        create_arena_narrow_spaces_2,
        create_arena_pref1,
        create_blackout_test_1
    )

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--target-dir',
        help='Directory to save data.')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Cuda device  or cpu (default:cuda:0 )')
    parser.add_argument(
        '--non-det', action='store_true', default=True,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action')
    parser.add_argument(
        '--frame-stack', type=int, default=4, help='Number of frame to stack')
    parser.add_argument(
        '--load-model', default='',
        help='directory to save agent logs (default: )')
    parser.add_argument(
        '--reduced-actions', action='store_true', default=False,
        help='Use reduced actions set')
    parser.add_argument(
        '--cnn', default='Fixup',
        help='Type of cnn. Options are CNN,Impala,Fixup,State')
    parser.add_argument(
        '--arenas-dir', default=None,
        help='If specific arenas used to collect data')
    parser.add_argument(
        '--state-stack', type=int, default=4,
        help='Number of steps to stack in states')
    parser.add_argument(
        '--realtime', action='store_true', default=False,
        help='If to plot in realtime. ')
    parser.add_argument(
        '--num-processes',type=int, default=35,
        help='how many training CPU processes to use (default: 16)')

    args = parser.parse_args()
    args.det = not args.non_det
    args.state = args.cnn == 'State'
    device = torch.device(args.device)

    list_arenas = [
        create_c1_arena,
        create_c2_arena,
        create_c3_arena,
        create_c3_arena_basic,
        create_c4_arena,
        # create_c5_arena,
        create_c6_arena,
        create_c6_arena_basic,
        # create_maze,
        # create_arena_choice,

        create_arena_cross,
        create_arena_push1,
        create_arena_push2,
        create_arena_tunnel1,
        create_arena_tunnel2,
        create_arena_ramp1,
        create_arena_ramp2,
        create_arena_ramp3,
        create_arena_narrow_spaces_1,
        create_arena_narrow_spaces_2,
    ]

    list_params = [
        {'is_train': False, 'time': 1000, 'max_reward': float( np.random.randint(5, 10))},             # create_c1_arena
        {'is_train': False, 'time': 1000, 'max_reward': float(np.random.randint(5, 10))},              # create_c2_arena
        {'is_train': False, 'time': 1000},                                                             # create_c3_arena
        {'is_train': False, 'time': 1000, 'num_walls': np.random.randint(5, 15)},                      # create_c3_arena_basic
        {'is_train': False, 'time': 1000, 'num_red_zones': 8, 'max_orange_zones': 3},                  # create_c4_arena
        # {'is_train': False, 'time': 1000},                                                             # create_c5_arena
        {'is_train': False, 'time': 1000},                                                             # create_c6_arena
        {'is_train': False, 'time': 1000, 'num_walls': np.random.randint(5, 15)},                      # create_c6_arena_basic
        # {'is_train': False, 'time': 1000, 'obj': random.choice(['CylinderTunnel','door', 'Cardbox1'])},  # create_maze
        # {'is_train': False, 'time': 1000},                                                               # create_arena_choice

        {'is_train': False, 'time': 1000},                                                               # create_arena_cross
        {'is_train': False, 'time': 1000},                                                               # create_arena_push1
        {'is_train': False, 'time': 1000},                                                               # create_arena_push2
        {'is_train': False, 'time': 1000},                                                               # create_arena_tunnel1
        {'is_train': False, 'time': 1000},                                                               # create_arena_tunnel2
        {'is_train': False, 'time': 1000},                                                               # create_arena_ramp1
        {'is_train': False, 'time': 1000},                                                               # create_arena_ramp2
        {'is_train': False, 'time': 1000},                                                               # create_arena_ramp3
        {'is_train': False, 'time': 1000},                                                               # create_arena_narrow_spaces_1
        {'is_train': False, 'time': 1000},                                                               # create_arena_narrow_spaces_2
    ]

    assert len(list_arenas) == len(list_params)

    collect_data(
        args.target_dir + 'train_position_data',
        args, list_arenas, list_params, num_samples=500000,
    )

    collect_data(
        args.target_dir + 'test_position_data',
        args, list_arenas, list_params, num_samples=100000,
    )
