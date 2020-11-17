import tqdm
import torch
import numpy as np
from tensorboardX import SummaryWriter
from ppo.envs import make_vec_envs
from object_functions import Loss, compute_error
from animal.object_detection_module.object_functions import make_animal_env
from ppo.model import (Policy, CNNBase, FixupCNNBase, ImpalaCNNBase,
                       StateCNNBase)


CNN = {'CNN': CNNBase, 'Impala': ImpalaCNNBase, 'Fixup': FixupCNNBase,
       'State': StateCNNBase}


def get_batch(env, actor_critic, device_agent, batch_size, sequence_size):

    recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size).to(device_agent)
    masks = torch.zeros(1, 1).to(device_agent)

    images = np.zeros([batch_size, sequence_size, 3, 84, 84])
    labels = np.zeros([batch_size, sequence_size, 1])

    for episode_num in range(batch_size):

        obs = env.reset()
        step = 0
        episode_obs = np.zeros([sequence_size, 3, 84, 84])
        episode_labels = np.zeros([sequence_size, 1])

        while step < sequence_size + 10:

            with torch.no_grad():
                _, action, _, _, _ = actor_critic.act(
                    obs, recurrent_hidden_states, masks,
                    deterministic=args.det)

            if step < 10:  # wait for things to fall down
                action = torch.zeros_like(action)

            # Observation reward and next obs
            obs, reward, done, info = env.step(action)

            if done:
                step = 0

            if step >= 10:
                episode_obs[step - 10, :, :, :] = obs[0].cpu().numpy()[0:3, :, :]
                episode_labels[step - 10, :] = info[0]['label']

            masks.fill_(0.0 if done else 1.0)

            step += 1

    images[episode_num, :, :, :, :] = episode_obs
    labels[episode_num, :, :] = episode_labels

    return images, labels


def object_module_train(args, model, batch_size=4, sequence_size=50):

    epoch = 1
    device_agent = torch.device(args.device_agent)
    device_module = torch.device(args.device_module)

    # DATA STUFF

    maker = make_animal_env(
        inference_mode=args.realtime,
        frame_skip=args.frame_skip, reduced_actions=args.reduced_actions,
        state=args.state)

    env = make_vec_envs(
        make=maker, num_processes=1, device=device_agent, log_dir=None,
        num_frame_stack=args.frame_stack, state_shape=None, num_state_stack=0)

    actor_critic = Policy(
        env.observation_space.shape, env.action_space,
        base=CNN[args.cnn],
        base_kwargs={'recurrent': args.recurrent_policy})

    if args.load_model:
        actor_critic.load_state_dict(
            torch.load(args.load_model, map_location=device_agent))
    actor_critic.to(device_agent)

    # TRAIN STUFF

    # Define logger
    writer = SummaryWriter(args.log_dir, flush_secs=5)

    # Define loss
    criterion = Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=10)
    model.to(device_module)
    lowest_test_loss = float('inf')

    while True:

        print('Epoch {}'.format(epoch))
        epoch += 1
        epoch_loss = 0
        epoch_error = 0
        model.train()
        t = tqdm.tqdm(range(1000))

        for idx in t:

            if idx != 0:
                t.set_postfix(
                    train_loss=avg_loss,
                    train_avg_error=avg_error,
                )

            images, labels = get_batch(
                env, actor_critic, device_agent, batch_size, sequence_size)

            images = torch.FloatTensor(images).to(device_module)
            labels = torch.LongTensor(labels).to(device_module)
            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(
                device_module)

            optimizer.zero_grad()
            pred_label, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = criterion.compute(labels, pred_label)
            error = compute_error(labels, pred_label)

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (idx + 1)
            epoch_error += error.item()
            avg_error = epoch_error / (idx + 1)

            loss.backward()
            optimizer.step()

        scheduler.step(avg_loss)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/error', avg_error, epoch)

        epoch_loss = 0
        epoch_error = 0
        model.eval()
        t = tqdm.tqdm(range(200))
        for idx in t:

            if idx != 0:
                t.set_postfix(
                    test_loss=avg_loss,
                    test_avg_error=avg_error,
                )

            images, labels = get_batch(
                env, actor_critic, device_agent, batch_size, sequence_size)

            images = torch.FloatTensor(images).to(device_module)
            labels = torch.LongTensor(labels).to(device_module)
            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(
                device_module)

            optimizer.zero_grad()
            pred_label, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = criterion.compute(labels, pred_label)
            error = compute_error(labels, pred_label)

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (idx + 1)
            epoch_error += error.item()
            avg_error = epoch_error / (idx + 1)

        if avg_loss < lowest_test_loss:
            lowest_test_loss = avg_loss
            model.save(
                "{}/model_{}.ckpt".format(args.log_dir, epoch), net_parameters)

        writer.add_scalar('test/loss', avg_loss, epoch)
        writer.add_scalar('test/error', avg_error, epoch)


if __name__ == "__main__":

    import argparse
    from object_model import ImpalaCNNObject

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--log-dir', help='Target log directory')
    parser.add_argument(
        '--device-agent', default='cuda:0',
        help='Cuda device  or cpu (default:cuda:0)')
    parser.add_argument(
        '--device-module', default='cuda:1',
        help='Cuda device  or cpu (default:cuda:1)')
    parser.add_argument(
        '--non-det', action='store_true', default=True,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--recurrent-module', action='store_true', default=False,
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
        '--state-stack', type=int, default=4,
        help='Number of steps to stack in states')
    parser.add_argument(
        '--realtime', action='store_true', default=False,
        help='If to plot in realtime. ')

    args = parser.parse_args()
    args.det = not args.non_det
    args.state = args.cnn == 'State'

    net_parameters = {
        'num_inputs': 3,
        'recurrent': args.recurrent_module,
        'hidden_size': 256,
        'image_size': 84
    }

    object_model = ImpalaCNNObject(**net_parameters)

    object_module_train(
        args,
        object_model,
        batch_size=4,
        sequence_size=50,
    )

