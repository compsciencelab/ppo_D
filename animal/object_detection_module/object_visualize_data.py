import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from object_dataset import DatasetObjects

import numpy as np

labels_to_object = {
     0: 'GoodGoal',
     1: 'BadGoal',
     2: 'GoodGoalMulti',
     3: 'Wall',
     4: 'Ramp',
     5: 'CylinderTunnel',
     6: 'WallTransparent',
     7: 'CylinderTunnelTransparent',
     8: 'Cardbox1',
     9: 'Cardbox2',
    10: 'UObject',
    11: 'LObject',
    12: 'LObject2',
    13: 'DeathZone',
    14: 'HotZone',
    15: 'lol'
}


def log_plots(log_dir, train_data, test_data, samples=100):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    dataset_train = DatasetObjects(train_data)

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": 1,
        "drop_last": True
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_parameters)

    t = tqdm.tqdm(dataloader_train)
    for idx, data in enumerate(t):

            if idx == samples:
                break

            obs, label = data

            obs = obs[0, :, :, :].permute(1, 2, 0).numpy()
            label = label[0].numpy()

            fig = plt.figure()
            ax1 = plt.axes()
            plt.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                labelbottom=False,
                labelleft=False)
            ax1.imshow(obs / 255.)

            writer.add_figure(
                'plots/{}_{}'.format(labels_to_object[label[0]], idx), fig, idx)


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--data-dir', help='Data directory')
    parser.add_argument('--log-dir', help='Target log directory')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    args.log_dir += "/plots/"

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    log_plots(
        log_dir=args.log_dir,
        train_data=args.data_dir + "/train_object_data.npz",
        test_data=args.data_dir + "/test_object_data.npz",
    )

