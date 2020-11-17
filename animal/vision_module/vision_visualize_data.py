import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from animal.vision_module.vision_dataset import DatasetVision
from animal.vision_module.vision_functions import plot_sample


def log_plots(log_dir, train_data, test_data, samples=100):

    if True:
        data = train_data
    else:
        data = test_data

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    dataset = DatasetVision(data)

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": 1,
        "drop_last": True
    }

    dataloader = DataLoader(dataset, **dataloader_parameters)

    t = tqdm.tqdm(dataloader)
    for idx, data in enumerate(t):

            if idx == samples:
                break

            obs, pos, rot, rot2 = data

            obs = obs[0, :, :, :].permute(1, 2, 0).numpy()
            pos = pos.numpy().squeeze()
            rot = rot.numpy().squeeze()
            rot2 = rot2.numpy().squeeze()

            fig = plot_sample(obs, pos, rot, rot2)

            writer.add_figure(
                'plots/{}'.format(idx), fig, idx)


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
        train_data=args.data_dir + "/train_position_data.npz",
        test_data=args.data_dir + "/test_position_data.npz",
    )
