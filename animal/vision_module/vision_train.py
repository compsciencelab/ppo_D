import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from vision_model import ImpalaCNNVision
from vision_dataset import DatasetVision, DatasetVisionRecurrent
from vision_functions import loss_func, plot_prediction


def vision_train(model, epochs, log_dir, train_data, test_data, device, batch_size=128):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    # Get data
    if model.is_recurrent:
        dataset_train = DatasetVisionRecurrent(train_data)
        dataset_test = DatasetVisionRecurrent(test_data)
    else:
        dataset_train = DatasetVision(
            data_filename=train_data,
            multiple_data_path="/shared/albert/train_object_data*")
        dataset_test = DatasetVision(
            data_filename=test_data,
            multiple_data_path="/test_object_data*")

    # Define dataloader
    dataloader_parameters = {
        "num_workers": 0,
        "shuffle": True,
        "pin_memory": True,
        "batch_size": batch_size,
        "drop_last": True
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_parameters)
    dataloader_test = DataLoader(dataset_test, **dataloader_parameters)

    device = torch.device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    model.to(device)
    lowest_test_loss = float('inf')

    for epoch in range(epochs):

        print('Epoch {}'.format(epoch))
        model.train()
        epoch_loss = 0
        epoch_pos_error = 0
        epoch_rot_error = 0
        t = tqdm.tqdm(dataloader_train)
        for idx, data in enumerate(t):

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            if idx != 0:
                t.set_postfix(
                    train_loss=avg_loss,
                    train_position_error=avg_pos_error,
                    train_rotation_error=avg_rot_error,
                )

            images, pos, rot, _ = data

            images = images.to(device)
            pos = pos.view(-1, 3).to(device)
            rot = rot.view(-1, 3).to(device)

            optimizer.zero_grad()
            pred_position, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss, pos_error, rot_error = loss_func(
                pos, rot, pred_position[:, 0:3], pred_position[:, 3:6])

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (idx + 1)
            epoch_pos_error += pos_error.item()
            avg_pos_error = epoch_pos_error / (idx + 1)
            epoch_rot_error += rot_error.item()
            avg_rot_error = epoch_rot_error / (idx + 1)

            loss.backward()
            optimizer.step()

        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            images = images.view(
                -1, model.num_inputs, model.image_size, model.image_size)
            figure = plot_prediction(
                images, pos, rot, pred_position[:, 0:3], pred_position[:, 3:6])
            writer.add_figure(
                'train_figure_epoch_{}'.format(epoch), figure, epoch)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/position_error', avg_pos_error, epoch)
        writer.add_scalar('train/rotation_error', avg_rot_error, epoch)

        model.eval()
        epoch_loss = 0
        epoch_pos_error = 0
        epoch_rot_error = 0
        t = tqdm.tqdm(dataloader_test)
        for idx, data in enumerate(t):
            if idx != 0:
                t.set_postfix(
                    test_loss=avg_loss,
                    test_position_error=avg_pos_error,
                    test_rotation_error=avg_rot_error,
                )

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            images, pos, rot, _ = data

            images = images.to(device)
            pos = pos.view(-1, 3).to(device)
            rot = rot.view(-1, 3).to(device)

            pred_position, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss, pos_error, rot_error = loss_func(
                pos, rot, pred_position[:, 0:3], pred_position[:, 3:6])

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (idx + 1)
            epoch_pos_error += pos_error.item()
            avg_pos_error = epoch_pos_error / (idx + 1)
            epoch_rot_error += rot_error.item()
            avg_rot_error = epoch_rot_error / (idx + 1)

        if epoch % 10 == 0:
            images = images.view(
                -1, model.num_inputs, model.image_size, model.image_size)
            figure = plot_prediction(
                images, pos, rot, pred_position[:, 0:3], pred_position[:, 3:6])
            writer.add_figure(
                'test_figure_epoch_{}'.format(epoch), figure, epoch)

        if avg_loss < lowest_test_loss:
            lowest_test_loss = avg_loss
            model.save(
                "{}/model_{}.ckpt".format(log_dir, epoch), net_parameters)

        writer.add_scalar('test/loss', avg_loss, epoch)
        writer.add_scalar('test/position_error', avg_pos_error, epoch)
        writer.add_scalar('test/rotation_error', avg_rot_error, epoch)


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--data-dir', help='Data directory')
    parser.add_argument('--log-dir', help='Target log directory')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Cuda device  or cpu (default:cuda:0 )')
    parser.add_argument(
        '--recurrent', action='store_true',
        default=False, help='use RNN model')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    net_parameters = {
        'num_inputs': 3,
        'recurrent': args.recurrent,
        'hidden_size': 256,
        'image_size': 84
    }

    model = ImpalaCNNVision(**net_parameters)

    vision_train(
        model, 5000, args.log_dir,
        train_data=args.data_dir + "/train_position_data.npz",
        test_data=args.data_dir + "/test_position_data.npz",
        device=args.device,
    )
