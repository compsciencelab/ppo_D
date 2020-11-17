import tqdm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from object_model import ImpalaCNNObject
from object_dataset import DatasetObjects, DatasetObjectRecurrent
from object_functions import Loss, compute_error


def object_module_train(model, epochs, log_dir, train_data, test_data, device, batch_size=4):

    # Define logger
    writer = SummaryWriter(log_dir, flush_secs=5)

    # Define loss
    criterion = Loss()

    # Get data
    if model.is_recurrent:
        batch_size = 4
        dataset_train = DatasetObjectRecurrent(train_data)
        dataset_test = DatasetObjectRecurrent(test_data)
    else:
        batch_size = 128
        dataset_train = DatasetObjects(train_data)
        dataset_test = DatasetObjects(test_data)

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
        epoch_error = 0

        t = tqdm.tqdm(dataloader_train)
        for idx, data in enumerate(t):

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            if idx != 0:
                t.set_postfix(
                    train_loss=avg_loss,
                    train_avg_error=avg_error,
                )

            images, label = data

            images = images.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred_label, _, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = criterion.compute(label, pred_label)
            error = compute_error(label, pred_label)

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

        model.eval()
        epoch_loss = 0
        epoch_error = 0
        t = tqdm.tqdm(dataloader_test)
        for idx, data in enumerate(t):
            if idx != 0:
                t.set_postfix(
                    test_loss=avg_loss,
                    test_avg_error=avg_error,
                )

            recurrent_hidden_states = torch.zeros(
                1, batch_size, model.recurrent_hidden_state_size).to(device)

            images, label = data

            images = images.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred_label, _, hx, _ = model(
                inputs=images,
                rnn_hxs=recurrent_hidden_states)

            loss = criterion.compute(label, pred_label)
            error = compute_error(label, pred_label)

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (idx + 1)
            epoch_error += error.item()
            avg_error = epoch_error / (idx + 1)

        if avg_loss < lowest_test_loss:
            lowest_test_loss = avg_loss
            model.save(
                "{}/model_{}.ckpt".format(log_dir, epoch), net_parameters)

        writer.add_scalar('test/loss', avg_loss, epoch)
        writer.add_scalar('test/error', avg_error, epoch)


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
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    net_parameters = {
        'num_inputs': 3,
        'recurrent': args.recurrent,
        'hidden_size': 256,
        'image_size': 84
    }

    model = ImpalaCNNObject(**net_parameters)

    object_module_train(
        model, 5000, args.log_dir,
        train_data=args.data_dir + "/train_object_data.npz",
        test_data=args.data_dir + "/test_object_data.npz",
        device=args.device,
    )

