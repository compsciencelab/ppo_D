import math
import json
import shutil
import tarfile
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from .settings import num_classes


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size, batch_first=True)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def save(self, filename, net_parameters):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, device=torch.device('cpu')):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict( torch.load(path + "/state.torch", map_location=device) )
        return net, net_parameters


class ImpalaCNNObject(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, image_size=84):
        super(ImpalaCNNObject, self).__init__(recurrent, hidden_size, hidden_size)

        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.image_size = image_size
        self.main = ImpalaCNN(image_size,num_inputs,hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(hidden_size, num_classes))

    def forward(self, inputs, rnn_hxs=None):
        x = self.main(inputs / 255.0)
        if self.is_recurrent:
            x, rnn_hxs = self.gru(x, rnn_hxs)

        return self.linear(x), F.sigmoid(self.linear(x)), rnn_hxs, x


class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.
    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self, image_size, depth_in, hidden_size):
        super().__init__()
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.BatchNorm2d(depth_out),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, hidden_size))

    def forward(self, x):

        in_shape = x.shape
        if len(in_shape) == 5:
            batch_size = x.shape[0]
            sequence_len = x.shape[1]
            channels = x.shape[2]
            height = x.shape[3]
            width = x.shape[4]
            x = x.view(batch_size*sequence_len, channels, height, width)

        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)

        if len(in_shape) == 5:
            x = x.view(batch_size, sequence_len, -1)

        return x


class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super().__init__()
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        self.conv1 = init_(nn.Conv2d(depth, depth, 3, padding=1))
        self.conv2 = init_(nn.Conv2d(depth, depth, 3, padding=1))
        self.norm1 = nn.BatchNorm2d(depth)
        self.norm2 = nn.BatchNorm2d(depth)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + x

