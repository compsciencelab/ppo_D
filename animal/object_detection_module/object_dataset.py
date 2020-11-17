import torch
import numpy as np
from torch.utils.data import Dataset


class DatasetObjects(Dataset):
    """
    Creates a dataset to train a object detection module without recurrence.
    """

    def __init__(
            self,
            data_filename,
    ):

        data = np.load(data_filename)

        self.observations = data['observations']
        self.labels = data['labels']
        self.frames_per_episode = data['frames_per_episode']

        self.num_samples = self.observations.shape[0]

    def __len__(self):
        return self.num_samples

    def get_frames_per_episode(self):
        return self.frames_per_episode

    def __getitem__(self, idx):
        obs = self.observations[idx, :, :, :]
        labels = self.labels[idx, :]
        return torch.FloatTensor(obs), torch.LongTensor(labels)


class DatasetObjectRecurrent(Dataset):
    """
    Creates a dataset to train a object detection module with recurrence.
    """

    def __init__(
            self,
            data_filename,
    ):

        data = np.load(data_filename)

        self.observations = data['observations']
        self.labels = data['labels']
        self.frames_per_episode = data['frames_per_episode']
        self.num_samples = (self.observations.shape[0] // self.frames_per_episode) - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = self.observations[
              self.frames_per_episode * idx:self.frames_per_episode * idx +
              self.frames_per_episode, :, :, :]
        labels = self.labels[
              self.frames_per_episode * idx:self.frames_per_episode * idx +
              self.frames_per_episode, :]
        return torch.FloatTensor(obs), torch.LongTensor(labels)
