import os
import gym
import uuid
import torch
import random
import animalai
import torch.nn as nn
from ppo.envs import TransposeImage
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import AnimalAIEnv
from animal.animal import RetroEnv, FrameSkipEnv
from animal.wrappers import RetroEnv, Stateful, FilterActionEnv
from animal.object_detection_module.object_arenas import create_object_arena, num_classes


def compute_error(label, logits):
    """ Compute batch error as number of objects wrongly classified. """

    label = label.view(-1, 1)
    logits = logits.view(-1, num_classes)
    prediction = torch.argmax(logits, dim=1)
    error = torch.sum((label.squeeze(1) != prediction), dim=0).double() / label.shape[0]

    return error


class Loss:
    """ Multi-target binary loss. """

    def __init__(self):

        self.loss = nn.CrossEntropyLoss()

    def compute(self, label, prediction):

        label = label.view(-1, 1)
        prediction = prediction.view(-1, num_classes)
        loss = self.loss(prediction, label.squeeze(1))

        return loss


def make_animal_env(inference_mode, frame_skip, reduced_actions, state):

    base_port = random.randint(0, 100)
    def make_env(rank):
        def _thunk():

            if 'DISPLAY' not in os.environ.keys():
                os.environ['DISPLAY'] = ':0'
            exe = os.path.join(os.path.dirname(animalai.__file__),'../../env/AnimalAI')
            env = AnimalAIEnv(
                environment_filename=exe,
                retro=False, worker_id=base_port + rank,
                docker_training=False,
                seed=0, n_arenas=1, arenas_configurations=None,
                greyscale=False, inference=inference_mode,
                resolution=None)

            env = RetroEnv(env)
            env = LabAnimalCollect(env)

            if reduced_actions:
                env = FilterActionEnv(env)

            if state:
                env = Stateful(env)

            if frame_skip > 0:
                env = FrameSkipEnv(env, skip=frame_skip)
                print("Frame skip: ", frame_skip, flush=True)

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    return make_env


class LabAnimalCollect(gym.Wrapper):
    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        self._arena_file = ''
        self._object = None
        self._env_steps = None
        self._label = None

    def step(self, action):
        action = int(action)
        obs, reward, done, info = self.env.step(action)
        self._env_steps += 1
        info['arena'] = self._arena_file
        info['label'] = self._label

        return obs, reward, done, info

    def reset(self, **kwargs):
        # Create new arena
        name = str(uuid.uuid4())
        object, label = create_object_arena("/tmp/", name)
        self._arena_file, arena = ("/tmo/{}.yaml".format(name), ArenaConfig(
            "/tmp/{}.yaml".format(name)))
        os.remove("/tmp/{}.yaml".format(name))

        self._object = object
        self._env_steps = 0
        self._label = label

        return self.env.reset(arenas_configurations=arena, **kwargs)
