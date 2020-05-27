# -*- coding: utf-8 -*-

# Torch Stack
import torch.utils.tensorboard as tensorboard

# ...
import torchrl.util.ugym


###############################################################################

class Agent(object):
    """Generic Agent interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._summary_w = None

    def init_summary_writter(self, log_path):
        """Initializes a summary writter to log the agent evolution."""
        if self._summary_w is not None:
            raise ValueError("summary writter can only be initialized once")
        self._summary_w = tensorboard.SummaryWriter(log_dir=log_path)

    def set_train_mode(self, mode=True):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, terminal):
        raise NotImplementedError

    def train(self, steps, progress=False):
        raise NotImplementedError

    def compute_action(self, state):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def save(self, path, replay_buffer=True):
        raise NotImplementedError

    def reset(self):
        pass

    def end_episode(self):
        pass

    @classmethod
    def load(cls, path, *args, **kwargs):
        raise NotImplementedError

    def set_eval_mode(self):
        """Sets the agent in evaluation mode."""
        self.set_train_mode(mode=False)

    def log_scalar(self, tag, value, step):
        """TO DOCUMENT"""
        if self._summary_w is not None:
            self._summary_w.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """TO DOCUMENT"""
        if self._summary_w is not None:
            self._summary_w.add_scalars(main_tag, tag_scalar_dict, step)


class HerAgent(Agent):
    """Generic Hindsight Experience Replay Agent interface."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not torchrl.util.ugym.is_her_env(env):
            raise ValueError("{} is not a valid HER environment".format(env))

        self.env = env

    @property
    def max_episode_steps(self):
        return self.env.spec.max_episode_steps

    @classmethod
    def load(cls, env, path, *args, **kwargs):
        raise NotImplementedError