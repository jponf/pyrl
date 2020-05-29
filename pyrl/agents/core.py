# -*- coding: utf-8 -*-

import abc
import six

# ...
import pyrl.util.ugym
import pyrl.util.summary


###############################################################################

@six.add_metaclass(abc.ABCMeta)
class Agent(object):
    """Generic Agent interface."""

    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self._summary = pyrl.util.summary.DummySummary()

    def init_summary_writter(self, log_path):
        """Initializes a tensorboard summary writter to track the agent
        evolution while trainig."""
        if self._summary is not None:
            raise ValueError("summary writter can only be initialized once")
        self._summary = pyrl.util.summary.Summary(log_dir=log_path)

    def set_eval_mode(self):
        """Sets the agent in evaluation mode."""
        self.set_train_mode(mode=False)

    @abc.abstractmethod
    def set_train_mode(self, mode=True):
        """Sets the agent training mode."""
        raise NotImplementedError

    @abc.abstractmethod
    def begin_episode(self):
        """Prepares the agent to run a new training episode.

        Some agents have to prepare to register a new training
        episode, for example by emptying buffers, reseting noise, etc.
        """

    @abc.abstractmethod
    def end_episode(self):
        """Indicates the agent that the episode that started in a previous
        call to `begin_episode` has finished.

        When `end_episode` is called the agent can use all the experience
        gathered on calls to `update` to compute metrics and move data
        from temporary to persisten buffers.
        """

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, terminal):
        """Registers the transition and updates any agent internal information
        useful for training.

        :param state: State from which the agent decided to take `action`.
        :param action: Action taken to move from `state` to `next_state`.
        :param reward: Reward received for taking `action` from `state`.
        :param next_state: State reached after taking `action`.
        :param terminal: Whether or not `next_state` is a terminal state.
        """

    @abc.abstractmethod
    def compute_action(self, state):
        """Computes the next action to take given the current `state` of
        the environment.

        This function may behave differently depending on the agent
        mode, evaluation or training, for example by adding noise to
        explore unknown states.
        """
        raise NotImplementedError

    def train(self, steps, progress=False):
        """Trains an agent for the specified number of `steps`.

        :param steps: The number of steps to train the agent for.
        :param progress: If set the training progress is printed on the
            standard output stream (using tqdm).
        """
        raise NotImplementedError

    def state_dict(self):
        """Returns a dictionary containing the whole state of the agent.

        The content depends on the type of agent and may include neural
        nets, running averages, etc.

        :return: A dictionary containing the whole state of the agent.
        :rtype: dict
        """
        raise NotImplementedError

    def load_state_dict(self, state):
        """Copies the state into this agent. Any additional key in the
        dictionary is ignored.

        Unless you know what you are doing you should only pass dictionaries
        returned by `state_dict()`.

        :param state: A dict containing a valid agent state.

        :raise KeyError: If a required key is not in the dictionary.
        """
        raise NotImplementedError

    def save(self, path, replay_buffer=True):
        """Saves the agent in the given `path`.

        Different agents may save their state using different formats
        but the preferred way is using `path` as a root directory that
        will contain all the agent components.

        :param replay_buffer: Whether the replay buffer should be saved
            or not.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path, *args, **kwargs):
        raise NotImplementedError


class HerAgent(Agent):
    """Generic Hindsight Experience Replay Agent interface."""

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not pyrl.util.ugym.is_her_env(env):
            raise ValueError("{} is not a valid HER environment".format(env))

        self.env = env

    @property
    def max_episode_steps(self):
        """The `max_episode_steps` attribute from the environment's spec."""
        return self.env.spec.max_episode_steps

    @classmethod
    def load(cls, env, path, *args, **kwargs):
        raise NotImplementedError
