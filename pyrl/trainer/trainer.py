# -*- encoding: utf-8 -*-

"""Simple parallel agent trainer."""


import enum
import importlib
import multiprocessing
import os
import random
import sys
import traceback

# OpenAI's Gym
import gym

# PyTorch
import torch

# ...
import pyrl.agents.core
import pyrl.util.logging
import pyrl.util.ugym


###############################################################################

if (sys.version_info.major, sys.version_info.minor) < (3, 4):
    raise ImportError("This module requires Python >= 3.4")


_LOG = pyrl.util.logging.get_logger()

mp = multiprocessing.get_context("spawn")  # pylint: disable=invalid-name


###############################################################################

class _Message(enum.Enum):
    RUN = 0
    EXIT = 1
    SYNC = 2


###############################################################################

class AgentTrainer(object):
    """Trains an agent and coordinates multiple agents training in paralel
    to exploit information learned by all of them.
    """

    def __init__(self, agent_cls, env_name, num_envs,
                 root_log_dir, num_cpus=1, seed=None):
        if num_cpus < 1:
            num_cpus = mp.cpu_count()
        if seed is None:
            seed = random.randint(0, 0x7fffffff)

        self._agent_cls = agent_cls
        self._env_name = env_name
        self._num_envs = num_envs
        self._num_cpus = min(num_envs, num_cpus)
        self._root_log_dir = root_log_dir
        self._seed = seed

        self.env = _initialize_env(self._env_name, self._agent_cls)
        self.agent = None

        self._trainers = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def initialize_agent(self, agent_kwargs=None, agent_path="",
                         demo_path=""):
        """Initializes the agent."""
        self.agent = _initialize_agent(self._agent_cls, self.env,
                                       agent_kwargs=agent_kwargs,
                                       agent_path=agent_path,
                                       demo_path=demo_path)

        num_threads = 0  # let the libraries decide
        if self._num_cpus > 1:
            num_threads = max(1, int(self._num_cpus / self._num_envs))
            _LOG.debug("Configuring workers with %d threads", num_threads)

        env_module = self.env.spec.entry_point.split(":")[0]
        for i in range(self._num_envs):
            log_dir = os.path.join(self._root_log_dir, "env{}".format(i))
            self._trainers.append(
                Trainer(
                    trainer_id=i, agent_cls=self._agent_cls,
                    env_name=self._env_name, env_module=env_module,
                    seed=self._seed + i, num_threads=num_threads,
                    agent_log_dir=log_dir, agent_kwargs=agent_kwargs,
                    agent_path=agent_path, demo_path=demo_path)
            )

    def start(self):
        """Starts trainer processes to run the training in parallel."""
        if not self._trainers:
            raise RuntimeError("agent not initialized")

        for i, worker in enumerate(self._trainers):
            _LOG.debug("Starting trainer %d", i)
            worker.start()

    def shutdown(self):
        """Signals trainers to terminate."""
        _LOG.debug("Shutting down")
        for i, trainer in enumerate(self._trainers):
            _LOG.debug("Sending EXIT to trainer %d", i)
            trainer.parent_pipe.send((_Message.EXIT, None))

        for i, trainer in enumerate(self._trainers):
            _LOG.debug("Joining trainer %d", i)
            trainer.join()

    def run(self, num_episodes, train_steps):
        """Lets the agents explore for `num_episodes` before performing
        `train_steps` training steps.

        :param num_episodes: Number of episodes to explore before training.
        :param train_steps: Number of training steps to perform after
            exploring. The special value 0 is translated to the same number
            of steps taken during exploration.
        """
        if not self._trainers:
            raise RuntimeError("there are no workers")
        if not all(x.is_alive() for x in self._trainers):
            raise RuntimeError("some workers are not running, did you call"
                               " start?")

        self._synchronize_agents()
        self._run(num_episodes, train_steps)

    def _synchronize_agents(self):
        """Synchronizes the workers with the master updates."""
        _LOG.debug("Synchronizing agents")
        self._call_msg(_Message.SYNC, self.agent.state_dict())
        _LOG.debug("Agents synchronized")

    def _run(self, num_episodes, train_steps):
        agent_states = []
        done = [False] * len(self._trainers)

        while not all(done):
            # Execute pending workers (up to num_cpus)
            to_run = []

            for i, trainer in enumerate(self._trainers):
                if not done[i]:
                    to_run.append(trainer)
                if len(to_run) >= self._num_cpus:
                    break

            _LOG.debug("Running trainers: %s",
                       ", ".join(str(x.trainer_id) for x in to_run))
            states = self._call_msg(_Message.RUN, (num_episodes, train_steps),
                                    to_run)
            agent_states.extend(states)

            for trainer in to_run:
                done[trainer.trainer_id] = True

        _LOG.debug("Updating master agent")
        if len(agent_states) > 1:
            self.agent.aggregate_state_dicts(agent_states)
        else:
            self.agent.load_state_dict(agent_states[0])

    def _call_msg(self, msg, data, trainers=None):
        if trainers is None:
            trainers = self._trainers

        results = []
        for trainer in trainers:
            trainer.parent_pipe.send((msg, data))

        for trainer in trainers:
            ret, err = trainer.parent_pipe.recv()
            if err:
                exc, trace = err
                print(trace, file=sys.stderr)
                raise exc
            results.append(ret)

        return results


###############################################################################

class Trainer(mp.Process):

    def __init__(self, trainer_id, agent_cls, env_name, env_module, seed,
                 num_threads, agent_log_dir, agent_kwargs=None, agent_path="",
                 demo_path=""):
        super().__init__()
        self.trainer_id = trainer_id
        self.agent_cls = agent_cls
        self.env_name = env_name
        self.env_module = env_module
        self.seed = seed
        self.num_threads = num_threads

        self.agent_kwargs = agent_kwargs
        self.agent_path = agent_path
        self.agent_log_dir = agent_log_dir
        self.demo_path = demo_path

        self.parent_pipe, self.child_pipe = mp.Pipe(duplex=True)

    def run(self):
        # Configure worker
        importlib.import_module(self.env_module)
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        # Initialize environment and agent
        env = _initialize_env(self.env_name, self.agent_cls)
        agent = _initialize_agent(self.agent_cls, env,
                                  agent_kwargs=self.agent_kwargs,
                                  agent_path=self.agent_path,
                                  demo_path=self.demo_path)

        env.seed(self.seed)
        agent.set_train_mode()
        agent.init_summary_writter(self.agent_log_dir)

        # Run loop
        self._run_loop(agent, env)

        # Clean up
        env.close()
        del env
        del agent

    def _run_loop(self, agent, env):
        try:
            while True:
                msg, data = self.child_pipe.recv()
                ret = None
                error = None
                try:
                    if msg == _Message.RUN:
                        ret = self._run(env, agent, *data)
                    elif msg == _Message.SYNC:
                        agent.load_state_dict(data)
                    elif msg == _Message.EXIT:
                        break
                    else:
                        raise RuntimeError("unknown message {}".format(msg))
                except Exception as err:  # pylint: disable=broad-except
                    trace = traceback.format_exc()
                    error = (err, trace)

                self.child_pipe.send((ret, error))
        except KeyboardInterrupt:
            pass

    def _run(self, env, agent, num_episodes, train_steps):
        total_steps = 0

        for _ in range(num_episodes):
            state = env.reset()
            agent.begin_episode()

            for _ in range(env.spec.max_episode_steps):
                action = agent.compute_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state=state,
                             action=action,
                             reward=reward,
                             next_state=next_state,
                             terminal=done or info.get("is_success", False))
                state = next_state

                total_steps += 1

                # Episode finished before exhausting the # of steps
                if done:
                    break

            agent.end_episode()

        if train_steps == 0:  # One training step for each time step
            train_steps = total_steps

        agent.train(train_steps)
        return agent.state_dict()


###############################################################################

def _initialize_env(env_or_name, agent_cls):
    if isinstance(env_or_name, gym.Env):
        env = env_or_name
    else:
        env = gym.make(env_or_name)

    is_her_agent = issubclass(agent_cls, pyrl.agents.core.HerAgent)
    is_her_env = pyrl.util.ugym.is_her_env(env)

    if is_her_env and not is_her_agent:
        env = pyrl.util.ugym.flatten_her_env(env)

    return env


def _initialize_agent(agent_cls, env,
                      agent_kwargs=None, agent_path="",
                      demo_path=""):
    if agent_kwargs is None:
        agent_kwargs = {}

    if agent_kwargs and agent_path:
        raise ValueError("Only agent_kwargs or agent_path can be given"
                         " but not both")

    agent = None
    if issubclass(agent_cls, pyrl.agents.core.Agent):
        if agent_kwargs:
            agent = agent_cls(env.observation_space, env.action_space,
                              **agent_kwargs)
        elif agent_path:
            agent = agent_cls.load(agent_path)
    elif issubclass(agent_cls, pyrl.agents.core.HerAgent):
        if agent_kwargs:
            agent = agent_cls(env, **agent_kwargs)
        elif agent_path:
            agent = agent_cls.load(agent_path, env)

        if demo_path:
            agent.load_demonstrations(demo_path)
    else:
        raise TypeError("agent_cls must be a subclass of Agent or HerAgent")

    if agent is None:
        raise ValueError("Either agent_kwargs or agent_path must be given")

    return agent
