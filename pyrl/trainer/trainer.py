# -*- encoding: utf-8 -*-

import importlib
import multiprocessing as mp
import os
import random
import tempfile

import dask.distributed

# OpenAI's Gym
import gym

# PyTorch
import torch

# ...
import pyrl.agents.core
import pyrl.util.logging
import pyrl.util.ugym


###############################################################################

_LOG = pyrl.util.logging.get_logger()


###############################################################################

class AgentTrainer(object):

    def __init__(self, agent_cls, env_name, num_envs,
                 root_log_dir, num_cpus=1, seed=None):
        super(AgentTrainer, self).__init__()

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

        self._workers = []

        tempdir = tempfile.mkdtemp(prefix="pyrl-dask-")
        with dask.config.set({"distributed.worker.daemon": True,
                              "temporary-directory": tempdir}):
            self.cluster = dask.distributed.LocalCluster(n_workers=num_cpus)
            self.client = dask.distributed.Client(self.cluster)

    def initialize_agent(self, agent_kwargs=None, agent_path="",
                         demo_path=""):
        self.agent, _ = _initialize_agent(self._agent_cls, self.env,
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
            future = self.client.submit(Trainer,
                                        # Submit as actor
                                        actor=True,
                                        workers=[i % self._num_cpus],
                                        # Trainer kwargs
                                        agent_cls=self._agent_cls,
                                        env_name=self._env_name,
                                        env_module=env_module,
                                        seed=self._seed + i,
                                        num_threads=num_threads,
                                        agent_log_dir=log_dir,
                                        agent_kwargs=agent_kwargs,
                                        agent_path=agent_path,
                                        demo_path=demo_path)
            self._workers.append(future.result())

    def run(self, num_episodes, train_steps):
        if not self._workers:
            raise RuntimeError("there are no workers")
        self.client.cancel(self._workers[0])
        _LOG.info("Synchronizing agents")
        sync_futures = [x.synchronize_agent(self.agent.state_dict())
                        for x in self._workers]
        for x in sync_futures:
            x.result()

        _LOG.info("Running agents")
        run_futures = [x.run_episodes(num_episodes, train_steps)
                       for x in self._workers]
        state_dicts = [x.result() for x in run_futures]

        self.agent.aggregate_state_dicts(state_dicts)


###############################################################################

class Trainer(object):

    def __init__(self, agent_cls, env_name, env_module, seed, num_threads,
                 agent_log_dir, agent_kwargs=None, agent_path="",
                 demo_path=""):
        super(Trainer, self).__init__()
        importlib.import_module(env_module)

        self.agent_cls = agent_cls
        self.env_name = env_name
        self.seed = seed
        self.num_threads = num_threads

        self.agent_kwargs = agent_kwargs
        self.agent_path = agent_path
        self.agent_log_dir = agent_log_dir
        self.demo_path = demo_path

        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        self.agent, self.env = _initialize_agent(
            self.agent_cls, self.env_name,
            agent_kwargs=self.agent_kwargs,
            agent_path=self.agent_path,
            demo_path=self.demo_path)

        self.env.seed(self.seed)
        self.agent.set_train_mode()
        self.agent.init_summary_writter(self.agent_log_dir)

    def synchronize_agent(self, agent_state):
        self.agent.load_state_dict(agent_state)

    def run_episodes(self, num_episodes, train_steps):
        total_steps = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            self.agent.begin_episode()

            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.compute_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminal=done or info.get("is_success", False))
                state = next_state

                total_steps += 1
                # Episode finished before exhausting the # of steps
                if done:
                    break

            self.agent.end_episode()

        if train_steps == 0:
            train_steps = total_steps

        self.agent.train(train_steps)
        return self.agent.state_dict()


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


def _initialize_agent(agent_cls, env_or_name,
                      agent_kwargs=None, agent_path="",
                      demo_path=""):
    if agent_kwargs is None:
        agent_kwargs = {}

    # Initialize environment
    env = _initialize_env(env_or_name, agent_cls)
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
    return agent, env
