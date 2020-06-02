# -*- encoding: utf-8 -*-

import enum
import multiprocessing as mp
import os
import random

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

class _Message(enum.Enum):
    RUN = 0
    EXIT = 1
    SYNC = 2


###############################################################################

class AgentTrainer(object):

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

        self._workers = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=False)

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

        for i in range(self._num_envs):
            log_dir = os.path.join(self._root_log_dir, "env{}".format(i))
            self._workers.append(
                Trainer(
                    agent_cls=self._agent_cls, env_name=self._env_name,
                    seed=self._seed + i, num_threads=num_threads,
                    agent_log_dir=log_dir, agent_kwargs=agent_kwargs,
                    agent_path=agent_path, demo_path=demo_path)
            )

    def start(self):
        if not self._workers:
            raise RuntimeError("agent not initialized")

        for i, worker in enumerate(self._workers):
            _LOG.debug("Starting worker %d", i)
            worker.start()

    def shutdown(self, wait=True):
        """Signals workers to terminate."""
        for worker in self._workers:
            worker.parent_pipe.send((_Message.EXIT, wait))

        if wait:
            for i, worker in enumerate(self._workers):
                _LOG.debug("Waiting worker %d", i)
                while worker.parent_pipe.recv() != _Message.EXIT:
                    pass

    def run(self, num_episodes, train_steps):
        if not self._workers:
            raise RuntimeError("there are no workers")
        if not all(x.is_alive() for x in self._workers):
            raise RuntimeError("some workers are not running, did you call"
                               " start?")

        self._synchronize_agents()
        self._run(num_episodes, train_steps)

    def _synchronize_agents(self):
        """Synchronizes the workers with the master updates."""
        _LOG.debug("Synchronizing agents")
        for worker in self._workers:
            worker.parent_pipe.send((_Message.SYNC, self.agent.state_dict()))

    def _run(self, num_episodes, train_steps):
        agent_states = []
        done = [False] * len(self._workers)

        while not all(done):
            # Execute pending workers (up to num_cpus)
            running = []
            for i, worker in enumerate(self._workers):
                if not done[i]:
                    _LOG.debug("Running environment %d on worker", i)
                    msg = (_Message.RUN, (num_episodes, train_steps))
                    worker.parent_pipe.send(msg)
                    running.append(i)
                if len(running) >= self._num_cpus:
                    break

            # Wait for workers to terminate
            for i in running:
                state = self._workers[i].parent_pipe.recv()
                agent_states.append(state)
                _LOG.debug("Environment %d episodes done", i)
                done[i] = True

        _LOG.debug("Updating master agent")
        if len(agent_states) > 1:
            self.agent.aggregate_state_dicts(agent_states)
        else:
            self.agent.load_state_dict(agent_states[0])


###############################################################################

class Trainer(mp.Process):

    def __init__(self, agent_cls, env_name, seed, num_threads,
                 agent_log_dir, agent_kwargs=None, agent_path="",
                 demo_path=""):
        super().__init__()
        self.agent_cls = agent_cls
        self.env_name = env_name
        self.seed = seed
        self.num_threads = num_threads

        self.agent_kwargs = agent_kwargs
        self.agent_path = agent_path
        self.agent_log_dir = agent_log_dir
        self.demo_path = demo_path

        # self.rc_pipe, self.wc_pipe = mp.Pipe(duplex=False)
        # self.rp_pipe, self.wp_pipe = mp.Pipe(duplex=False)
        self.parent_pipe, self.child_pipe = mp.Pipe(duplex=True)

    def run(self):
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)

        agent, env = _initialize_agent(self.agent_cls, self.env_name,
                                       agent_kwargs=self.agent_kwargs,
                                       agent_path=self.agent_path,
                                       demo_path=self.demo_path)

        env.seed(self.seed)
        agent.set_train_mode()
        agent.init_summary_writter(self.agent_log_dir)
        self._run_loop(agent, env)

    def _run_loop(self, agent, env):
        try:
            while True:
                msg, data = self.child_pipe.recv()

                if msg == _Message.RUN:
                    self._run(env, agent, *data)
                elif msg == _Message.EXIT:
                    if data:  # Master is waiting
                        self.child_pipe.send(_Message.EXIT)
                    break
                elif msg == _Message.SYNC:
                    agent.load_state_dict(data)
                else:
                    raise RuntimeError("unknown message {}".format(msg))
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

        if train_steps == 0:
            train_steps = total_steps

        agent.train(train_steps)
        self.child_pipe.send(agent.state_dict())


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
