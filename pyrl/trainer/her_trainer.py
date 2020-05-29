# -*- encoding: utf-8 -*-

import enum
import multiprocessing as mp
import os
import random

# OpenAI's Gym
import gym

# ...
import pyrl.util.logging


###############################################################################

_LOG = pyrl.util.logging.get_logger()


###############################################################################

class _Message(enum.Enum):
    RUN = 0
    EXIT = 1
    SYNC = 2


###############################################################################

class HerAgentTrainer(object):

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

        self.env = gym.make(env_name)
        self.agent = None

        self._workers = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def initialize_agent(self, agent_kwargs=None, agent_path="",
                         demo_path=""):
        self.agent = _initialize_agent(self._agent_cls, self._env_name,
                                       agent_kwargs=agent_kwargs,
                                       agent_path=agent_path)
        if demo_path:
            self.agent.load_demonstrations(demo_path)

        for i in range(self._num_envs):
            log_dir = os.path.join(self._root_log_dir, "env{}".format(i))
            self._workers.append(
                HerTrainer(
                    agent_cls=self._agent_cls, env_name=self._env_name,
                    seed=self._seed + i, agent_log_dir=log_dir,
                    agent_kwargs=agent_kwargs, agent_path=agent_path,
                    demo_path=demo_path)
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
            worker.wc_pipe.send((_Message.EXIT, wait))

        if wait:
            for i, worker in enumerate(self._workers):
                _LOG.debug("Waiting worker %d", i)
                while worker.rp_pipe.recv() != _Message.EXIT:
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
        for worker in self._workers:
            worker.wc_pipe.send((_Message.SYNC, self.agent.state_dict()))

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
                    worker.wc_pipe.send(msg)
                    running.append(i)
                if len(running) >= self._num_cpus:
                    break

            # Wait for workers to terminate
            for i in running:
                state = self._workers[i].rp_pipe.recv()
                agent_states.append(state)
                _LOG.debug("Environment %d episodes done", i)
                done[i] = True

        _LOG.debug("Updating master agent")
        self.agent.load_state_dict(agent_states)


###############################################################################

class HerTrainer(mp.Process):

    def __init__(self, agent_cls, env_name, seed,
                 agent_log_dir, agent_kwargs=None, agent_path="",
                 demo_path=""):
        super().__init__()
        self.agent_cls = agent_cls
        self.env_name = env_name
        self.seed = seed
        self.agent_kwargs = agent_kwargs
        self.agent_path = agent_path
        self.agent_log_dir = agent_log_dir
        self.demo_path = demo_path

        self.rc_pipe, self.wc_pipe = mp.Pipe(duplex=False)
        self.rp_pipe, self.wp_pipe = mp.Pipe(duplex=False)

    def run(self):
        agent = _initialize_agent(self.agent_cls, self.env_name,
                                  agent_kwargs=self.agent_kwargs,
                                  agent_path=self.agent_path)
        if self.demo_path:
            _LOG.debug("Loading demonstrations")
            agent.load_demonstrations(self.demo_path)

        agent.env.seed(self.seed)
        agent.set_train_mode()
        agent.init_summary_writter(self.agent_log_dir)

        while True:
            msg, data = self.rc_pipe.recv()

            if msg == _Message.RUN:
                self._run(agent, *data)
            elif msg == _Message.EXIT:
                if data:  # Master is waiting
                    self.wp_pipe.send(_Message.EXIT)
            elif msg == _Message.SYNC:
                agent.load_state_dict(data)
            else:
                raise RuntimeError("unknown message {}".format(msg))

    def _run(self, agent, num_episodes, train_steps):
        for _ in range(num_episodes):
            state = agent.env.reset()
            agent.begin_episode()

            for _ in range(agent.max_episode_steps):

                action = agent.compute_action(state)
                next_state, reward, done, info = agent.env.step(action)
                agent.update(state=state,
                             action=action,
                             reward=reward,
                             next_state=next_state,
                             terminal=info["is_success"])
                state = next_state

                # Episode finished before exhausting the # of steps
                if done:
                    break

            agent.end_episode()

        agent.train(train_steps)
        self.wp_pipe.send(agent.state_dict())


def _initialize_agent(agent_cls, env_name, agent_kwargs=None, agent_path=""):
    if agent_kwargs is None:
        agent_kwargs = {}
    env = gym.make(env_name)

    if agent_kwargs and agent_path:
        raise ValueError("Only agent_kwargs or agent_path can be given"
                         " but not both")
    if agent_kwargs:
        return agent_cls(env, **agent_kwargs)
    if agent_path:
        return agent_cls.load(agent_path, env)

    raise ValueError("Either agent_kwargs or agent_path must be given")
