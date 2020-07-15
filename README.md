# Python Reinforcement Learning Agents

Different reinforcement learning agents implemented in Python. The current
version uses Pytorch ![Pytorch Logo][pytorch-16px], but more frameworks may be supported in the future.


## Requirements 📋

The main requirement is a Python 3 interpreter, preferably 3.5 or newer but
it should work just as fine with previous versions. If you do not have a
valid interpreter you are probably running Windows or MacOS, for these systems
we recommend using [Conda][miniconda].

Additionally, to avoid polluting your Python's system-wide environment we
recommend creating a virtual environment, for example:

```
python3 -m venv ~/.virtualenvs/rl
```

or, if you use conda:

```
conda create --name rl
```

Finally, with the Python environment set up, we just need to install the
required packages, which are all avaliable through PyPi, run

```
pip install -r requirements.txt
```

to install them all.


## Implemented Algorithms

The current version implements the following list of reinforcement learning
algorithms.

 * Discrete Action Spaces
   * **TODO**
 * Continuous Action Spaces
   * [Deep Deterministic Policy Gradient (DDPG)][ddpg]
   * [Twin Delayed Deep Deterministic Policy Gradient (TD3)][td3]
   * [Soft Actor-Critic (SAC)][sac]

Additionally, some of them have also been implemented using the [Hindsight
Experience Replay][her] technique, thereby they can be used in environments
with sparse rewards. These are: *DDPG*, *TD3* and *SAC*.


## Using the agents in Gym environments ![gym-32px]

The packages can be run as a Python script using the command `python -m pyrl`,
by running the package you can choose among different entry points to train/test
the different agents on OpenAI gym environments. For example, to train the
*DDPG* agent on the *Bipedal Walker* environment use the command:

```bash
python3 -m pyrl ddpg-train BipedalWalker-v3
```

The command above will run a generic *DDPG* training routine on the *Bipedal
Walker* environment, and save the agent in a subdirectory named *ddpg*. But
there are more options, to get the list of all the available entry points and
some package wide options use the *-h/--help* flag:

```bash
python3 -m pyrl --help
```

Moreover, there are several options specific to to each train/test entry points
that can be used to configure some aspects of the train/test routines. To get
these options use the *-h/--help* flag after the entry point, for example:

```bash
python -m pyrl ddpg-train --help
python -m pyrl ddpg-test --help
```

### Example DDPG on Bipedal Walker

![DDPG training on Bipedal Walker GIF](images/gif/ddpg-bipedalwalker-train.gif)

**TODO:** Add bipedal walker test GIF.

## Package structure ![technologist-32px]

This package does not have a formal documentation yet, methods are just
documented within the code. In the meantime, if instead of using the package
entry points you would rather write your own routines or integrate the agents
in your project, here we provide you with some insights to help you navigate
the package and use the agents.

First, we would like to show you the different subpackages and explain what
can be found inside them, then we will explain the Agent interface, which
defines the methods that all Agents must implement and should be enough to
train and use them.

 * *pyrl.agents*: This package contains the agents interface, their
    implementation and some utility modules that implement some components
    used by more than one agent, for example *pyrl.agents.noise* contains
    different action and parameter noise strategies, *pyrl.agents.replay_buffer*
    contains replay buffers designed to be memory efficient, etc.

 * *pyrl.cli*: Inside this package you will find the "command line interface",
    a.k.a the entry points of the generic train/test routines, that work on
    OpenAI gym environments. The code in this module may help you understand
    how to train an agent and later on use it however you want.

 * *pyrl.trainer*: The trainer is an utility that runs multiple copies of the
    same agent on the same environment with different seeds. The key point is
    that after each training step the "master" agent aggregates the results of
    all the copies and then synchronizes the result.

 * *pyrl.util*: Contains routines and classes that can be useful in various
    places.

### Agent Interface

**TODO**


## More Examples

 * SAC on *FetchReach* Environment
   ![SAC on FetchReach GIF](images/gif/her-sac-fetch-reach.gif)
 * HER+TD3 on *FetchPickAndPlace* Environment
   **TOOD**
 * HER+SAC on *FetchPickAndPlace* Environment
   **TODO**

<!-- ***** References ***** -->
[ddpg]: https://arxiv.org/abs/1509.02971 "arXiv: Continuous control with deep reinforcement learning"
[td3]: https://arxiv.org/abs/1802.09477 "arXiv: Addressing Function Approximation Error in Actor-Critic Methods"
[sac]: https://arxiv.org/abs/1801.01290 "arXiv: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
[her]: https://arxiv.org/abs/1707.01495 "arXiv: Hindisght Experience Replay"

[miniconda]: https://docs.conda.io/en/latest/miniconda.html "Free minimal installer for conda"

<!-- ***** Images ***** -->
[gym-16px]: images/gym-16.png "OpenAI Gym Logo 16px"
[gym-32px]: images/gym-16.png "OpenAI Gym Logo 32px"

[pytorch-32px]: images/pytorch-32.png "Pytorch Logo 32x32"
[pytorch-16px]: images/pytorch-16.png "Pytorch Logo 16x16"

[technologist-16px]: images/technologist-16.png "Technologist Emoji 16x16"
[technologist-32px]: images/technologist-32.png "Technologist Emoji 32x32"