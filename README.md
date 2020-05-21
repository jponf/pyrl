Robotic Arm RL
==============

**Reinforcement Learning applied to a robotic arm.**

Before changing/running the content of this project, please make
sure that your environment is properly configured by performing
the steps mentioned in [Environment Setup](#env_setup).
Additional information can be found in the
[References](#references) section.


# Environment Setup <a id="env_setup"></a>

First of all make sure all the necessary software packages are
installed, the instructions to install them can be found in the
accompaning [INSTALL](INSTALL.md) file.

After installing the software, apply the necessary fixes from
[FIXES](FIXES.md).

# Project Structure

The project is developed using Python 2.7, which is the version supported
by ROS. Nevertheless, since Python 2.7 support terminates in 2020 the
project uses `__future__` imports and the `builtins` module to make the
code as close as possible to modern Python 3.

```
.
├── references
|   └── *.[txt|pdf]
├── robotrl
|   ├── envs
|   ├── resources
|   └── util
├── INSTALL.md
├── README.md
└── requirements.txt
```

The `references` directory contains information that can help you understand
some pieces of the project, such as the algorithms or the structure. `robotrl`
is the main Python package that contains the code developed in this project,
it is subdivided in `envs` (conains OpenAI environments), `resources`
(used to store non source files) and `util` (utility code).

# Gym Environments

The following list contains the names of the OpenAI gym environments defined
in this project and how to `make` them using gym's API. You can find the code
that registers them in [robotrl \_\_init\_\_ file](./robotrl/__init__.py).

 * OpenManipulatorReacherGazebo-v0
   * ```python
     gym.make("robotrl:OpenManipulatorReacherGazebo-v0")
     ```


# Running the simulation



## Referenes <a id="references"></a>

 * [Install Gazebo-ROS Packages](http://gazebosim.org/tutorials?tut=ros_installing&cat=connect_ros#Installgazebo_ros_pkgs)
 * [Gazebo Tutorials](http://gazebosim.org/tutorials)

### Papers

 * [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
 * [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
 * [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
 * [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
 * [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
 * [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
 * [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
 * [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)

 *The list order is random and does not reflect the importance of the papers*