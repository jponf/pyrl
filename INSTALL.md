Software Components Installation
================================

This section will briefly describe how to install the different components
required by this project. The following steps were performed on a
Windows machine running Windows subsystem for linux (WSL) v1 + Ubuntu 18.04, and, even though this means that they *should* work seamlesly on
an equivalent set up, there are no guarantees of their reproducibility.

> If you use another Linux flavor (natively or via WSL) consider using
  Ubuntu 18.04 since this distribution is supported by both, the ROS
  and the OpenManipulator teams.


## Basic software

The following list contains some software that is recommended to install,
not all are necessery for the installation of the following packages but
may be necessary and/or convenient for other operations.

* Tools and Libs to build packages for your distribution
  * Debian/Ubuntu: `apt install build-essential`
* Python setuptools + pip
  * Debian/Ubuntu: `apt install python-pip python-setuptools`

## Installing Gazebo

Installing Gazebo is quite simple, we start by adding the
software repository and its *apt* key.

```bash
$ sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
$ wget http://packages.osrfoundation.org/gazebo.key -O - \
     | sudo apt-key add -
```

Then, we update the software catalog and install Gazebo.

```bash
sudo apt update
sudo apt install gazebo9
```

> **NOTE:** We are installing Gazebo 9 since at the time of
>           writing this guide ROS Melodic did not come with
>           the packages to interact with Gazebo 10

Once the installation has finished, just run `gazebo` in the
terminal and the Gazebo loading window should pop up (unless
you are running it from Windows using WSL).

### Additional steps for WSL

Gazebo is composed of two components `gzserver` + `gzclient`, and whereas
the server should run fine in a pure terminal environment, the client is a
GUI-based program and thereby it will not run in an environment without
support for graphical applications, such as WSL. We can address this problem
by installing an X server in Windows and setting the appropriate environment
variable into the WSL.

There are different solutions to install a X server in Windows,
but the one we have tried is VcXsrv:

* https://sourceforge.net/projects/vcxsrv/

Once installed we can run the XLaunch program (from the windows start menu)
with the following options:

* Multiple windows
* Display number   (if available): -1
* Start no client
* Clipboard (checked)
* Primary Selection (checked)
* Native opengl **(uncheked)**
* Disable access control **(uncheked)**

Finally, with the X server running (an icon should appear in the system's
tray area), we just need to set up the *DISPLAY* environment variable
inside WSL: `export DISPLAY=:0` and then
run `gazebo`.

> **NOTE:** to avoid having to set the *DISPLAY* variable every
> time we can add the `export DISPLAY=:0` command to our
> `~/.bashrc` file.

> **NOTE 2:** Your WSL distribution may come with the file
*/etc/profile.d/wsl-integration.sh*. This file automatically sets the
DISPLAY variable if it detects an X server, but it also set the
LIBGL_ALWAYS_REDIRECT environment variable, which will crash Gazebo client.
You must `unset` the variable before running Gazebo or, preferably, modify
the */etc/profile.d/wsl-integration.sh* so that it does not create the
variable in the first place.

## Installing ROS

First of all we will add the software repository and its *apt* key.

```bash
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
                   --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

If the `apt-key` command fails we can use curl to retrieve the key manually:

```bash
$ curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
```

Now we simply update the software list of our distribution
and install the desired ROS distribution package. For the sake
of simplicity here we will install a meta-package that install many complements of the Melodic ROS distribution, but for real
deployments it is recommended to install only the necessary
packages.

```bash
$ sudo apt update
$ sudo apt install ros-melodic-desktop-full
```

After installing the base ROS package we can continue installing
the packages to connect ROS with Gazebo and OpenManipulator.

```bash
$ sudo apt install ros-melodic-industrial-core \
                   ros-melodic-gazebo-ros-pkgs \
                   ros-melodic-moveit          \
                   ros-melodic-ros-control     \
                   ros-melodic-effort-controllers
```

Finally, we can init and update the ROS dependencies system:

```bash
$ sudo rosdep init
$ rosdep update
```

and make sure the environment variables are properly set
everytime we log into our computer:

```bash
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
```

# Installing OpenMANIPULATOR ROS Packages

Once ROS has been properly installed, and the environment variables properly
set, we can install the packages to interact with the OpenMANIPULATOR robot.
To do so, we will create a catkin workspace, clone a bunch of repositories and
finally run catkin to build the cloned projects.

```bash
$ mkdir -p robotis_ws/ros/src
$ cd robotis_ws/ros/src
$ git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
$ git clone https://github.com/ROBOTIS-GIT/dynamixel-workbench.git
$ git clone https://github.com/ROBOTIS-GIT/dynamixel-workbench-msgs.git
$ git clone https://github.com/ROBOTIS-GIT/open_manipulator.git
$ git clone https://github.com/ROBOTIS-GIT/open_manipulator_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/open_manipulator_simulations.git
$ git clone https://github.com/ROBOTIS-GIT/robotis_manipulator.git
$ cd ..
$ catkin_make install
```

After catkin has finished building all the packages run the *setup*
script to initialize the environment variables. To avoid having to
run the setup step every time you can add the call to the *setup*
script into your user *.bashrc* or *.bash_profile* files.

```bash
$ echo "source `realpath install/setup.bash`" >> ~/.bashrc
$ source ~/.bashrc
```

# OpenAI ROS

Now we have all the components to use the Open Manipulator robot in
Gazebo, but we cannot interact with it from Python. We could write
all the code to do so from scratch or we can use the project OpenAI-ROS,
which provides some building blocks to connect Gazebo simulations
and ROS with our Python code. This project uses catkin and thereby
the installation steps are similar to the previous ones.

```bash
$ mkdir -p openai_ros/ros/src
$ cd openai_ros/ros/src
$ git clone https://bitbucket.org/theconstructcore/openai_ros.git
$ cd ..
$ catkin_make install
```

Finally, just as before, we add the *setup* script in our *.bashrc*
or *.bash_profile* files

```bash
$ echo "source `realpath install/setup.bash`" >> ~/.bashrc
$ source ~/.bashrc
```

<!--
## Georgia Tech kinematic utils

**THIS PART IS DEPRECATED AND WILL BE REMOVED**

Kinematics and Geometry utilities for KDL

```bash
sudo apt install ros-melodic-urdf-parser-plugin \
                 ros-melodic-urdfdom-py
git clone https://github.com/gt-ros-pkg/hrl-kdl.git
cd hrl-kdl/hrl-geom
pip install -U .
cd ../pykdl_utils
pip install -U .
```
-->
<!-- # Installing Intel Realsense D435

> This section has not been tested yet!

To this end, we are going to use the packages provided by Intel for Ubuntu
16.04/18.04. Hence the first step is adding Intel's repository.

https://github.com/intel/gazebo-realsense

```bash
$ sudo sh -c 'echo "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo $(lsb_release -sc) main" > /etc/apt/sources.list.d/intel-latest.list'
$ curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x6F3EFCDE' | sudo apt-key add -
$ sudo apt update
```

With Intel's repository configured we can now update the software list and
install Intel's **realsense** libraries and some additional ROS packages.

```bash
$ sudo apt install librealsense2-dev \
           librealsense2-utils       \
           ros-melodic-rgbd-launch   \
           ros-melodic-ddynamic-reconfigure
```

Finally, as with OpenMANIPULATOR, we can install Intel's realsense ROS
packages.

```bash
$ mkdir -p intel_ws/src
$ cd intel_ws/src
$ git clone git clone https://github.com/intel-ros/realsense.git
$ cd ..
$ catkin_make install
```

And run the *setup* script to initialize the environment variables.

```bash
$ echo "source `realpath install/setup.bash`" >> ~/.bashrc
$ source ~/.bashrc
``` -->