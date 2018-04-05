# Installation

A short guide to install this package is below. The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.

*Use absolute paths everywhere.*

## Linux

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and obtain the mujoco class license key from canvas (talk to instructors to obtain this if you don't have canvas access).
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `~/.bashrc` by adding the following paths and source it. *Please use the correct username and use absolute paths like in the below examples.*
```
export LD_LIBRARY_PATH="/home/aravind/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using
```
$ conda update conda
$ cd <path/to/drl_hw1>
$ conda env create -f setup/linux.yml
```
- Add the directory to pythonpath. Change `~/.bashrc` and append:
```
export PYTHONPATH="</path/to/drl_hw1>:$PYTHONPATH"
```
- Source the bash so that all paths are correctly configured
```
$ source ~/.bashrc
$ source activate hw1-env
```

## Mac OS

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and obtain the mujoco class license key from canvas (talk to instructors to obtain this if you don't have canvas access).
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Update `~/.bash_profile` by adding the following paths and source it. *Please use the correct username and use absolute paths like in the below examples.*
```
export LD_LIBRARY_PATH="Users/aravind/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/aravind/.mujoco/mjpro150/bin
```
- Install this package using
```
$ conda update conda
$ cd path/to/drl_hw1
$ conda env create -f setup/mac.yml
```
- Add the directory to pythonpath. Change `~/.bash_profile` and append (*again, please use the full correct path*):
```
export PYTHONPATH="</path/to/drl_hw1>:$PYTHONPATH"
```
- Source the bash so that all paths are correctly configured
```
$ source ~/.bash_profile
$ source activate hw1-env
```

## Windows

We recommend using Windows Subsystem for Linux: https://docs.microsoft.com/en-us/windows/wsl/install-win10

This gives you a Linux environment within Windows without the overhead of a Virtual Machine. In addition, you will need a X-Windows Server. https://sourceforge.net/projects/vcxsrv/ VcXsrv is one option if you don't already have one; if you use this you'll have to configure it to not use it's native OpenGL.

You can then start up the Bash prompt through WSL (Start menu -> Ubuntu), and follow the linux instructions above.

When you are testing, you will have to set the DISPLAY env variable:

```
$ export DISPLAY=:0.0
```

This variable can also be set in your .bashrc if you'd like.

## Known Issues

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.
```
$ MJPL python script.py
```

- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```
