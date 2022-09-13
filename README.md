# Robot Training Simulator

## System

* Ubuntu 18.04

## Install

1. Clone the repo:
```
git clone --recurse-submodules https://github.com/hdaniel0707/robot_training_simulator.git
```
Or for inner use:
```
git clone --recurse-submodules git@github.com:hdaniel0707/robot_training_simulator.git
```
If you have already cloned without the `--recurse-submodules` tag, then run:
```
git submodule update --init --recursive
```
Not recommended: For updating the submodules to the up-to-date commits run:
```
git submodule update --remote --merge
```

2. Install (download and extract) CoppeliaSim (V-REP) 4.1.0 version: https://www.coppeliarobotics.com/previousVersions

3. Install PyRep as in its README file: `backend/PyRep/README.md` [link](backend/PyRep/README.md)

  PyRep requires version **4.1** of CoppeliaSim. Download:
  - [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
  - [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
  - [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)


  Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

  ```bash
  export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  ```

  __Remember to source your bashrc (`source ~/.bashrc`) or
  zshrc (`source ~/.zshrc`) after this.

  Finally install the python library:

  ```bash
  cd backend/PyRep
  pip3 install -r requirements.txt
  pip3 install -e .
  ```

  _Although you can use CoppeliaSim on any platform, communication via PyRep is currently only supported on Linux._


4. Install RLBench as in its README file: `backend/RLBench/README.md` [link](backend/RLBench/README.md)

  Hopefully you have now installed PyRep and have run one of the PyRep examples.
  Now lets install RLBench:

  I used pip3 instead of pip:
  ```bash
  pip3 install -r requirements.txt
  pip3 install -e .
  ```
