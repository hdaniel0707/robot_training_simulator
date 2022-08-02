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

4. Install RLBench as in its README file: `backend/RLBench/README.md` [link](backend/RLBench/README.md)
