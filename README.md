# T-ASE-2024-4227
## Version Requirements

- create a anaconda environment via: `conda create -n BCI-DRL python=3.8`
- activate the virtual env via: `conda activate BCI-DRL`
- install the requirements via: `requirements.txt`

## `BCI`

### `Tarining `

In the folder `BCI`, the script `ECCA.py` is used for offline training. Due to project limitations, only one subject dataset is provided in the `datasets` folder.

The `dataProcessor.py` file encapsulates the `EEGDataProcessor` class for processing EEG signals.

The `ReceiveData.py` file contains the `LSLDataCollector` class for real-time EEG data acquisition.

To collect EEG data in real-time, you must use an EEG acquisition device that supports the **Lab Streaming Layer (LSL)**. The installation package for LSL is located in the directory `\utils\labstreaminglayer-master`. The `ReceiveData` script, which is based on LSL, is responsible for acquiring EEG data. Please modify line 19 of this script to match the channel presets for your specific EEG device.

**Stimulation Interface**: Located in the folder `utils/SSVEP_App-master/bin/SSVEP_App_debug.exe`. To run this, you need to download and install [openFrameworks 0.10.1](https://openframeworks.cc/) and [Visual Studio 2017](https://visualstudio.microsoft.com/). 

## `BCI-FT`

The `env_cars` class is constructed in `MPC/BCI-FT.py`.

This file implements the `BCI-FT` algorithm described as Algorithm 1 in the corresponding paper.

## `BCI-DRL`

The `env_cars` class is used to enable environment interaction for all DRL methods.

`DQN.py` includes implementations of both `DDQN` and `DUE` algorithms.

`PPO.py` contains the implementation of the `PPO` algorithm.

`SAC.py` contains the implementation of the `SAC` algorithm.
