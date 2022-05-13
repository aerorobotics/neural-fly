# Neural-Fly enables rapid learning for agile flight in strong winds

Michael O'Connell[^equal],
Guanya Shi[^equal],
Xichen Shi,
Kamyar Azizzadenesheli,
Anima Anandkumar,
Yisong Yue, and
Soon-Jo Chung[^corresponding]

[^equal]: equal contribution and alphabetical order

[^corresponding]: corresponding author - sjchung@caltech.edu

This data and code is provided as part of the Science Robotics research article "Neural-Fly enables rapid learning for agile flight in strong winds", published on May 4th, 2022 [here](https://www.science.org/doi/abs/10.1126/scirobotics.abm6597).

## Training and validation script
Please run `training-and-validation.ipynb`, which demonstrates the Domain Adversarially Invariant Meta Learning (DAIML) algorithm. This script trains a wind-invariant representation of the aerodynamic effects on a quadrotor. After training the model, some simple statistics and plots are generated which show the model performance fitting to the training and testing data. 

## Experiment data
Additionally, the data from the experiment results present in the paper is provided. To load the data, run the following in python

    import utils
    Data = utils.load_data(folder='data/experiment')

This will load all of the experiment data as a list of dictionaries. The `i`'th experiment, field `field`, at the `j`th timestep, can be accessed with `Data[i][field][j]`. Available fields are given in the following table.

| `field` | description |
|---------|-------------|
| `t` | time in seconds |
| `p` | position vector in meters |
| `p_d` | desired position vector in meters |
| `v` | velocity vector in m/s |
| `v_d` | desired velocity in m/s |
| `q` | attitude represented as a unit quaternion |
| `R` | rotation matrix (body frame to world frame) |
| `w` | (this is a misnomer - should be $\omega$) angular velocity in rad/s |
| `T_sp` | thrust setpoint sent to the flight controller |
| `q_sp` | attitude command sent to the flight controller |
| `hover_throttle` | throttle at hover computed as a best fit function of the battery voltage |
|  `fa` | aerodynamic residual force computed using numerical differentiation of `v` and `T_sp`, `q`, and `hover_throttle` |
| `pwm` | motor speed commands from the flight controller |
| `method` | identifies controller used, see paper for details on implementation |
| `condition` | wind condition for experiments, where the number corresponds to the fan array duty cycle and corresponds to <ul><li>`nowind` = 0 m/s</li><li>`10wind` = 1.3 m/s</li><li>`20wind` = 2.5m/s</li><li>`30wind` = 3.7 m/s</li><li>`35wind` = 4.2 m/s</li><li>`40wind` = 4.9 m/s</li><li>`50wind` = 6.1 m/s</li><li>`70wind` = 8.5 m/s</li><li>`70p20sint` = 8.5+2.4sin(t) m/s</li><li>`100wind` = 12.1 m/s</ul>


## Videos
Check out our overview video:

[![Full overview video](https://img.youtube.com/vi/iCFcU3i2xIM/mqdefault.jpg)](https://www.youtube.com/watch?v=iCFcU3i2xIM "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds")

60 second overview video created by Caltech's Office of Strategic Communications:

[![60 second overview video](https://img.youtube.com/vi/y3Z5ZJK6FDg/mqdefault.jpg)](https://www.youtube.com/watch?v=y3Z5ZJK6FDg "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds")

## Citation
The data and code here are for personal and educational use only and provided without warantee; written permission from the authors is required for further use. Please cite our work as follows.

> @article{
doi:10.1126/scirobotics.abm6597,
author = {Michael O’Connell  and Guanya Shi  and Xichen Shi  and Kamyar Azizzadenesheli  and Anima Anandkumar  and Yisong Yue  and Soon-Jo Chung },
title = {Neural-Fly enables rapid learning for agile flight in strong winds},
journal = {Science Robotics},
volume = {7},
number = {66},
pages = {eabm6597},
year = {2022},
doi = {10.1126/scirobotics.abm6597},
URL = {https://www.science.org/doi/abs/10.1126/scirobotics.abm6597}}