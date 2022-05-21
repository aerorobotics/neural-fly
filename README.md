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

Please run `training-and-validation.ipynb`, which demonstrates the Domain Adversarially Invariant Meta Learning (DAIML) algorithm. DAIML is the offline learning process for Neural-Fly. This script trains a wind-invariant representation of the aerodynamic effects on a quadrotor. After training the model, some simple statistics and plots are generated which show the model performance fitting to the training and testing data. 

## Filenaming scheme

Filenames are structured as

    <VEHICLE>_<TRAJECTORY>_<METHOD>_<CONDITION>.csv

For specific details please refer to the article.

| Field | Description |
| --- | --- |
| `<VEHICLE>` | <li>`custom`: drone built using consumer off-the-shelf components with PX4 flight controller</li><li> `intel`: Intel-Aero RTF Drone, used for data collection for Neural-Fly-Transfer controller |
| `<TRAJECTORY>` | <li>`random3`: randomized trajectory created by randomly sampling 2 waypoints and generating a smooth spline from the current position through both waypoints; continuous derivatives through snap </li><li> `random2`: similar to `random3` except only one random waypoint is generated</li><li> `figure8`: a lemniscate trajectory given by <ul> `(x(t),y(t),z(t)) = (1.25 * sin(t), 0, 0.75 * sin(2 * t)` |
| `<METHOD>` | <li>`'baseline'`: nonlinear baseline control </li><li> `'indi'`: incremental nonlinear dynamics inversion control </li><li> `'L1'`: L1 adaptive control </li><li> `'NF-C'`: Neural-Fly-Constant, our adaptive controller without any learning </li><li> `'NF-T'`: Neural-Fly-Transfer, our learning based adaptive control with the ML model trained on data from the Intel-Aero drone </li><li> `'NF'`: Neural-Fly, our learning based adaptive control with the ML model trained on data collected with the custom drone , </li> |
| `<CONDITION>` | wind condition for experiments, where the number corresponds to the fan array duty cycle and converts to <ul><li>`nowind` = 0 m/s</li><li>`10wind` = 1.3 m/s</li><li>`20wind` = 2.5m/s</li><li>`30wind` = 3.7 m/s</li><li>`35wind` = 4.2 m/s</li><li>`40wind` = 4.9 m/s</li><li>`50wind` = 6.1 m/s</li><li>`70wind` = 8.5 m/s</li><li>`70p20sint` = 8.5+2.4sin(t) m/s</li><li>`100wind` = 12.1 m/s</ul>

## Experiment data

Additionally, the data from the experiment results present in the paper is provided. To load the data, run the following in python

    import utils
    Data = utils.load_data(folder='data/experiment')

This will load all of the experiment data as a list of dictionaries. The `i`th experiment, field `field`, at the `j`th timestep, can be accessed with `Data[i][field][j]`. Most fields are ndarrays except the metadata fields, pulled from the filename. Available fields are given in the following table.

| `field` | description |
|---------|-------------|
| `'t'` | time in seconds |
| `'p'` | position vector in meters |
| `'p_d'` | desired position vector in meters |
| `'v'` | velocity vector in m/s |
| `'v_d'` | desired velocity in m/s |
| `'q'` | attitude represented as a unit quaternion |
| `'R'` | rotation matrix (body frame to world frame) |
| `'w'` | (this is a misnomer - should be $\omega$) angular velocity in rad/s |
| `'T_sp'` | thrust setpoint sent to the flight controller |
| `'q_sp'` | attitude command sent to the flight controller |
| `'hover_throttle'` | throttle at hover computed as a best fit function of the battery voltage |
|  '`fa'` | aerodynamic residual force computed using numerical differentiation of `v` and `T_sp`, `q`, and `hover_throttle` |
| `'pwm'` | motor speed commands from the flight controller |
| `'vehicle'` | `<VEHICLE>` field from filename |
| `'trajectory'` | `<TRAJECTORY>` field from filename |
| `'method'` | `<METHOD>` field from filename |
| `'condition'` | `<CONDITION>` field from filename |

## Videos

Check out our overview video:

[![Full overview video](https://img.youtube.com/vi/iCFcU3i2xIM/mqdefault.jpg)](https://www.youtube.com/watch?v=iCFcU3i2xIM "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds")

60 second overview video created by Caltech's Office of Strategic Communications:

[![60 second overview video](https://img.youtube.com/vi/y3Z5ZJK6FDg/mqdefault.jpg)](https://www.youtube.com/watch?v=y3Z5ZJK6FDg "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds")

## Citation

The data and code here are for personal and educational use only and provided without warranty; written permission from the authors is required for further use. Please cite our work as follows.

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
URL = {<https://www.science.org/doi/abs/10.1126/scirobotics.abm6597>}}
