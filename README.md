# Reinforcement Learning Summer School 2026 - Policy Gradient Tutorial

Direct Policy Search: black-box optimization (BBO) and policy gradient (PG).

Website: https://rlsummerschool.com/

Slides: https://araffin.github.io/slides/dqn-tutorial/

Stable-Baselines3 repo: https://github.com/DLR-RM/stable-baselines3

RL Summer School 2026: https://2026.rlsummerschool.com/

## Content

<!--1. Fitted Q-Iteration (FQI) [Colab Notebook](https://colab.research.google.com/github/araffin/rlss23-dqn-tutorial/blob/main/notebooks/1_fitted_q_iteration_fqi.ipynb)-->

## Run Locally (instead of using Google colab)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. [optional] Create a virtual env with a specific python version: `uv venv --python 3.12 --clear`
3. Run `uv run --with jupyter jupyter lab notebooks` or `make open-notebook`

## Solutions

Solutions can be found in the [notebooks/solutions/](https://github.com/araffin/rlss26-pg-tutorial/tree/master/notebooks/solutions) folder.
The code in `pg_tutorial` package can also be used to bypass some exercises.


## Vehicle Dynamics

The environment uses a differential-drive robot with the following dynamics:

### Action Space

The action space is a 2D continuous space representing wheel speeds:
$$\mathbf{a} = [\omega_{\text{left}}, \omega_{\text{right}}] \in [-1, 1]^2$$

### Wheel Kinematics

The wheel speeds are decoded into linear velocities:
$$v_{\text{left}} = \omega_{\text{left}} \cdot r_{\text{wheel}}$$
$$v_{\text{right}} = \omega_{\text{right}} \cdot r_{\text{wheel}}$$

where $r_{\text{wheel}}$ is the wheel radius.

### Inertia and Friction

Wheel speeds are affected by inertia (first-order lag) and friction:
$$\omega_{\text{left}}^{\text{new}} = \eta \cdot \omega_{\text{left}}^{\text{prev}} + (1 - \eta) \cdot \omega_{\text{left}}^{\text{target}}$$
$$\omega_{\text{right}}^{\text{new}} = \eta \cdot \omega_{\text{right}}^{\text{prev}} + (1 - \eta) \cdot \omega_{\text{right}}^{\text{target}}$$

where $\eta$ is the inertia factor, followed by friction:
$$\omega_{\text{left}}^{\text{final}} = (1 - \mu) \cdot \omega_{\text{left}}^{\text{new}}$$
$$\omega_{\text{right}}^{\text{final}} = (1 - \mu) \cdot \omega_{\text{right}}^{\text{new}}$$

where $\mu$ is the friction coefficient.

### Differential Drive Kinematics

The forward and angular velocities are computed as:
$$v = \frac{v_{\text{left}} + v_{\text{right}}}{2}$$
$$\omega = \frac{v_{\text{right}} - v_{\text{left}}}{L}$$

where $L$ is the wheel base (distance between wheels).

### Position Integration

The robot position is integrated using a no-slip differential-drive model:

For straight-line motion ($|\omega| < \epsilon$):
$$x_{t+1} = x_t + v \cdot \cos(\theta_t) \cdot \Delta t$$
$$y_{t+1} = y_t + v \cdot \sin(\theta_t) \cdot \Delta t$$

For curved motion (arc integration):
$$x_{t+1} = x_t + R \cdot (\sin(\theta_t + \Delta\theta) - \sin(\theta_t))$$
$$y_{t+1} = y_t - R \cdot (\cos(\theta_t + \Delta\theta) - \cos(\theta_t))$$
$$\theta_{t+1} = \theta_t + \Delta\theta$$

where $R = v/\omega$ is the turn radius and $\Delta\theta = \omega \cdot \Delta t$.
