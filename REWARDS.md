<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Reward functions used by `train.sh`

Snapshot of the shared X1 and Go2 configuration on **2026-10-12**.

Both robots use **four positive reward terms and thirteen nonzero penalty terms**.
The positive weights sum to **1.0**. Nine penalties use
$f_i=\exp(-k_i E_i)-1$, with nonnegative errors summed without averaging;
their combined weighted penalty is bounded by $[-2.5,0]$. The two foot-periodicity
terms apply a gated exponential-minus-one to each foot before summing, with
weight 0.3 per foot for each component and a combined bound of $[-1.2,0]$.
Foot clearance adds a per-foot absolute swing-trajectory error penalty with weight 0.1
and no command gate. Leg-permutation symmetry adds a penalty with weight 0.1,
bounded by $[-0.1,0]$. Torque-difference smoothness remains disabled by a zero weight. `--no-trs`
controls learning-time symmetry separately from the environment rewards.

The positive terms use exponential tracking scores for velocity, squared base-height
error, and squared roll plus pitch. Swing force, stance speed, and absolute swing-height
error contribute separate penalties. Both velocity-tracking terms use
the edited standard implementations with a direct denominator of 0.25.
Previously trained checkpoints may have used different settings.

## Reward table

Each equation defines the **raw function output** $f_i$, before multiplying by its
weight $w_i$ and the control-step duration. All configured weights are nonnegative;
penalty functions themselves return nonpositive values.

| Configuration term | Raw reward or penalty $f_i$ | Weight $w_i$ |
| --- | --- | ---: |
| `track_lin_vel_xy_exp` | $f_{xy}=\exp[-((v_x-v_x^*)^2+(v_y-v_y^*)^2)/0.25]$ | 0.40 |
| `track_ang_vel_z_exp` | $f_{\mathrm{yaw}}=\exp[-(\omega_z-\omega_z^*)^2/0.25]$ | 0.30 |
| `base_height` | $f_{\mathrm{height}}=\exp[-100(h-h^*)^2]$ | 0.15 |
| `base_roll_pitch_exp` | $f_{\mathrm{orientation}}=\exp[-10(\rho^2+\eta^2)]$ | 0.15 |
| `foot_clearance` | $f_{\mathrm{clear}}=\sum_\ell s_\ell[\exp(-40\lvert z_\ell^*-z_\ell\rvert)-1]$ | 0.10 |
| `foot_periodicity_force` | $f_F=\sum_\ell S_\ell[\exp(-0.05F_\ell)-1]$ | 0.30 |
| `foot_periodicity_velocity` | $f_u=\sum_\ell T_\ell[\exp(-2u_\ell)-1]$ | 0.30 |
| `hip_action_penalty` | $f_{\mathrm{hip}}=\exp[-\sum_{j\in\mathrm{hips}}(q_j-q_j^{\mathrm{default}})^2]-1$ | 0.10 |
| `joint_target_limits` | $f_{\mathrm{limit}}=\exp[-10E_{\mathrm{limit}}]-1$ | 2.00 |
| `action_rate_l2` | $f_{\mathrm{rate}}=\exp[-0.05\sum_j(a_{t,j}-a_{t-1,j})^2]-1$ | 0.05 |
| `lin_vel_z_l2` | $f_{v_z}=\exp[-5(v_z^b)^2]-1$ | 0.10 |
| `dof_vel_l2` | $f_{\dot q}=\exp[-0.01\sum_j\dot q_j^2]-1$ | 0.05 |
| `dof_torques_l2` | $f_\tau=\exp[-0.004\sum_j\tau_j^2]-1$ | 0.05 |
| `action_smoothness` | $f_{\Delta^2a}=\exp[-0.05\sum_j(a_{t,j}-2a_{t-1,j}+a_{t-2,j})^2]-1$ | 0.05 |
| `ang_vel_xy_l2` | $f_{\omega_{xy}}=\exp[-((\omega_x^b)^2+(\omega_y^b)^2)]-1$ | 0.05 |
| `dof_acc_l2` | $f_{\ddot q}=\exp[-4\times10^{-6}\sum_j((\dot q_{t,j}-\dot q_{t-1,j})/\Delta t)^2]-1$ | 0.05 |
| `leg_permutation_symmetry` | $f_{\mathrm{sym}}=e^{-5\overline E}-1$ | 0.10 |
| `smoothness` (zero weight) | $f_{\mathrm{smooth}}=-G_{\mathrm{stable}}\left(1-e^{-0.10D_\tau}\right)$ | 0 |
| `alive_bonus` (zero weight) | $f_{\mathrm{alive}}=1$ | 0 |
| `termination_penalty` (zero weight) | $f_{\mathrm{term}}=\mathbf 1[\text{non-timeout termination}]$ | 0 |

The retained `hip_action_penalty` and `joint_target_limits` configuration keys
now select measured-position penalties. `action_rate_l2` selects
`action_rate_l2_exp_penalty`, using squared action differences. The nine
joint, body-velocity, and action penalties have no stability gate. The old action-based hip, L1 action-rate,
and target-limit helpers remain available for explicit legacy configurations.
Both robots use the same weights, with these height settings:

| Robot | Base-height target $h^*$ [m] | Clearance target $H$ [m] | Foot offset $o$ [m] | Command nominal height range [m] |
| --- | ---: | ---: | ---: | --- |
| X1 | 0.40 | 0.08 | 0.0275 | 0.45–0.60 |
| Go2 | 0.35 | 0.06 | 0.022 | 0.35–0.45 |

The foot offsets match the spherical foot collision radii in the robot URDFs.
The peak commanded foot-link height is $H+o$: 0.1075 m for X1 and
0.082 m for Go2. Each foot follows a sinusoidal swing-height target with that
peak, starting and ending at the resting foot offset $o$. The command nominal
height range remains a separate gait-timing setting.

## Final reward composition

The physics step is $0.005\,\mathrm{s}$ and decimation is 4, so the control-step
duration is $\Delta t=0.02\,\mathrm{s}$. The reward manager first computes

$$
r_i=\Delta t\,w_i f_i.
$$

The environment sums all terms, including any `termination_penalty`, and clips
the final reward to $[0,\infty)$:

$$
R=\max\!\left(\sum_i r_i,0\right).
$$

Ji22 composition and its diagnostics are commented out. Its helper remains
available for restoring the previous experiment. Termination penalties, if
enabled, are included before clipping so the delivered reward remains nonnegative.

The current termination weight is zero. Before timestep scaling, define

$$
\begin{aligned}
A={}&0.40f_{xy}+0.30f_{\mathrm{yaw}}+0.15f_{\mathrm{height}}
+0.15f_{\mathrm{orientation}},\\
B={}&0.10f_{\mathrm{hip}}+2.00f_{\mathrm{limit}}+0.05f_{\mathrm{rate}}+0.10f_{v_z}\\
&+0.05f_{\dot q}+0.05f_\tau+0.05f_{\Delta^2a}+0.05f_{\omega_{xy}}+0.05f_{\ddot q}
+0.30f_F+0.30f_u+0.10f_{\mathrm{clear}}+0.10f_{\mathrm{sym}}.
\end{aligned}
$$

At the current step duration, the delivered reward is therefore

$$
\boxed{R=0.02\max(A+B,0).}
$$

Each positive raw term is at most 1, so $A\le1$. The total reward is bounded
by **$[0,1]$ before timestep scaling**, or **$[0,0.02]$ per step**. The thirteen
penalty terms are added to the positive sum before clipping at zero.

`Episode_Reward/<term>` logs report individual weighted terms before this
clipping, averaged over reset environments and normalized by episode length.
Their sum does not reconstruct the delivered reward when clipping occurs. The
three zero-weight terms are skipped; termination conditions remain active.

## Command-velocity curriculum

Go2 and X1 use the same enabled command-velocity curriculum:

| Command | Configured initial range | Maximum range |
| --- | --- | --- |
| Forward/backward velocity [m/s] | -0.5 to 0.5 | -3.0 to 3.0 |
| Lateral velocity [m/s] | -0.6 to 0.6 | -0.6 to 0.6 |
| Yaw rate [rad/s] | -0.5 to 0.5 | -2.0 to 2.0 |

Sampling uses 21 forward-velocity bins, one lateral-velocity bin, and 21 yaw-rate
bins. Initial bins are selected by their centers, and commands are sampled
throughout each active bin, so initial samples can extend slightly beyond the
configured initial ranges. A successful command window adds 0.2 to the sampling
weights of its bin and its immediate neighbors, capped at 1.0.

Success requires both the mean planar tracking score to exceed **0.85** and
the mean yaw tracking score to exceed **0.70**. Scores use
$\exp(-\mathrm{squared\ error}/0.25)$ independently of the outer reward weights.
Gait correctness and foot-clearance error do not participate in this success check.

## Notation

| Symbol | Meaning |
| --- | --- |
| $v_x^*,v_y^*,\omega_z^*$ | Commanded body-frame planar velocity [m/s] and yaw rate [rad/s] |
| $v_x,v_y,\omega_z$ | Measured body-frame planar velocity [m/s] and yaw rate [rad/s] |
| $z,\rho,\eta$ | Base world height [m], roll [rad], and pitch [rad] |
| $h,h^*$ | Ground-relative base height and its robot-specific target [m] |
| $z_\ell,H,o$ | Ground-relative foot-link height, clearance target, and resting foot offset [m] |
| $z_\ell^*,s_\ell$ | Commanded swing-height trajectory [m] and sigmoid swing gate |
| $a_{t,j}$ | Dimensionless action stored in the action manager |
| $q_j,\widehat q_j$ | Measured joint position and processed, clamped joint target [rad] |
| $q_j^{\mathrm{default}},\dot q_j$ | Default joint position [rad] and measured joint velocity [rad/s] |
| $v_z^b,\omega_x^b,\omega_y^b$ | Body-frame vertical velocity [m/s] and roll/pitch angular velocities [rad/s] |
| $\tau_{t,j}$ | Applied joint torque [N·m] at reward evaluation |
| $\phi_\ell,d,\theta_\ell$ | Foot phase [cycles], stance duty factor, and sampled foot offset [cycles] |
| $\ell$ | Leg in FL, FR, RL, RR order |

Use $[x]_+=\max(x,0)$, $\operatorname{clip}(x,l,u)=\min(\max(x,l),u)$,
and $\sigma(x)=(1+e^{-x})^{-1}$. Numerical constants assume the stated units.
Joint sums cover all 12 joints unless otherwise specified.

## Helper equations

### Foot periodicity

The contact-force magnitude uses the full three-dimensional net contact force
[N] and takes the maximum over three stored sensor samples. Foot speed [m/s]
uses all three world-frame linear velocity components:

$$
F_\ell=\max_{h\in\{1,2,3\}}\lVert\mathbf F^w_{\ell,h}\rVert_2,
\qquad u_\ell=\lVert\mathbf v^w_{\mathrm{foot},\ell}\rVert_2.
$$

Two periodicity penalties discourage contact force during swing and foot
motion during stance. Their internal coefficients are $w_S=0.05\,\mathrm{N}^{-1}$
and $w_T=2\,\mathrm{s/m}$. Each foot contributes

$$
r_{F,\ell}=0.3S_\ell[\exp(-0.05F_\ell)-1],\qquad
r_{u,\ell}=0.3T_\ell[\exp(-2u_\ell)-1].
$$

The swing and stance gates multiply the per-foot exponential-minus-one outputs.
The four foot contributions are summed afterwards, without averaging. Configuring
weight 0.3 on each summed component gives every foot that same weight. Stance
contact itself is not required by these functions; stance motion incurs a penalty.

Let $C_{16}$ be the zero-location CDF used by `scipy.stats.vonmises_line`, with
concentration $\kappa=16$. The implemented smoothed periodic interval weight is

$$
V(\phi;a,b)=\sum_{m\in\{-1,0,1\}}
C_{16}\!\left(2\pi(\phi-a-m)\right)
\left[1-C_{16}\!\left(2\pi(\phi-b-m)\right)\right],
$$

$$
S_\ell=V(\phi_\ell;0,1-d),\qquad T_\ell=V(\phi_\ell;1-d,1).
$$

The command helpers return negative weights, so the penalty functions negate
them to obtain the nonnegative gates $S_\ell$ and $T_\ell$. These complementary
gates satisfy $S_\ell+T_\ell=1$. Each foot's combined weighted penalty is bounded
by $[-0.3,0]$, giving $[-1.2,0]$ over all four feet. The lower bound is
approached as force and speed errors increase.

The duty factor varies with the gait command. Foot phases follow the signed
clock and signed offsets implemented by `GaitVelocityCommand`.

Running `bash test.sh` saves `figure11_foot_periodicity.png` and
`foot_periodicity.csv` in each checkpoint/gait output directory. The plot shows
history-max net contact force [N], world-frame foot speed [m/s], and each foot's
force and velocity penalties, with commanded stance and swing shaded. The CSV
reports mean force, speed, `mean_force_penalty`, and `mean_speed_penalty` per leg,
phase, and command direction. These penalties include the exponential transformation
and phase gates but exclude the external 0.3 weights and timestep scaling.
Missing phases have zero samples and `nan` means.

Direction-test playback calculates cycle duration and stance duty factor from
the specified velocity using the same timing curve and `add_noise_period`
setting as training. Timing is recalculated at startup, after episode resets,
and when the velocity command changes. It remains fixed while a command is
held, so timing noise is not sampled again on every control step. Startup and
reset timing are applied immediately; later changes use the configured smooth
gait transition without restarting the phase clock. The curve depends on the
absolute forward velocity component and the midpoint of
`commands.base_velocity.base_height_range`; lateral and yaw-only commands
therefore use the zero-forward-speed timing. Set
`env.commands.base_velocity.add_noise_period=False` to use the nominal curve
without timing noise during evaluation.

Reward-time signals are captured before resets or gait resampling and saved as
`foot_periodicity_*` arrays in `sim_data.npz`. Previous positive-reward recordings
remain readable using their `mean_force_cost` and `mean_speed_cost` columns.
The existing ground-reaction-force plots use instantaneous force,
including available friction, rather than the history-max net force used here.

### Foot clearance

The active `swing_trajectory_penalty` mode penalizes absolute height error
relative to a sinusoidal swing trajectory. The trajectory includes the resting
foot offset, with no command gate and the same foot-link peak heights:

$$
p_\ell=\operatorname{clip}\!\left(\frac{\phi_\ell}{1-d},0,1\right),\qquad
z_\ell^*=o+H\sin(\pi p_\ell),\qquad
s_\ell=\sigma\!\left(\kappa[(1-d)-\phi_\ell]\right),
$$

$$
e_\ell=\lvert z_\ell^*-z_\ell\rvert,\qquad
f_{\mathrm{clear}}=\sum_\ell s_\ell\left[\exp\!\left(-\frac{e_\ell}{0.025}\right)-1\right].
$$

The denominator is `height_scale=0.025` m, giving coefficient
$40\,\mathrm{m}^{-1}$ on absolute height error. The outer weight is 0.1. Each foot's
exponential is evaluated before summing; there is no averaging. The sigmoid
swing gate $s_\ell$ uses $\kappa=16$ and differs from the von Mises periodicity
gate $S_\ell$.

Foot height is measured above the environment origin. The trajectory rises
from the resting offset $o$ to $H+o$ at mid-swing and returns to $o$ at swing
end, remaining at $o$ during stance. Heights below and above the trajectory
are penalized equally for equal deviations; matching the trajectory gives
zero penalty. The sigmoid swing gate still scales the penalty. The active
configuration sets `use_absolute_error=True`, the robot-specific
`foot_height_offset`, and `use_command_gate=False`, so stationary feet and
zero velocity commands still incur a penalty when a foot deviates from its
commanded swing trajectory.
The weighted clearance term is bounded by $[-0.1\sum_\ell s_\ell,0]$, contained
within $[-0.4,0]$ for four feet.

Playback records the negative clearance penalty, foot heights, trajectory
targets, swing weights, and `foot_clearance_height_errors` (absolute error in
the active mode). The existing `foot_clearance_shortfalls` records only
underlifting. Training also logs `Diagnostics/foot_clearance_mean_height_error`,
the swing-weighted mean error. `figure7_foot_clearance.png` shows the trajectory
and shades the active swing weight. Explicit legacy modes remain available:
`phase_penalty` retains the command gate, zero baseline, and shortfall-only
error. Set `use_absolute_error=False` and `foot_height_offset=0.0` on the
clearance term to reproduce the previous ungated shortfall trajectory with
the configured peak. `speed_tracking_reward` selects
the previous positive speed-weighted clearance reward with `sigma=0.01`.

### Action rate and action smoothness

$$
E_{\mathrm{rate}}=\sum_{j=1}^{12}(a_{t,j}-a_{t-1,j})^2,
\qquad
E_{\mathrm{smooth}}=\sum_{j=1}^{12}(a_{t,j}-2a_{t-1,j}+a_{t-2,j})^2.
$$

Both terms use raw, dimensionless policy actions before target scaling and
clamping, without averaging, timestep division, or stability gates. Their
exponential coefficients are both 0.05; both outer weights are
0.05. Action histories are initialized to zero independently for each reset
environment. The first action therefore contributes to both penalties.

### Velocity, torque, and acceleration

Base linear and angular velocities use the body frame. `ang_vel_xy_l2`
penalizes both roll and pitch angular velocity, while excluding yaw. Joint
velocity and torque errors sum squares over the 12 configured robot joints.
Torques are the applied torques at reward evaluation.

`JointAccelerationExpPenalty` differences measured joint velocities between
consecutive control steps and divides by $\Delta t=0.02$ s. It keeps its own
history instead of using a backend's physics-substep acceleration. Reset
environments initialize that history from their reset joint velocities; other
environments retain their history.

### Torque smoothness (disabled by zero weight)

The retained `SmoothnessPenalty` function measures

$$
D_\tau=\sum_{j=1}^{12}\lvert\tau_{t,j}-\tau_{t-1,j}\rvert.
$$

Its legacy stability gate is

$$
G_{\mathrm{stable}}=\operatorname{clip}\!\left(
\sigma(20(z-0.25))e^{-2(\rho^2+\eta^2)},0,1\right).
$$

With the current weight of 0.0, the reward manager skips its evaluation and it
does not enter the penalty sum $B$. If enabled, the function sets $D_\tau$ to zero
on the first reward evaluation after reset, uses the stability gate above, and
applies exponential scale $0.10\,(\mathrm{N\,m})^{-1}$. The torque difference is
not divided by $\Delta t$.

### Hip joint position

The configured term uses the four measured hip angles and their default
positions, with weight 0.10:

$$
E_{\mathrm{hip}}=\sum_{j\in\mathrm{hips}}(q_j-q_j^{\mathrm{default}})^2,
\qquad f_{\mathrm{hip}}=\exp(-E_{\mathrm{hip}})-1.
$$

Hip joint names come from positions 0, 3, 6, and 9 in the logical joint order
and are resolved to articulation indices. This selects Go2's hip joints and
X1's abduction joints even when simulator storage order differs. Angles are
in radians; the exponential coefficient is $1\,\mathrm{rad}^{-2}$.

### Measured joint-position limits

Let $L_j,U_j$ be the full runtime joint limits [rad], including X1's startup
calf-limit overrides. The allowed region covers their central 90%:

$$
L_j^{\mathrm{soft}}=L_j+0.05(U_j-L_j),\qquad
U_j^{\mathrm{soft}}=U_j-0.05(U_j-L_j),
$$

$$
E_{\mathrm{limit}}=\sum_j\left([L_j^{\mathrm{soft}}-q_j]_+
+[q_j-U_j^{\mathrm{soft}}]_+\right),\qquad
f_{\mathrm{limit}}=\exp(-10E_{\mathrm{limit}})-1.
$$

This uses measured joint angles and linear violations in radians, without
squaring or averaging. The coefficient is $10\,\mathrm{rad}^{-1}$ and weight
is 2.0. The bounds are computed from full limits so existing soft limits are
not shrunk twice. Target clamping remains a separate simulator safety step.

### Leg-permutation symmetry

The current weight is 0.1 for both robots. The weighted penalty
$0.1[\exp(-5\overline E)-1]$ enters the penalty sum $B$ and is bounded by
$[-0.1,0]$ before timestep scaling.

Map the measured joint angles into logical coordinates:

$$
\overline{\mathbf q}_\ell=\mathbf s_\ell\odot\mathbf q_\ell.
$$

Go2 uses $\mathbf s_\ell=(1,1,1)$ for all legs. X1 uses
$\mathbf s_{FL}=\mathbf s_{FR}=(1,1,1)$ and
$\mathbf s_{RL}=\mathbf s_{RR}=(1,-1,-1)$ to express its mirrored front–rear posture.

Joint order within each leg is hip abduction, thigh pitch, and calf pitch.
The six candidate pairs are

$$
\mathcal P=\{(FL,FR),(RL,RR),(FL,RL),(FR,RR),(FL,RR),(RL,FR)\}.
$$

Define the synchronization mask using circular offset distance:

$$
d_{ab}=\min\!\left((\theta_a-\theta_b)\bmod1,(\theta_b-\theta_a)\bmod1\right),
\qquad I_{ab}=\mathbf1[d_{ab}\leq0.02].
$$

Let $\mathbf m_{ab}=(-1,1,1)$ for opposite left/right sides and $(1,1,1)$ for
the same side. Using fixed normalization ranges
$\mathbf Q=(1.3264,5.236,5.06)$ rad,

$$
e_{ab,k}=\frac{\lvert\overline q_{a,k}-m_{ab,k}\overline q_{b,k}\rvert}{Q_k+10^{-6}},
\qquad \beta_{ab,k}=\frac{e^{e_{ab,k}/0.5}}{\sum_{n=1}^3e^{e_{ab,n}/0.5}},
\qquad E_{ab}=\sum_{k=1}^3\beta_{ab,k}e_{ab,k},
$$

$$
\overline E=\frac{\sum_{(a,b)\in\mathcal P}I_{ab}E_{ab}}
{\max\!\left(1,\sum_{(a,b)\in\mathcal P}I_{ab}\right)}.
$$

The raw penalty is zero when no pair is synchronous.
It uses measured joint angles without subtracting the default pose. Changing
`--no-trs` does not change the environment reward's weight.

## Disabled terms

`smoothness`, `alive_bonus`, and `termination_penalty`
are retained in the configuration with weight 0.0 and skipped during reward
computation. To restore the previous torque-smoothness penalty explicitly,
pass `env.rewards.smoothness.weight=0.1` when launching training.

To disable the active leg-permutation symmetry penalty, pass
`env.rewards.leg_permutation_symmetry.weight=0.0`.

The following terms are set to `None`: `feet_air_time`, `flat_orientation_l2`,
`dof_pos_limits`, `undesired_contacts`, `cmd`, `track_lin_vel_x_exp`,
`track_lin_vel_y_exp`, `base_roll_exp`, `foot_periodicity`, `sagittal_plane`, and `straight_line_motion`.

## Sources and other formats

- [Training launcher](train.sh) and [CLI options](scripts/symm_locomotion/symm_cli.py).
- [X1 reward binding](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/flat_env_cfg.py).
- [Go2 reward binding](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/flat_env_cfg.py).
- [Shared reward weights and parameters](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/flat_env_cfg.py).
- [Shared reward functions](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symm_quadruped.py).
- [Standard planar tracking reward](source/isaaclab/isaaclab/envs/mdp/rewards.py).
- [X1 logical joint signs](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/dobot_x1_symm.py).
- [Final reward composition](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/env.py) and [reward manager](source/isaaclab/isaaclab/managers/reward_manager.py).
- [Inherited configuration disabling undesired contacts](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py).
- Earlier **2026-10-05 X1 snapshot**: [LaTeX source](logs/reward_documentation/train_rewards_2026-10-05.tex) and [compiled PDF](logs/reward_documentation/train_rewards_2026-10-05.pdf). These retain the earlier force coefficient, mean-squared action-rate formula, and nonzero torque-smoothness and leg-permutation symmetry weights; the current configuration is documented above.
