<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Reward functions used by `train.sh`

Snapshot of the current workspace configuration on **2026-10-06**.
The active command in [train.sh](train.sh) selects
`Isaac-Velocity-Flat-Dobot-X1-Symm-v0` with TRS enabled, 10,000 training iterations,
2,048 environments, and 20 frames of policy observation history. It uses
**11 nonzero environment reward terms** and supplies no reward overrides.

The second command, which includes `--no-trs`, is currently commented out. If
enabled, it would use the same environment rewards: `--no-trs` disables symmetry
data augmentation and learning-time symmetry losses, while the environment's
`leg_permutation_symmetry` reward remains enabled. The observation-history setting
does not change these reward functions.

These equations describe the current source configuration. Previously trained
checkpoints may have used different settings.

Compared with the 2026-10-05 documentation, the foot-periodicity contact-force
coefficient is now **0.025 N⁻¹**, up from 0.005 N⁻¹. Its weight remains **0.30**;
all other reward formulas and weights in the table remain unchanged.

## Reward table

Each equation defines the **raw function output** $f_i$, before multiplying by its
weight $w_i$ and the control-step duration. All configured weights are nonnegative;
penalty functions themselves return nonpositive values.

| Configuration term | Raw reward or penalty $f_i$ | Weight $w_i$ |
| --- | --- | ---: |
| `track_lin_vel_xy_exp` | $f_{xy}=\exp\!\left[-\dfrac{(v_x-v_x^*)^2+(v_y-v_y^*)^2}{0.20^2}\right]$ | 0.50 |
| `track_ang_vel_z_exp` | $f_{\mathrm{yaw}}=\exp\!\left[-\left(\dfrac{\omega_z-\omega_z^*}{0.20}\right)^2\right]$ | 0.50 |
| `base_roll_exp` | $f_{\mathrm{roll}}=e^{-\lvert\rho\rvert/0.25}-1$ | 0.30 |
| `foot_periodicity` | $f_{\mathrm{period}}=-\displaystyle\sum_{\ell=1}^{4}\left[S_\ell(1-e^{-0.025F_\ell})+T_\ell(1-e^{-2u_\ell})\right]$ | 0.30 |
| `base_height` | $f_{\mathrm{height}}=e^{-20\lvert z-0.35\rvert}-1$ | 0.30 |
| `foot_clearance` | $f_{\mathrm{clear}}=-G_{\mathrm{cmd}}\displaystyle\sum_{\ell=1}^{4}W_\ell\left(1-e^{-[h_\ell^*-h_\ell]_+/0.025}\right)$ | 0.10 |
| `hip_action_penalty` | $f_{\mathrm{hip}}=e^{-0.5H}-1$ | 0.15 |
| `joint_target_limits` | $f_{\mathrm{limit}}=-\dfrac1{12}\displaystyle\sum_{j=1}^{12}\left[\operatorname{clip}\!\left(\dfrac{x_j-0.90}{0.10},0,1\right)\right]^2$ | 0.05 |
| `action_rate_l2` | $f_{\mathrm{rate}}=-G_{\mathrm{stable}}\left(1-e^{-0.10D_a}\right)$ | 0.05 |
| `leg_permutation_symmetry` | $f_{\mathrm{sym}}=e^{-5\overline E}-1$ | 0.20 |
| `smoothness` | $f_{\mathrm{smooth}}=-G_{\mathrm{stable}}\left(1-e^{-0.10D_\tau}\right)$ | 0.10 |
| `alive_bonus` (zero weight) | $f_{\mathrm{alive}}=1$ | 0 |
| `termination_penalty` (zero weight) | $f_{\mathrm{term}}=\mathbf 1[\text{non-timeout termination}]$ | 0 |

The `action_rate_l2` key selects `action_rate_exp_penalty`, a bounded exponential
of the mean squared action difference. The base-height reward target is **0.35 m**;
the command's nominal height range of 0.45–0.60 m is a separate gait-timing setting.

## Final reward composition

The physics step is $0.005\,\mathrm{s}$ and decimation is 4, so the control-step
duration is $\Delta t=0.02\,\mathrm{s}$. The reward manager first computes

$$
r_i=\Delta t\,w_i f_i.
$$

The environment then applies its Ji22 composition. With $\mathcal I$ containing
all configured terms except `termination_penalty`,

$$
P=\sum_{i\in\mathcal I}\max(r_i,0),\qquad
N=\sum_{i\in\mathcal I}\min(r_i,0),\qquad
R=P\exp\!\left(\frac{N}{0.02}\right)+r_{\mathrm{term}}.
$$

The current termination weight is zero. Define the weighted penalty sum before
multiplication by $\Delta t$:

$$
\begin{aligned}
B={}&0.30f_{\mathrm{roll}}+0.30f_{\mathrm{period}}+0.30f_{\mathrm{height}}
     +0.10f_{\mathrm{clear}}+0.15f_{\mathrm{hip}}\\
   &+0.05f_{\mathrm{limit}}+0.05f_{\mathrm{rate}}+0.20f_{\mathrm{sym}}
     +0.10f_{\mathrm{smooth}}.
\end{aligned}
$$

At the current step duration, the delivered reward is therefore

$$
\boxed{R=0.02\left(0.50f_{xy}+0.50f_{\mathrm{yaw}}\right)e^B.}
$$

The two tracking rewards provide the positive component; the nine penalties
reduce it through the exponential multiplier. `Episode_Reward/<term>` logs use
the accumulated individual weighted terms before this composition, averaged over
reset environments and divided by the configured episode duration (30 s). Their
sum does not reconstruct the delivered reward. The reward manager skips the two
zero-weight terms; episode termination conditions remain active despite the zero
explicit termination reward.

## Notation

| Symbol | Meaning |
| --- | --- |
| $v_x^*,v_y^*,\omega_z^*$ | Commanded body-frame planar velocity [m/s] and yaw rate [rad/s] |
| $v_x,v_y,\omega_z$ | Measured body-frame planar velocity [m/s] and yaw rate [rad/s] |
| $z,\rho,\eta$ | Base world height [m], roll [rad], and pitch [rad] |
| $a_{t,j}$ | Dimensionless action stored in the action manager |
| $q_j,\widehat q_j$ | Measured joint position and processed, clamped joint target [rad] |
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

The periodicity penalty discourages contact force during swing and foot motion
during stance. The coefficients are $0.025\,\mathrm{N}^{-1}$ and
$2\,\mathrm{s/m}$, respectively. The term sums over the four feet. The training
configuration explicitly passes `force_scale=0.025`, overriding the function's
standalone default of 0.001.

For example, a 100 N contact force with swing weight $S_\ell=1$ contributes
$-(1-e^{-2.5})\approx-0.9179$ before the reward weight and step duration are
applied. The previously documented coefficient of 0.005 would give about
$-0.3935$. Increasing the coefficient strengthens swing-contact penalties, with
the exponential still saturating toward $-1$ per foot's force component. This
term does not directly require stance contact: its stance component only
penalizes foot speed.

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

The duty factor varies with the gait command. Foot phases follow the signed
clock and signed offsets implemented by `GaitVelocityCommand`.

### Swing foot clearance

Foot height $h_\ell$ is the foot-link world height minus the environment-origin
height [m]. The peak target is 0.10 m:

$$
\xi_\ell=\operatorname{clip}\!\left(\frac{\phi_\ell}{\max(1-d,10^{-6})},0,1\right),
\qquad h_\ell^*=0.10\sin(\pi\xi_\ell),
\qquad W_\ell=\sigma\!\left(16[(1-d)-\phi_\ell]\right).
$$

The command gate allows either translation or turning to enable the penalty:

$$
G_{\mathrm{cmd}}=\operatorname{clip}\!\left(
\max\!\left[
\frac{\sqrt{(v_x^*)^2+(v_y^*)^2}}{0.20\,\mathrm{m/s}},
\frac{\lvert\omega_z^*\rvert}{0.20\,\mathrm{rad/s}}
\right],0,1\right).
$$

Only clearance shortfall is penalized, with scale 0.025 m. The sigmoid weight
$W_\ell$ differs from the periodicity weight $S_\ell$.

### Stability, action change, and torque smoothness

$$
G_{\mathrm{stable}}=\operatorname{clip}\!\left(
\sigma(20(z-0.25))e^{-2(\rho^2+\eta^2)},0,1\right),
$$

$$
D_a=\frac1{12}\sum_{j=1}^{12}(a_{t,j}-a_{t-1,j})^2,
\qquad D_\tau=\sum_{j=1}^{12}\lvert\tau_{t,j}-\tau_{t-1,j}\rvert.
$$

The stability gate uses a 0.25 m height threshold, height coefficient
$20\,\mathrm{m}^{-1}$, and orientation coefficient $2\,\mathrm{rad}^{-2}$.
$D_\tau$ is set to zero on the first reward evaluation after reset. Neither
difference is divided by $\Delta t$. The torque-change scale is
$0.10\,(\mathrm{N\,m})^{-1}$.

### Hip action magnitude

For zero-based hip-action indices $\mathcal A=\{0,3,6,9\}$,

$$
\alpha_j=\frac{e^{\lvert a_{t,j}\rvert/0.5}}
{\sum_{k\in\mathcal A}e^{\lvert a_{t,k}\rvert/0.5}},
\qquad H=\sum_{j\in\mathcal A}\alpha_j\lvert a_{t,j}\rvert.
$$

These are unscaled actions, rather than measured hip angles or joint targets.

### Joint-target limits

Let $L_j,U_j$ be the current soft joint-position limits [rad]. With $\epsilon$
equal to the machine epsilon of the limit tensor's dtype,

$$
x_j=\left\lvert\frac{2(\widehat q_j-L_j)}{\max(U_j-L_j,\epsilon)}-1\right\rvert.
$$

The penalty activates within 5% of the full joint range at each end. It uses
processed targets after the environment's soft-limit clamp.

### Leg-permutation symmetry

Map the measured joint angles into the X1 logical convention:

$$
\overline{\mathbf q}_\ell=\mathbf s_\ell\odot\mathbf q_\ell,
\qquad \mathbf s_{FL}=\mathbf s_{FR}=(1,1,1),
\qquad \mathbf s_{RL}=\mathbf s_{RR}=(1,-1,-1).
$$

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

The penalty is zero when no pair is synchronous. It uses measured joint angles
without subtracting the default pose, and remains active with `--no-trs`.

## Disabled terms

The following terms are set to `None`: `lin_vel_z_l2`, `ang_vel_xy_l2`,
`dof_torques_l2`, `dof_acc_l2`, `feet_air_time`, `flat_orientation_l2`,
`dof_pos_limits`, `undesired_contacts`, `cmd`, `track_lin_vel_x_exp`,
`track_lin_vel_y_exp`, `sagittal_plane`, and `straight_line_motion`.

## Sources and other formats

- [Training launcher](train.sh) and [CLI options](scripts/symm_locomotion/symm_cli.py).
- [X1 reward binding](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/flat_env_cfg.py).
- [Shared reward weights and parameters](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/flat_env_cfg.py).
- [Shared reward functions](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symm_quadruped.py).
- [Standard planar tracking reward](source/isaaclab/isaaclab/envs/mdp/rewards.py).
- [X1 joint-axis signs](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/dobot_x1_symm.py).
- [Final reward composition](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/env.py) and [reward manager](source/isaaclab/isaaclab/managers/reward_manager.py).
- [Inherited configuration disabling undesired contacts](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py).
- Earlier **2026-10-05 snapshot**: [LaTeX source](logs/reward_documentation/train_rewards_2026-10-05.tex) and [compiled PDF](logs/reward_documentation/train_rewards_2026-10-05.pdf). These retain the earlier force coefficient; the current configuration is documented above.
