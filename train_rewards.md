<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Reward functions used by `train.sh`

Snapshot of the current workspace on **2026-09-29**. These equations describe the
configuration selected by the root [train.sh](train.sh), including local changes.
They are derived from the source; no training run is needed to read this document.

Both training commands select `--robot x1`, which resolves to
`Isaac-Velocity-Flat-Dobot-X1-Symm-v0`. They use **the same 11 nonzero reward terms**.
The second command's `--no-trs` flag disables symmetry data augmentation and
policy/value symmetry losses in the learning algorithm. It does **not** disable
the environment's `leg_permutation_symmetry` reward. The observation-history
override of 20 frames does not change the rewards.

## Reward weights and final composition

Each equation below defines the **raw function output** $f_i$, before the
configured weight $w_i$ and control-step duration are applied. All configured
weights below are positive; penalty functions themselves return nonpositive values.

| Configuration term | Raw output | Weight $w_i$ | Implementation |
| --- | --- | ---: | --- |
| `track_lin_vel_xy_exp` | $f_{xy}$ | 0.50 | `base_mdp.track_lin_vel_xy_exp` |
| `track_ang_vel_z_exp` | $f_{yaw}$ | 0.50 | `symm_mdp.track_ang_vel_z_exp` |
| `base_roll_exp` | $f_{roll}$ | 0.30 | `base_roll_exp_penalty` |
| `foot_periodicity` | $f_{period}$ | 0.30 | `foot_periodicity_penalty` |
| `base_height` | $f_{height}$ | 0.30 | `base_height_range_penalty` |
| `foot_clearance` | $f_{clear}$ | 0.10 | `foot_clearance_penalty` |
| `hip_action_penalty` | $f_{hip}$ | 0.15 | `hip_action_penalty` |
| `joint_target_limits` | $f_{limit}$ | 0.05 | `joint_position_target_limit_penalty` |
| `action_rate_l2` | $f_{rate}$ | 0.05 | `action_rate_exp_penalty` |
| `leg_permutation_symmetry` | $f_{sym}$ | 0.20 | `leg_permutation_symmetry_penalty` |
| `smoothness` | $f_{smooth}$ | 0.10 | `SmoothnessPenalty` |

The physics step is $0.005\,\mathrm{s}$ and decimation is 4, giving
$\Delta t=0.02\,\mathrm{s}$ (50 control steps per second). The
[reward manager][reward-manager] first computes each weighted contribution:

$$
r_i=\Delta t\,w_i f_i.
$$

The [environment][env] then applies its Ji22 reward composition. With $\mathcal I$
denoting all configured terms except `termination_penalty`:

$$
P=\sum_{i\in\mathcal I}\max(r_i,0),\qquad
N=\sum_{i\in\mathcal I}\min(r_i,0),\qquad
R=P\exp\!\left(\frac{N}{0.02}\right)+r_{term}.
$$

The termination weight is currently zero, so $r_{term}=0$. Define the weighted
penalty sum before multiplication by $\Delta t$:

$$
\begin{aligned}
B={}&0.30 f_{roll}+0.30 f_{period}+0.30 f_{height}
     +0.10 f_{clear}+0.15 f_{hip}\\
   &+0.05 f_{limit}+0.05 f_{rate}+0.20 f_{sym}+0.10 f_{smooth}.
\end{aligned}
$$

For the current $\Delta t=0.02$ and Ji22 scale $0.02$, the delivered reward is thus:

$$
\boxed{R=0.02\left(0.50 f_{xy}+0.50 f_{yaw}\right)\exp(B).}
$$

The two tracking terms provide positive reward; all other active terms reduce
it through the exponential multiplier. The previous rule that clipped the
additive running reward at zero is commented out in the current source.
`Episode_Reward/<term>` logs accumulate individual weighted terms before this
composition, so summing those logs does not reconstruct the delivered reward.

## Notation

All equations refer to one environment at one control step $t$. The time index
is omitted except where action or torque history is needed.

| Symbol | Meaning |
| --- | --- |
| $(v_x^*,v_y^*,\omega_z^*)$ | Commanded body-frame planar velocity [m/s] and yaw rate [rad/s] |
| $(v_x,v_y,\omega_z)$ | Measured body-frame root velocity [m/s] and yaw rate [rad/s] |
| $z,\rho,\eta$ | Root world height [m], roll [rad], and pitch [rad] |
| $a_{t,j}$ | Dimensionless action stored in the action manager; 12 joints in FL, FR, RL, RR leg order |
| $q_j,\widehat q_j$ | Measured joint position and processed, clipped joint-position target [rad] |
| $\tau_{t,j}$ | Applied joint torque at the reward evaluation [N m] |
| $\phi_\ell,d,\theta_\ell$ | Foot phase [cycles], stance duty factor, and commanded foot phase offset [cycles] |
| $\ell$ | Foot/leg in $\{\mathrm{FL},\mathrm{FR},\mathrm{RL},\mathrm{RR}\}$ |

Use $\operatorname{clip}(x,l,u)=\min(\max(x,l),u)$,
$[x]_+=\max(x,0)$, and $\operatorname{sigmoid}(x)=(1+e^{-x})^{-1}$.
Numerical constants assume the units listed above. Sums over $j$ cover all 12
joints unless otherwise specified. Within each leg, joint order is hip abduction,
thigh pitch, calf pitch.

## 1. Planar velocity tracking

Term: `track_lin_vel_xy_exp`; weight **0.50**.

$$
f_{xy}=\exp\!\left(-\frac{(v_x-v_x^*)^2+(v_y-v_y^*)^2}{0.2^2}\right).
$$

The configured `std` is $0.2\,\mathrm{m/s}$. Both axes contribute to a single
exponential. Source: [standard reward functions][base-rewards].

## 2. Yaw-rate tracking

Term: `track_ang_vel_z_exp`; weight **0.50**.

$$
f_{yaw}=\exp\!\left(-\left(\frac{\omega_z-\omega_z^*}{0.20}\right)^2\right).
$$

The configured `error_scale` is $0.20\,\mathrm{rad/s}$. This uses body-frame
angular velocity. Source: `track_ang_vel_z_exp` in [shared rewards][shared-mdp].

## 3. Base-roll penalty

Term: `base_roll_exp`; weight **0.30**.

$$
f_{roll}=\exp\!\left(-\frac{|\rho|}{0.25}\right)-1.
$$

The roll scale is $0.25\,\mathrm{rad}$. This is an absolute-error penalty; pitch
does not enter this term. Source: `base_roll_exp_penalty` in [shared rewards][shared-mdp].

## 4. Foot-periodicity penalty

Term: `foot_periodicity`; weight **0.30**.

Let $F_\ell$ be the maximum magnitude of the foot's net world-frame contact force
over the contact sensor's three stored samples [N], and let $u_\ell$ be the
magnitude of the foot's full world-frame linear velocity [m/s]:

$$
F_\ell=\max_{h\in\mathcal H}\|\mathbf F^w_{\ell,h}\|_2,\qquad
u_\ell=\|\mathbf v^w_\ell\|_2,\qquad |\mathcal H|=3.
$$

With the phase weights defined below, the raw penalty is:

$$
f_{period}=-\sum_\ell\left[
S_\ell\left(1-e^{-0.005 F_\ell}\right)
+T_\ell\left(1-e^{-2u_\ell}\right)
\right].
$$

It penalizes contact force during swing and foot speed during stance. The
coefficients are $0.005\,\mathrm{N}^{-1}$ and $2\,\mathrm{s/m}$, respectively.
The implementation sums over feet, without dividing by four.

The foot phases use a shared signed clock $\psi$ and signed offsets $o_\ell$:

$$
\psi_{t+1}=\operatorname{remainder}\!\left(\psi_t+s_t\frac{\Delta t}{T_t},1\right),
\qquad
\phi_{\ell,t}=\operatorname{remainder}(\psi_t+o_{\ell,t},1).
$$

The direction $s_t$ is the sign of the first command above its deadband, in
priority order: $v_x$, $v_y$, then $\omega_z$. Both linear deadbands are
$0.05\,\mathrm{m/s}$; the yaw deadband is $0.05\,\mathrm{rad/s}$. At or below all
deadbands, the previous direction is retained. Each environment starts with
direction $+1$ on reset before applying its new command.

The offset target is $o_\ell^{\mathrm{target}}=s_t\theta_\ell$. Direction changes
use the existing offset transition, preserving the current clock phase.
Period and duty-factor sampling still depend on forward speed.

The exact smoothed phase weights use the CDF $C_\kappa$ of
`scipy.stats.vonmises_line` with location zero and concentration $\kappa=16$.
For an interval $[a,b]$ expressed in cycles, define:

$$
V(\phi;a,b)=\sum_{m\in\{-1,0,1\}}
C_{16}\!\left(2\pi(\phi-a-m)\right)
\left[1-C_{16}\!\left(2\pi(\phi-b-m)\right)\right].
$$

$$
S_\ell=V(\phi_\ell;0,1-d),\qquad
T_\ell=V(\phi_\ell;1-d,1).
$$

These are the implemented products of CDF values, including the neighboring
cycles. `periodic_force_weights()` returns $-S_\ell$ and
`periodic_speed_weights()` returns $-T_\ell$. The duty factor is updated by the
gait command; the fallback value of 0.45 is not a constant throughout training.
Source: `foot_periodicity_penalty`, `_collect_single_body_contact_force_norms`,
and `_von_mises_phase_indicator` in [shared rewards][shared-mdp].

## 5. Base-height penalty

Term: `base_height`; weight **0.30**.

$$
f_{height}=\exp\!\left(-20\,|0.35-z|\right)-1.
$$

The target is **0.35 m**, with coefficient $20\,\mathrm{m}^{-1}$. Despite the
function name `base_height_range_penalty`, its `height_range` argument is
currently unused. The command's nominal height range of $(0.45,0.60)$ m is used
for gait timing, not as the height-reward target. Source: [shared configuration][reward-config]
and `base_height_range_penalty` in [shared rewards][shared-mdp].

## 6. Swing foot-clearance penalty

Term: `foot_clearance`; weight **0.10**; mode: `phase_penalty`.

Let $h_\ell=z^w_{foot,\ell}-z^w_{env\ origin}$ be the foot-link height above the
flat ground [m]. The swing trajectory and shortfall are:

$$
s=\max(1-d,10^{-6}),\qquad
\xi_\ell=\operatorname{clip}\!\left(\frac{\phi_\ell}{s},0,1\right),\qquad
h_\ell^*=0.10\sin(\pi\xi_\ell),\qquad
\delta h_\ell=[h_\ell^*-h_\ell]_+.
$$

The soft swing indicator and command gate are:

$$
W_\ell=\operatorname{sigmoid}\!\left(16\left[(1-d)-\phi_\ell\right]\right),
$$

$$
G_{cmd}=\operatorname{clip}\!\left(
\max\!\left(\frac{\sqrt{(v_x^*)^2+(v_y^*)^2}}{0.20},
\frac{|\omega_z^*|}{0.20}\right),0,1\right).
$$

The two command scales are $0.20\,\mathrm{m/s}$ and $0.20\,\mathrm{rad/s}$.
Translation or turning in place can fully enable the penalty:

$$
f_{clear}=-G_{cmd}\sum_\ell W_\ell
\left(1-\exp\!\left[-\frac{\delta h_\ell}{0.025}\right]\right).
$$

The peak target is 0.10 m, and shortfall scale is 0.025 m. Heights above the
trajectory are not penalized by this term. It sums over feet, without averaging.
The sigmoid swing weight here differs from the Von Mises weights in periodicity.
Source: `foot_clearance_penalty` in [shared rewards][shared-mdp].

## 7. Hip-action penalty

Term: `hip_action_penalty`; weight **0.15**.

For zero-based hip action indices $\mathcal A=\{0,3,6,9\}$, define:

$$
\alpha_j=\frac{e^{|a_{t,j}|/0.5}}{\sum_{k\in\mathcal A}e^{|a_{t,k}|/0.5}},
\qquad H=\sum_{j\in\mathcal A}\alpha_j|a_{t,j}|.
$$

$$
f_{hip}=e^{-0.5H}-1.
$$

This uses the dimensionless actions stored in the action manager. It does not
use measured hip angles or the scaled joint targets. Larger hip actions receive
more weight through the softmax. Source: `hip_action_penalty` in [shared rewards][shared-mdp].

## 8. Joint-target limit penalty

Term: `joint_target_limits`; weight **0.05**.

Let $L_j,U_j$ be the current soft joint-position limits [rad], and
$\widehat q_j$ the processed joint target after action processing and the
environment's soft-limit clamp. With $\epsilon$ the machine epsilon of the
limit tensor's dtype:

$$
x_j=\left|\frac{2(\widehat q_j-L_j)}{\max(U_j-L_j,\epsilon)}-1\right|,
\qquad
b_j=\operatorname{clip}\!\left(\frac{x_j-0.90}{0.10},0,1\right).
$$

$$
f_{limit}=-\frac{1}{12}\sum_{j=1}^{12}b_j^2.
$$

The configured margin is 5% of the full joint range at each end. Targets in
the middle 90% incur no penalty; a target at a limit has $b_j=1$. Joint targets
start from the default pose plus $0.25a_t$ and are then clipped. The reward
uses these resulting targets, not the measured joint positions.
Source: `joint_position_target_limit_penalty` in [shared rewards][shared-mdp]
and `_clamp_processed_joint_position_targets` in the [environment][env].

## 9. Action-change penalty

Term: `action_rate_l2`; weight **0.05**.

The stability gate shared with torque smoothness is:

$$
G_{stable}=\operatorname{clip}\!\left(
\operatorname{sigmoid}\!\left(20(z-0.25)\right)
\exp\!\left[-2(\rho^2+\eta^2)\right],0,1\right).
$$

Here the height threshold is 0.25 m, the height coefficient is
$20\,\mathrm{m}^{-1}$, and the orientation coefficient is
$2\,\mathrm{rad}^{-2}$. Define:

$$
D_a=\frac{1}{12}\sum_{j=1}^{12}(a_{t,j}-a_{t-1,j})^2.
$$

$$
f_{rate}=-G_{stable}\left(1-e^{-0.10D_a}\right).
$$

Although the configuration key is `action_rate_l2`, the selected implementation
is the bounded exponential `action_rate_exp_penalty`. It uses a mean squared
action difference and does not divide by $\Delta t$. The stability gate is
enabled by default. Source: [shared rewards][shared-mdp].

## 10. Leg-permutation symmetry penalty

Term: `leg_permutation_symmetry`; weight **0.20** in both training commands.

First map the measured joint angles into the X1 logical convention, using
elementwise multiplication $\odot$:

$$
\overline{\mathbf q}_\ell=\mathbf s_\ell\odot\mathbf q_\ell,\qquad
\mathbf s_{FL}=\mathbf s_{FR}=(1,1,1),\qquad
\mathbf s_{RL}=\mathbf s_{RR}=(1,-1,-1).
$$

The six candidate pairs are:

$$
\mathcal P=\{(FL,FR),(RL,RR),(FL,RL),(FR,RR),(FL,RR),(RL,FR)\}.
$$

A pair is active only when its commanded phase offsets are within 0.02 cycles
on the circle:

$$
d_{ab}=\min\!\left((\theta_a-\theta_b)\bmod 1,
(\theta_b-\theta_a)\bmod 1\right),\qquad
I_{ab}=\mathbf 1[d_{ab}\leq0.02].
$$

Let $\mathbf m_{ab}=(-1,1,1)$ for pairs on opposite left/right sides and
$(1,1,1)$ for pairs on the same side. For joint type $k\in\{1,2,3\}$:

$$
\mathbf Q=(1.3264,5.236,5.06)\;\mathrm{rad},\qquad
e_{ab,k}=\frac{|\overline q_{a,k}-m_{ab,k}\overline q_{b,k}|}{Q_k+10^{-6}},
$$

$$
\beta_{ab,k}=\frac{e^{e_{ab,k}/0.5}}{\sum_{n=1}^{3}e^{e_{ab,n}/0.5}},\qquad
E_{ab}=\sum_{k=1}^{3}\beta_{ab,k}e_{ab,k},\qquad
\overline E=\frac{\sum_{(a,b)\in\mathcal P}I_{ab}E_{ab}}
{\max\!\left(1,\sum_{(a,b)\in\mathcal P}I_{ab}\right)}.
$$

$$
f_{sym}=e^{-5\overline E}-1.
$$

If no pair is synchronous, the penalty is zero. These are absolute measured
joint angles, without subtracting the default pose. The normalization ranges
are fixed reward scales, not dynamically read joint limits.
Source: `leg_permutation_symmetry_penalty` in [shared rewards][shared-mdp],
with signs from the [X1 adapter][x1-mdp].

## 11. Torque-smoothness penalty

Term: `smoothness`; weight **0.10**.

$$
D_\tau=
\begin{cases}
0, & \text{first reward evaluation after reset},\\
\displaystyle\sum_{j=1}^{12}|\tau_{t,j}-\tau_{t-1,j}|, & \text{otherwise}.
\end{cases}
$$

$$
f_{smooth}=-G_{stable}\left(1-e^{-0.10D_\tau}\right).
$$

This is a sum of absolute applied-torque changes between reward evaluations,
with scale $0.10\,(\mathrm{N\,m})^{-1}$. It is not an absolute-torque cost or a
torque derivative divided by $\Delta t$. The previous-torque state is reset per
environment, and the stability gate from term 9 is enabled.
Source: `SmoothnessPenalty` in [shared rewards][shared-mdp].

## Configured terms with zero weight and disabled terms

Two additional terms are configured with zero weight and skipped by the reward manager:

$$
f_{alive}=1,\qquad w_{alive}=0,\qquad
f_{term}=\mathbf 1[\text{non-timeout termination}],\qquad w_{term}=0.
$$

These correspond to `alive_bonus` and `termination_penalty`. Episode termination
conditions remain active even though the explicit termination reward is zero.

The following reward terms are set to `None`: `lin_vel_z_l2`, `ang_vel_xy_l2`,
`dof_torques_l2`, `dof_acc_l2`, `feet_air_time`, `flat_orientation_l2`,
`dof_pos_limits`, `undesired_contacts`, `cmd`, `track_lin_vel_x_exp`,
`track_lin_vel_y_exp`, `sagittal_plane`, and `straight_line_motion`.
`undesired_contacts` is disabled in the inherited [Go2 rough configuration][go2-rough];
the others are disabled in the [shared reward configuration][reward-config].

## Source trace

1. [Root launcher](train.sh) selects `--robot x1` for both commands.
2. [Convenience CLI][cli] maps X1 to the task ID and handles `--no-trs`.
3. [X1 registration][x1-registration] selects `DobotX1SymmFlatEnvCfg` and
   `DobotX1SymmManagerBasedRLEnv`.
4. [X1 environment configuration][x1-config] calls `configure_rewards` with X1
   joints, foot sensors, and logical joint signs, using the shared numerical defaults.
5. [Shared reward configuration][reward-config] supplies weights and parameters.
6. [Shared reward functions][shared-mdp] and [standard reward functions][base-rewards]
   implement the raw equations.
7. [Base locomotion configuration][velocity-config] sets physics timing; the
   [reward manager][reward-manager] applies weights and $\Delta t$.
8. [X1 environment class][x1-env] inherits the [shared environment][env], which
   delivers the Ji22-composed reward to the learner.

[cli]: scripts/symm_locomotion/symm_cli.py
[x1-registration]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/__init__.py
[x1-config]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/flat_env_cfg.py
[x1-env]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/env.py
[x1-mdp]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/dobot_x1_symm.py
[reward-config]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/flat_env_cfg.py
[shared-mdp]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symm_quadruped.py
[base-rewards]: source/isaaclab/isaaclab/envs/mdp/rewards.py
[env]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/env.py
[reward-manager]: source/isaaclab/isaaclab/managers/reward_manager.py
[velocity-config]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
[go2-rough]: source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py
