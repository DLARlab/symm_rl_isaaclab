# Milestone 5: Actor-only TRS V5

- Evidence collected: 2026-09-02 through 2026-09-05
- Documented: 2026-09-05
- Publication branch: `jding/72d_actor_trs_v5`
- Milestone-4 base: `da361752654938efeea526098f30a008dd7320b5`
- Baseline: [Milestone 4: gait-closure parameter V4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md), documented
  2026-09-01
- Status: actor-only TRS default implemented and single-seed coefficient/ramp development screen complete; the
  matched actor/value component ablation and confirmatory multi-seed evaluation remain open

Timeline:
[Milestone 1](MILESTONE_1_60D_TO_72D.md) ·
[Milestone 2](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md) ·
[Milestone 3](MILESTONE_3_GAIT_FAMILY_V2.md) ·
[Milestone 4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md) · **Milestone 5**

## Scope and decision boundary

This milestone audits every change from the Milestone-4 publication commit, separates the ordinary PPO critic from
the auxiliary time-reversal (TR) critic-consistency hypothesis, and reports the completed actor-only coefficient and
ramp screen for both robots. The principal method changes from

$$
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+
\lambda_\pi^{\mathrm{TR}}\mathcal L_\pi^{\mathrm{TR}}
+
\lambda_V^{\mathrm{TR}}\mathcal L_V^{\mathrm{TR}},
\qquad
\lambda_V^{\mathrm{TR}}=0.05,
$$

to

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+
\lambda_\pi^{\mathrm{TR}}\mathcal L_\pi^{\mathrm{TR}},
\qquad
\lambda_V^{\mathrm{TR}}=0.
}
$$

The ordinary clipped PPO objective, generalized advantage estimation, returns, critic network, critic optimizer,
value normalization, adaptive-KL behavior, and standard value regression are outside this change. In particular,

$$
\mathcal L_{\mathrm{PPO},V}
=
\mathbb E
\left[
\left(
V_\varphi(o_t)-\widehat G_t
\right)^2
\right]
$$

remains part of $\mathcal L_{\mathrm{PPO}}$. Only the auxiliary loss

$$
\mathcal L_V^{\mathrm{TR}}
=
\mathbb E
\left[
m(o_t)
\left(
V_\varphi(\mathcal T_Oo_t)
-
\operatorname{sg}[V_\varphi(o_t)]
\right)^2
\right]
$$

receives a default coefficient of zero. Its implementation, schedules, logging, diagnostics, and checkpoint fields
remain available as the **optional time-reversal critic-consistency ablation**. The audit does not revise any
configuration or numerical claim reported in Milestones 1–4; those documents describe the runs that were actually
performed.

## Terminology

1. **Standard PPO value regression** trains $V_\varphi(o_t)$ toward the on-policy forward return target
   $\widehat G_t$. It is the ordinary critic component of PPO and is controlled separately from every TR term.
2. **TR policy consistency** is a pointwise actor regularizer. It compares the transformed action prescribed at
   $o_t$ with the policy output at the algebraically transformed observation $\mathcal T_Oo_t$, subject to the TR
   validity mask and coefficient schedule. It does not add reversed transitions to the rollout buffer.
3. **TR critic consistency** is the optional auxiliary penalty $\mathcal L_V^{\mathrm{TR}}$ above. It equates two
   forward-critic predictions after an observation transform; it is not standard PPO value regression and does not
   by itself construct a backward return target.
4. **Reverse-transition augmentation** maps a recorded transition to a candidate reverse transition, for example
   $(s_t,a_t,r_t,s_{t+1})$ to
   $(\mathcal T s_{t+1},\mathcal T_Aa_t,r_t^{\mathrm{rev}},\mathcal T s_t)$, and adds that sample to a learning
   dataset or replay buffer. This is distinct from both pointwise consistency losses. TR minibatch augmentation is
   disabled in the experiments proposed below.
5. **Forward-looking value** is the expected discounted reward accumulated from the present toward increasing time.
   It is the quantity estimated by the standard PPO critic.
6. **Backward-looking value** is the expected discounted reward sequence obtained by indexing the same trajectory
   toward decreasing time. It is a mathematical comparison quantity, not the target used by ordinary PPO.
7. **Complete-cycle return** is a finite sum containing each reward in one periodic cycle exactly once. Without
   discount weights, its value is unchanged when cycle orientation is reversed.

**Setting the TR critic-consistency coefficient to zero does not remove the PPO critic and does not turn PPO into an
actor-only algorithm.** “Actor only” in this document always abbreviates “TR policy consistency enabled and TR
critic consistency disabled”; the standard PPO actor and critic both continue to train.

## Reversing versus commuting symmetry

Let $F_c$ denote one forward dynamics step under command $c$. First consider an ordinary, forward-time symmetry $S$,
with $S_c$ denoting its action on commands:

$$
S F_c=F_{S_c c}S.
$$

Iterating this relation gives

$$
F_{S_c c}^{\,k}(S s_t)
=
S F_c^{\,k}(s_t)
=
S s_{t+k}.
$$

Thus $S$ maps the future of one state to the transformed future of the other state. If the policy, transition law,
initial distribution, and reward are also compatible with $S$, the two forward reward sequences match. Under those
additional assumptions, conventional forward-value invariance can follow:

$$
V_{S_c c}^{+}(S s_t)=V_c^{+}(s_t).
$$

Time reversal is not a commuting symmetry of this form. A reversing symmetry $\mathcal T$ instead satisfies

$$
\mathcal T F_c=F_{-c}^{-1}\mathcal T.
$$

Multiplying on the left by $F_{-c}$ and on the right by $F_c^{-1}$ yields the equivalent one-step relation

$$
F_{-c}\mathcal T=\mathcal T F_c^{-1}.
$$

Applying it to $s_t$ gives

$$
F_{-c}(\mathcal T s_t)
=
\mathcal T(F_c^{-1}s_t)
=
\mathcal T s_{t-1}.
$$

Suppose for induction that $F_{-c}^{\,k}(\mathcal T s_t)=\mathcal T s_{t-k}$. Then

$$
\begin{aligned}
F_{-c}^{\,k+1}(\mathcal T s_t)
&=F_{-c}\!\left(F_{-c}^{\,k}(\mathcal T s_t)\right) \\
&=F_{-c}(\mathcal T s_{t-k}) \\
&=\mathcal T s_{t-k-1}.
\end{aligned}
$$

Therefore, for every nonnegative integer $k$,

$$
\boxed{
F_{-c}^{\,k}(\mathcal T s_t)
=
\mathcal T s_{t-k}.
}
$$

In words: **the forward future of the time-reversed state follows the transformed past of the original trajectory.**
This reversal of temporal orientation is exactly why the usual argument for a commuting symmetry does not establish
equality between the two forward-looking values.

## Correct forward/backward value relation

For $0\leq\gamma<1$, define the forward-looking command-conditioned value

$$
V_c^+(s_t)
=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_c(s_{t+k})
\right],
$$

and define the corresponding backward-looking value along the matched trajectory by

$$
V_c^-(s_t)
=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_c(s_{t-k})
\right].
$$

For stochastic dynamics, the expectation in the second definition is over the matched reverse trajectory law; the
deterministic trajectory notation below is the pathwise special case. Assume pointwise reward compatibility,

$$
r_{-c}(\mathcal T s)=r_c(s).
$$

Using the reversing relation derived above,

$$
\begin{aligned}
V_{-c}^+(\mathcal T s_t)
&=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k
r_{-c}\!\left(F_{-c}^{\,k}(\mathcal T s_t)\right)
\right] \\
&=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k
r_{-c}(\mathcal T s_{t-k})
\right] \\
&=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_c(s_{t-k})
\right] \\
&=V_c^-(s_t).
\end{aligned}
$$

Hence the value identity supported by time reversal is

$$
\boxed{
V_{-c}^+(\mathcal T s_t)
=
V_c^-(s_t).
}
$$

Time reversal alone does **not** imply

$$
V_{-c}^+(\mathcal T s_t)
=
V_c^+(s_t).
$$

That stronger equality requires the original trajectory's discounted future and discounted past to have equal
expected reward. It may happen because of additional stationarity, reward, phase, or cycle structure, but it is not
a consequence of reversing dynamics. Moreover, the implemented auxiliary loss compares learned forward-critic
predictions; it does not supervise $V_c^-$ from trajectory-derived past-return targets.

## Periodic-orbit derivation

Consider a deterministic reward sequence with period $N$, so $r_{t+N}=r_t$, and again let
$0\leq\gamma<1$. Split the forward infinite sum into complete periods, with $k=qN+j$:

$$
\begin{aligned}
V_t^+
&=\sum_{k=0}^{\infty}\gamma^k r_{t+k} \\
&=\sum_{q=0}^{\infty}\sum_{j=0}^{N-1}
\gamma^{qN+j}r_{t+j} \\
&=\left(\sum_{q=0}^{\infty}\gamma^{qN}\right)
\left(\sum_{j=0}^{N-1}\gamma^j r_{t+j}\right) \\
&=\boxed{
\frac{
\sum_{j=0}^{N-1}\gamma^j r_{t+j}
}{
1-\gamma^N
}}.
\end{aligned}
$$

The same decomposition in the opposite orientation gives

$$
\begin{aligned}
V_t^-
&=\sum_{k=0}^{\infty}\gamma^k r_{t-k} \\
&=\sum_{q=0}^{\infty}\sum_{j=0}^{N-1}
\gamma^{qN+j}r_{t-j} \\
&=\boxed{
\frac{
\sum_{j=0}^{N-1}\gamma^j r_{t-j}
}{
1-\gamma^N
}}.
\end{aligned}
$$

Because the denominators agree, $V_t^+=V_t^-$ exactly when

$$
\boxed{
\sum_{k=0}^{N-1}
\gamma^k
\left(
r_{t+k}-r_{t-k}
\right)=0.
}
$$

This condition holds for a constant reward and may hold approximately when reward is nearly phase independent. It
can also hold accidentally at individual phases. It is not guaranteed by the gait's time-reversing structure: for a
non-palindromic phase-dependent reward sequence, the same rewards receive different discount weights in the two
orientations.

This discounted value must be distinguished from the undiscounted complete-cycle sum

$$
J_{\mathrm{cycle}}
=
\sum_{k=0}^{N-1}r_{t+k}.
$$

Modulo the period, $\{t+k:0\leq k<N\}$ and $\{t-k:0\leq k<N\}$ enumerate the same phases. Therefore

$$
\sum_{k=0}^{N-1}r_{t+k}
=
\sum_{k=0}^{N-1}r_{t-k}.
$$

The undiscounted complete-cycle sum is orientation independent because reversal merely reorders a finite set of
rewards. Standard PPO instead learns a discounted, forward-looking value, so complete-cycle permutation invariance
does not establish the auxiliary critic equality.

## Fixed points of the phase map

The duty-aware phase reflection is

$$
\mathcal R_\beta(\phi)
=
\operatorname{wrap}(1-\beta-\phi),
$$

where `wrap` maps phase to $[0,1)$. A fixed point satisfies

$$
\phi\equiv 1-\beta-\phi\pmod 1,
$$

or

$$
2\phi\equiv 1-\beta\pmod 1.
$$

The two solutions modulo one are

$$
\boxed{
\phi_1=\frac{1-\beta}{2},
\qquad
\phi_2=\frac{2-\beta}{2}\pmod1.
}
$$

These are fixed points of the phase reflection itself. They are not the only phases where two scalar value
predictions could happen to be equal. Equality can also arise from constant or locally symmetric rewards,
phase-aliased observations, function-approximation error, averaging, or accidental cancellation in the discounted
sum. Conversely, a phase fixed point does not repair violations in the rest of the physical or observation state.

## Previous-action channel limitation

The 72-dimensional actor and critic observation documented in Milestone 1 contains a 12-dimensional previous-action
channel. For the forward trajectory

$$
s_{t-1}
\xrightarrow{a_{t-1}}
s_t
\xrightarrow{a_t}
s_{t+1},
$$

the original observation at $s_t$ contains $a_{t-1}$. In the reversed trajectory, however,

$$
\mathcal T s_{t+1}
\xrightarrow{\mathcal T_Aa_t}
\mathcal T s_t,
$$

so the previous action in a trajectory-consistent observation at $\mathcal T s_t$ should be
$\mathcal T_Aa_t$, not $\mathcal T_Aa_{t-1}$. A pointwise transformation of $o_t$ only has access to the stored
$a_{t-1}$ channel and cannot reconstruct $a_t$ exactly. This remains a mismatch even when $\mathcal T_A$ is the
identity for the chosen joint-position-offset action representation, because generally $a_t\ne a_{t-1}$.

Consequently, the transformed 72D observation can be an exact algebraic involution,
$\mathcal T_O^2o=o$, while not being a trajectory-consistent Markov observation of the reversed process. This limits
both actor and critic interpretations of pointwise TR consistency and motivates reporting the actor regularizer as a
structural prior rather than a theorem about the physical trajectory.

## Physical and task asymmetry

Exact temporal reversibility is broken or weakened by the following features of simulated and physical locomotion:

- **Impact:** inelastic foot-strike and collision impulses discard information and do not generally admit a feasible
  reversed contact transition.
- **Friction:** dissipative and stick-slip contact forces distinguish the two temporal orientations.
- **PD damping:** velocity-proportional damping removes mechanical energy in forward simulation.
- **Actuator saturation:** clipping a requested effort is many-to-one and may make the inverse transition
  unavailable.
- **Target clipping:** clipping requested joint targets likewise loses the pre-clipping request and changes reachable
  reverse actions.
- **Termination:** falls, invalid contacts, and other terminal conditions truncate one orientation rather than
  producing a matched continuation.
- **Finite episodes:** reset distributions and horizon truncation make the available past and future contexts
  different.
- **Command resampling:** a reversed segment can cross a command boundary whose prior command is not recoverable from
  the current pointwise observation.
- **Reward terms that depend on future task success:** completion, survival, and terminal outcomes are inherently
  oriented toward the forward task horizon.
- **Phase-dependent reward:** discount weights attach to different phase rewards under reversal, as the periodic-orbit
  derivation shows.
- **Asymmetric morphology:** geometric, inertial, actuation, or joint-limit differences can invalidate a proposed leg
  or action transform.

These effects need not make a TR regularizer useless, but they prevent mechanical time reversal from serving as a
general proof of forward-critic invariance.

## Relation to prior work

[Barkley, Zhang, and Fridovich-Keil, *An Investigation of Time Reversal Symmetry in Reinforcement
Learning*](https://proceedings.mlr.press/v242/barkley24a.html) ([PMLR 242:68–79,
2024](https://proceedings.mlr.press/v242/barkley24a/barkley24a.pdf);
[arXiv:2311.17008](https://doi.org/10.48550/arXiv.2311.17008)) formalize dynamically action-reversible MDPs and use
time-symmetric data augmentation to turn observed transitions into counterfactual reverse transitions for off-policy,
model-free learning. Their analysis explicitly treats transformed samples as likely off-policy, identifies friction
and impulsive contact as violations of the ideal assumptions, and reports that augmentation can help or harm
depending on environment, reward, and actuator conditions.

[Jiang et al., *Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement
Learning*](https://arxiv.org/html/2505.13925) ([arXiv:2505.13925v2](https://doi.org/10.48550/arXiv.2505.13925))
study manipulation task pairs. Their TR-DRL framework learns reverse actions with inverse dynamics, checks proposed
reversals with a learned forward-dynamics consistency filter, augments learning with accepted reversed transitions,
and uses time-reversal-guided reward shaping for partially reversible transitions.

Thus the relevant prior work exploits reversed transitions, likely off-policy data, learned reverse actions,
dynamics-consistency filtering, and reward shaping. Those mechanisms reason about transition tuples or task-paired
trajectories and include checks or assumptions about whether reversal is feasible. Neither work provides a general
justification for imposing

$$
V(\mathcal T s)=V(s)
$$

on a standard forward-looking discounted critic. That equation is a separate empirical hypothesis and must be tested
as such in this locomotion setting.

## Configuration and computation contract

The default coefficient resolution and the canonical enable flag are deliberately separate. `None` preserves older
positive-coefficient configurations by inferring enablement from the resolved coefficient; an explicit boolean takes
precedence:

| `use_tr_value_consistency` | `value_loss_coeff` | Effective behavior |
| --- | ---: | --- |
| `None` | `0.0` | Disabled. |
| `None` | `0.05` | Enabled for legacy compatibility. |
| `False` | `0.05` | Disabled by the explicit canonical flag. |
| `True` | `0.0` | Enable flag is set but contribution is zero; emit a clear warning or diagnostic. |
| `True` | `0.05` | Enabled. |

The canonical field must not be permanently defaulted to `False`, because doing so would silently change historical
positive-coefficient configurations. Data augmentation remains independently disabled by default.

When the effective value coefficient is zero and `log_disabled_raw_consistency=false`, a PPO update must not evaluate
the critic on transformed observations, construct the auxiliary value graph, add an auxiliary value gradient, or
change critic optimizer behavior. When `log_disabled_raw_consistency=true`, a diagnostic may evaluate

$$
V_\varphi(\mathcal T_Oo_t)-V_\varphi(o_t),
$$

but that computation must be detached or performed under `torch.no_grad()` and must not enter the optimization loss.
An explicit positive coefficient must retain the prior stop-gradient loss exactly. These requirements keep the
disabled default free of measurable training memory or runtime overhead while preserving the optional ablation.

## Offline value-consistency diagnostic

Before spending full simulation budgets, recorded complete-cycle per-step rewards can directly test the premise that
discounted future and past returns are close. For a horizon $H$ within one complete cycle, compute

$$
G_{t,H}^+
=
\sum_{k=0}^{H-1}\gamma^k r_{t+k},
\qquad
G_{t,H}^-
=
\sum_{k=0}^{H-1}\gamma^k r_{t-k}.
$$

Windows must never wrap across different episodes, command intervals, or gait rows. The analysis should report:

- `future_return_mean`;
- `past_return_mean`;
- `future_past_return_rmse`;
- `future_past_return_normalized_rmse`;
- `future_past_return_correlation`;
- `future_past_return_gap_by_phase`;
- `future_past_return_gap_by_gait_family`;
- `future_past_return_gap_by_command_sign`;
- `future_past_return_gap_by_robot`.

If critic predictions were recorded, it should additionally report

$$
V(o_t)-G_{t,H}^+,
\qquad
V(\mathcal T_Oo_t)-G_{t,H}^-,
$$

and the quantity imposed by the optional auxiliary loss,

$$
V(\mathcal T_Oo_t)-V(o_t).
$$

The implemented diagnostic is
`scripts/symm_locomotion/analyze_tr_value_consistency.py`, with the direct PowerShell wrapper
`scripts/symm_locomotion/analyze_tr_value_consistency.ps1`. Its NPZ contract deliberately requires the unambiguous
field `step_rewards` and an explicit true `complete_cycle` marker. A two-dimensional reward array declares one
complete cycle per row and rejects any recorded episode, command-interval, or gait-row change within a row. A
one-dimensional stream additionally requires at least one `cycle_id`, `episode_id`, `command_interval_id`, or
`gait_row` boundary field. Missing phase is reported with `phase_source=inferred_uniform_index` and an arbitrary zero
phase, rather than presented as a recorded physical phase.

An archive containing only aggregate tables cannot support this calculation. The diagnostic fails clearly when its
per-step, completeness, or boundary evidence is insufficient; it does not fabricate trajectories from aggregate
statistics.

Synthetic checks should establish the numerical boundary before real data are interpreted:

1. constant reward gives $G_{t,H}^+=G_{t,H}^-$ at every phase for matched full-period windows;
2. a non-palindromic periodic reward generally gives $G_{t,H}^+\ne G_{t,H}^-$;
3. an undiscounted full-cycle sum is invariant to reversal;
4. a discounted full-cycle sequence need not be invariant;
5. equality of two scalar returns does not require a fixed state or a fixed phase.

## Audited implementation delta from Milestone 4

Milestone 4 is fixed at commit `da361752654938efeea526098f30a008dd7320b5`. There are no intervening commits;
the following source and publication changes are the complete Milestone-5 delta:

1. The CLI, shared symmetric-quadruped PPO builder, and the Go2 and X1 runner configurations change the auxiliary
   TR value-consistency coefficient from `0.05` to `0.0`. TR policy consistency remains enabled at its independently
   configured coefficient.
2. `use_tr_value_consistency=None` still infers enablement from a positive legacy coefficient. Explicit canonical
   flags take precedence, so archived positive-coefficient configurations remain reproducible and callers can still
   run critic-only or actor-plus-critic ablations.
3. The actor-only path no longer evaluates the critic on transformed observations. Optional disabled-term logging
   runs under `torch.no_grad()` and cannot alter the standard PPO critic gradient. An explicit value enable flag with
   a zero resolved coefficient emits a warning instead of silently suggesting an optimization contribution.
4. `analyze_tr_value_consistency.py` and its PowerShell wrapper add a strict offline future-versus-past return
   diagnostic. They require per-step rewards, complete-cycle evidence, and unambiguous boundaries, and record source
   hashes and stratified phase, gait-family, command-sign, and robot results.
5. The comparison validator now accepts a terminated or resumed cell only when the study is complete and every
   required metric domain has coverage. It validates status accounting, preserves cell status and termination fields
   in `evaluation_cells.csv`, and emits an outcome-free-aggregation warning.
6. Local-only Go2 registry entries were redirected from live training roots to their `logs/analysis` archive paths.
   This is data-location housekeeping, not a method change.
7. The publication archive advances from the five-run combined-loss V4 cohorts to the actor-only cohorts below.
   Raw legacy folders are no longer published; their scientific record remains summarized in Milestones 1–4.

Captured run provenance still names `jding/gait-closure-v4-archive` where that was the branch at training or
evaluation time. Those strings and archived source diffs are historical evidence and are intentionally not rewritten
as part of the publication-branch rename.

## Fixed experimental contract

- The standard PPO critic, value regression, generalized advantage estimation, optimizer, and normalization remain
  enabled in every condition.
- The 72-dimensional observation, 12-dimensional action, time-reversal-closed ten-gait library, equal-family gait
  sampling, and disabled gait curriculum remain unchanged.
- Each training run uses 20,000 iterations, 512 environments, and 24 steps per environment: 245,760,000 environment
  transitions.
- TR transition augmentation remains disabled. The actor-only treatments use TR policy consistency with value
  coefficient `0`, a 500-iteration warmup, and linear scheduling.
- X1 retains its V4 reward and action-feasibility profile and training seed 42. Go2 retains the V4 recovery profile:
  foot phase `0.3 * sum`, requested-overflow joint-limit weight `0.2`, per-joint feasible actor bounds with zero
  margin, normalized requested-joint-target policy space, command validity, and training seed 43.
- Evaluation uses the full-V3 10-gait by 6-speed Cartesian grid at seed 42, or 60 cells per run. The reused baseline
  folders also retain their separate Milestone-4 seed-4242 evaluation archives as provenance; those older metrics are
  not mixed into the V5 tables.

## Archived Actor TRS V5 cohorts

No new no-TRS training was required. Each robot reuses its configuration-matched Milestone-4 baseline checkpoint and
compares it with newly trained actor-only policies.

### Dobot X1, seed 42

| Label | Archived run | Policy/value | Warmup/ramp |
| --- | --- | ---: | ---: |
| NoTRS | `2026-08-25_06-08-02_notrs_x1def_s42` | `0/0` | inactive |
| Actor-m0.1 | `2026-09-03_00-15-11_m5_x1_actor_only_trs_m0p1_v0_w500_r0_x1def_s42` | `0.1/0` | `500/0` |
| Actor-m0.2 | `2026-09-03_11-19-54_m5_x1_actor_only_trs_m0p2_v0_w500_r0_x1def_s42` | `0.2/0` | `500/0` |
| Actor-m0.3 | `2026-09-03_23-50-10_m5_x1_actor_only_trs_m0p3_v0_w500_r0_x1def_s42` | `0.3/0` | `500/0` |

### Unitree Go2, seed 43

| Label | Archived run | Policy/value | Warmup/ramp |
| --- | --- | ---: | ---: |
| NoTRS | `2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0/0` | inactive |
| Actor-m0.1 | `2026-09-03_00-14-58_m5_go2_actor_only_trs_m0p1_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.1/0` | `500/0` |
| Actor-m0.2 | `2026-09-03_12-49-47_m5_go2_actor_only_trs_m0p2_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.2/0` | `500/0` |
| Actor-m0.3 | `2026-09-04_00-27-07_m5_go2_actor_only_trs_m0p3_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.3/0` | `500/0` |
| Actor-m0.1-r500 | `2026-09-04_23-04-00_m5_go2_actor_only_trs_m0p1_v0_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.1/0` | `500/500` |
| Actor-m0.2-r500 | `2026-09-04_23-04-13_m5_go2_actor_only_trs_m0p2_v0_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.2/0` | `500/500` |

Within each robot, initialization records show matching actor, critic, and optimizer hashes, and the comparison
validates the matched architecture, reward, command, randomization, optimizer, gait, and training-budget signatures.
The baseline and treatment runs were nevertheless produced from different source revisions or dirty-source states.
The results are therefore single-seed descriptive development evidence, not a strict causal estimate of removing the
TR critic-consistency term.

## Performance audit

`AUC` is mean-reward area over all 20,000 iterations and `Tail` is mean reward over the final 1,000 iterations. `vx`
and `yaw` are family-balanced RMSE in `m/s` and `rad/s`; `Gait` is boundary-excluded contact-pattern agreement. The
four load columns are family-balanced mean absolute front/hind imbalance percentages, not total effort or load.
Lower is better except for AUC, Tail, and Gait.

### Unitree Go2

| Run | AUC | Tail | vx | yaw | Gait | Torque^2 | Norm. torque^2 | Work | GRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoTRS | 32.534 | 40.528 | 0.03439 | **0.05904** | **88.803%** | 26.651% | 28.274% | 13.557% | 19.102% |
| Actor-m0.1 | **32.779** | **41.156** | 0.03900 | 0.06061 | 88.354% | 19.893% | 22.666% | 10.152% | 17.209% |
| Actor-m0.2 | 32.697 | 40.898 | 0.03916 | 0.05953 | 88.057% | 18.178% | 20.534% | 19.530% | 21.524% |
| Actor-m0.3 | 32.387 | 40.355 | **0.03335** | 0.06124 | 86.983% | 22.019% | 25.826% | 14.908% | 23.473% |
| Actor-m0.1-r500 | 32.562 | 40.366 | 0.03500 | 0.06158 | 87.972% | **16.796%** | **19.757%** | 10.874% | **16.399%** |
| Actor-m0.2-r500 | 32.549 | 40.993 | 0.04028 | 0.05907 | 88.101% | 19.817% | 23.060% | **7.576%** | 18.199% |

Actor-m0.1 has the strongest observed learning; Actor-m0.3 has the lowest forward-velocity RMSE. NoTRS retains the
best gait agreement and lowest yaw RMSE, although Actor-m0.2-r500 is nearly tied on yaw. The ramped m0.1 treatment
leads torque-squared, normalized-torque-squared, and GRF balance, while ramped m0.2 leads work balance. Every actor
treatment reduces gait agreement by 0.449–1.820 percentage points relative to NoTRS.

The matched ramp comparison is mixed:

- At actor coefficient `0.1`, r500 improves forward RMSE by `0.00399`, torque-squared imbalance by `3.097 pp`,
  normalized-torque-squared imbalance by `2.909 pp`, and GRF imbalance by `0.810 pp`. It lowers AUC by `0.217`, tail
  reward by `0.790`, and gait agreement by `0.382 pp`, while slightly worsening yaw and work balance.
- At actor coefficient `0.2`, r500 improves work by `11.955 pp`, GRF by `3.325 pp`, yaw by `0.00046`, gait agreement
  by `0.045 pp`, and tail reward by `0.095`. It worsens forward RMSE by `0.00112`, torque-squared balance by
  `1.639 pp`, normalized-torque-squared balance by `2.526 pp`, and AUC by `0.148`.

Ramp-up is therefore a trade-off rather than a uniformly superior schedule.

### Dobot X1

| Run | AUC | Tail | vx | yaw | Gait | Torque^2 | Norm. torque^2 | Work | GRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoTRS | 34.667 | 42.307 | **0.03136** | 0.03559 | **96.747%** | 17.687% | 26.149% | 23.917% | 6.999% |
| Actor-m0.1 | **35.652** | **42.729** | 0.03232 | **0.02910** | 95.575% | 7.764% | 16.743% | 13.028% | 4.748% |
| Actor-m0.2 | 32.918 | 42.066 | 0.03222 | 0.03747 | 95.626% | **5.148%** | 15.966% | **7.466%** | **3.668%** |
| Actor-m0.3 | 32.418 | 41.398 | 0.03163 | 0.03547 | 95.735% | 5.468% | **11.276%** | 12.849% | 4.024% |

Actor-m0.1 has the strongest observed learning and yaw tracking. NoTRS retains the best forward tracking and gait
agreement. Actor-m0.2 leads torque-squared, work, and GRF balance; Actor-m0.3 leads normalized-torque-squared balance.
Every actor-only treatment improves all four balance metrics versus NoTRS while reducing gait agreement by about
1.0–1.2 percentage points. This cohort contains no r500 X1 treatment, so it cannot answer the X1 ramp-up question.

### Descriptive relation to Milestone 4

The table below reports Actor TRS V5 minus the nominally matched Milestone-4 actor-plus-value treatment. Negative
RMSE or load deltas are improvements; positive gait deltas are improvements. Source-provenance differences and one
training seed per robot prevent interpreting any row as the isolated causal effect of the TR value term.

| Robot and treatment | AUC | Tail | vx | yaw | Gait | Torque^2 / norm. torque^2 / work / GRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Go2 m0.1-r0 | -0.272 | -0.034 | +0.00574 | -0.00090 | +1.782 pp | -2.790 / -3.754 / -2.483 / -7.810 pp |
| Go2 m0.2-r0 | +0.700 | +1.348 | +0.00335 | -0.00383 | +1.430 pp | -0.201 / -1.135 / +10.252 / +3.809 pp |
| Go2 m0.1-r500 | +1.001 | +1.502 | -0.00974 | -0.00075 | +0.534 pp | -7.524 / -7.599 / -2.082 / -5.650 pp |
| Go2 m0.2-r500 | -0.319 | +0.052 | +0.00625 | +0.00377 | +0.277 pp | -1.524 / -2.751 / -3.417 / -0.676 pp |
| X1 m0.1-r0 | -0.359 | -0.453 | -0.00146 | -0.00661 | -0.012 pp | +1.711 / +3.578 / -2.887 / +1.798 pp |
| X1 m0.2-r0 | -0.363 | -0.326 | +0.00031 | +0.00896 | +0.328 pp | -1.601 / +3.059 / -10.009 / +0.174 pp |

The changes do not support a blanket claim for or against the critic-consistency ablation. They do support the
narrower observation that actor-only TRS can improve selected learning or balance endpoints while tracking, gait,
reward, and different load measures continue to trade off.

## Evaluation completeness and integrity

- X1 contributes 240/240 metric-complete cells and Go2 contributes 360/360. Every run has tracking, gait, raw-load,
  normalized-load, and GRF coverage for all 60 requested cells.
- Go2 Actor-m0.1 has one deterministic, metric-complete termination at
  `gait_08_gallop__vx_neg_0p5__seed_0042`: `calf_height` at step 698/750. It retains 423 post-settle samples and 21
  complete gait cycles. The outcome-free aggregation includes those usable metrics and the report discloses the
  termination; all other cells are valid.
- The comparison validates terminal-checkpoint, TensorBoard event, initialization, and full-V3 input hashes. Both
  robot archives were regenerated with the same final comparison implementation, and all 45 declared output hashes
  per robot validate.

## Evidence, reproduction, and publication policy

The resolved manifests, provenance, tables, reports, and 16 figures per robot are in:

- [Dobot X1 Actor TRS V5 report](dobot_x1_symm_flat/actor_trs_v5_analysis/REPORT.md)
- [Unitree Go2 Actor TRS V5 report](unitree_go2_symm_flat/actor_trs_v5_analysis/REPORT.md)

Regenerate the committed outputs from the curated run roots:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\actor_trs_v5_analysis\reproduce.py
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\actor_trs_v5_analysis\reproduce.py
```

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs/dobot_x1_symm_flat/actor_trs_v5_analysis/reproduce.py
./isaaclab.sh -p logs/rsl_rl/good_runs/unitree_go2_symm_flat/actor_trs_v5_analysis/reproduce.py
```

Each of the ten current training directories publishes every non-checkpoint artifact, including TensorBoard events,
exports, source and command provenance, recordings, evaluation archives, and metrics. Only its terminal training
checkpoint, `model_19999.pt`, is published; 200 intermediate checkpoint files remain local and ignored. The two
`legacy/` trees also remain local and ignored, but no legacy path is present in the publication index. Their methods,
results, evidence boundaries, and relevant numerical conclusions remain documented in Milestones 1–4.

## Required four-condition experiment

Each robot requires a matched $2\times2$ component ablation:

| Condition | TR policy consistency | TR critic consistency |
| --- | :---: | :---: |
| PPO | No | No |
| Actor only | Yes | No |
| Critic only | No | Yes |
| Actor + critic | Yes | Yes |

For X1, the exploratory coefficient pairs $(\lambda_\pi^{\mathrm{TR}},\lambda_V^{\mathrm{TR}})$ are

$$
(0,0),\quad
(0.10,0),\quad
(0,0.05),\quad
(0.10,0.05).
$$

For Go2, the selected actor coefficient after the current action-feasibility and reward corrections is $0.10$, so the
same four coefficient pairs are used in the commands below. This fixes the current experimental candidate; it does
not claim that the actor-only condition will reproduce or exceed an archived combined-loss result.

Within each robot, all four conditions must use the same code revision, environment count, rollout length, training
budget, reward and action-feasibility configuration, command distribution, coefficient schedule, evaluation grid,
and predeclared seed set. Use at least a multi-seed confirmatory cohort, compare robots separately, publish per-seed
results, and report uncertainty rather than selecting the best checkpoint or metric after observing outcomes. Keep
standard PPO value regression enabled in every condition and keep TR transition augmentation disabled.

The contrasts answer distinct questions:

- actor only minus PPO estimates the incremental effect of TR policy consistency;
- critic only minus PPO estimates the incremental effect of the critic hypothesis;
- actor + critic minus actor only estimates the critic hypothesis when the actor constraint is present;
- the difference of those two critic contrasts tests an actor-by-critic interaction.

Predeclare learning AUC and terminal reward summaries together with tracking, gait fidelity, termination, effort,
work, and ground-reaction-force balance endpoints. Inspect nonfinite gradients, transformed-observation validity, and
the actually resolved coefficient schedules. The archived V4 combined-loss runs are development evidence and do not
replace this component ablation: they are single-seed cohorts, and Go2 also changed action-feasibility and reward
settings. Do not claim that actor-only TRS improves performance until full matched training and multi-seed evaluation
have been completed.

## Direct PowerShell commands

The commands below use the direct launcher and do not require `$env:CONDA_PREFIX`, `conda run`, `launch_study.py`, or
a JSON study launcher. They are single-seed completion-report examples; repeat the complete four-condition block for
every predeclared seed, changing both `--seed` and `--run-name`. All commands explicitly disable TR augmentation. The
PPO conditions retain the matched `500/1000` schedule metadata, but both TR coefficients and enable flags are zero, so
that schedule is inactive.

### Unitree Go2: PPO, no TRS

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot go2 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.0 `
  --tr-value-coef 0.0 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name go2_ppo_notrs_m0_v0_seed42 `
  --no-conda-run `
  --no-trs `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=false `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Unitree Go2: actor-only TRS (default method)

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot go2 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.10 `
  --tr-value-coef 0.0 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name go2_actor_only_trs_m0p1_v0_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=false `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Unitree Go2: critic-only TRS ablation

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot go2 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.0 `
  --tr-value-coef 0.05 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name go2_critic_only_trs_m0_v0p05_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=true `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Unitree Go2: actor-plus-critic TRS ablation

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot go2 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.10 `
  --tr-value-coef 0.05 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name go2_actor_value_trs_m0p1_v0p05_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=true `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Dobot X1: PPO, no TRS

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot x1 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.0 `
  --tr-value-coef 0.0 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name x1_ppo_notrs_m0_v0_seed42 `
  --no-conda-run `
  --no-trs `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=false `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Dobot X1: actor-only TRS (default method)

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot x1 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.10 `
  --tr-value-coef 0.0 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name x1_actor_only_trs_m0p1_v0_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=false `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Dobot X1: critic-only TRS ablation

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot x1 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.0 `
  --tr-value-coef 0.05 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name x1_critic_only_trs_m0_v0p05_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=true `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

### Dobot X1: actor-plus-critic TRS ablation

```powershell
.\scripts\symm_locomotion\train.ps1 `
  --robot x1 `
  --iterations 20000 `
  --num-envs 512 `
  --seed 42 `
  --mirror 0.10 `
  --tr-value-coef 0.05 `
  --tr-warmup-iterations 500 `
  --tr-rampup-iterations 1000 `
  --tr-ramp-shape half_cosine `
  --expected-branch jding/72d_actor_trs_v5 `
  --run-name x1_actor_value_trs_m0p1_v0p05_seed42 `
  --no-conda-run `
  agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true `
  agent.algorithm.symmetry_cfg.use_tr_value_consistency=true `
  agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

## Scientific decision

**The default method uses TR policy consistency only. TR critic consistency is retained as an optional empirical
hypothesis and must be supported through an actor-only/value-only/combined ablation before being included in the
principal method.**

This decision follows from the temporal orientation of the value target, not from a negative empirical result. The
archived combined-loss runs remain valid evidence for their recorded configurations. They cannot identify the
critic term's marginal contribution, and they do not establish that an actor-only replacement will improve reward,
tracking, gait fidelity, or load balance.

## Unresolved scientific risks

- The pointwise actor constraint may itself be biased by dissipative contacts, clipping, morphology, and the
  trajectory-inconsistent previous-action channel.
- A small empirical future/past return gap on recorded cycles may be distribution specific and may not generalize to
  falls, transients, command switches, or out-of-distribution gaits.
- The critic ablation may act as useful regularization despite lacking a general invariance theorem; only the matched
  factorial experiment can measure that effect.
- Coefficient, warmup, ramp, mask, and policy-output-space choices may interact, so a single coefficient pair does not
  characterize the whole method.
- Existing V4 and V5 evidence is single-seed development evidence and is insufficient for uncertainty-aware method
  selection or attribution of the actor/value component effects.
- Simulation results do not establish hardware reversibility or robustness to unmodeled impact, friction, delay,
  saturation, and sensing effects.
