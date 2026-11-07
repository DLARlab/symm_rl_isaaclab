<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Late-training reward component comparison

Primary window: iterations 19000-19999. Sensitivity windows: final 500 and 2000 iterations.

Reward components are weighted episodic sums divided by the nominal maximum episode duration (30 s).
Higher values are preferable; for negative terms, higher means a smaller penalty.
The environment applies nonlinear positive/negative reward composition, so component deltas are not additive return contributions.
These are descriptive comparisons of one trained policy per robot and method.
Episode durations and the active curriculum bins differ between policies; this is not a matched-command evaluation.

## Go2

| Reward component | PPO | PPO-TRS | Difference | Change / absolute PPO |
|---|---:|---:|---:|---:|
| track_lin_vel_xy_exp | 0.270359 | 0.280589 | +0.010229 | +3.78% |
| track_ang_vel_z_exp | 0.213877 | 0.222546 | +0.008669 | +4.05% |
| base_height | 0.127299 | 0.130194 | +0.002896 | +2.27% |
| action_smoothness | -0.018202 | -0.015460 | +0.002742 | +15.06% |
| foot_clearance | -0.052858 | -0.050583 | +0.002275 | +4.30% |
| base_roll_exp | -0.023037 | -0.020982 | +0.002055 | +8.92% |
| foot_periodicity_velocity | -0.156400 | -0.154402 | +0.001998 | +1.28% |
| dof_acc_l2 | -0.025605 | -0.023806 | +0.001799 | +7.03% |
| action_rate_l2 | -0.009923 | -0.008663 | +0.001260 | +12.70% |
| foot_periodicity_force | -0.100549 | -0.099612 | +0.000937 | +0.93% |
| dof_vel_l2 | -0.030971 | -0.030773 | +0.000198 | +0.64% |
| smoothness | -0.055720 | -0.055709 | +0.000011 | +0.02% |
| alive_bonus | 0.000000 | 0.000000 | +0.000000 | -- |
| termination_penalty | 0.000000 | 0.000000 | +0.000000 | -- |
| leg_permutation_symmetry | -0.120353 | -0.120488 | -0.000135 | -0.11% |
| joint_target_limits | -0.001109 | -0.001265 | -0.000156 | -14.04% |
| dof_torques_l2 | -0.031471 | -0.031852 | -0.000381 | -1.21% |
| hip_action_penalty | -0.010188 | -0.010656 | -0.000468 | -4.59% |
| lin_vel_z_l2 | -0.008835 | -0.009400 | -0.000565 | -6.40% |

| Training diagnostic | PPO | PPO-TRS |
|---|---:|---:|
| Derived/foot_clearance_mean_overshoot | 0.008035 | 0.006336 |
| Diagnostics/foot_clearance_mean_height_error | 0.013243 | 0.011587 |
| Diagnostics/foot_clearance_mean_shortfall | 0.005208 | 0.005251 |
| Diagnostics/foot_clearance_mean_swing_height | 0.054607 | 0.052888 |
| Diagnostics/foot_clearance_mean_target_height | 0.051780 | 0.051804 |
| Diagnostics/ji22_multiplier_mean | 0.491809 | 0.506280 |
| Diagnostics/ji22_negative_reward_mean | -0.014890 | -0.014269 |
| Diagnostics/ji22_positive_reward_mean | 0.014103 | 0.014264 |
| Diagnostics/joint_target_clipped_fraction | 0.011088 | 0.007696 |
| Episode_Termination/base_contact | 0.000000 | 0.000000 |
| Episode_Termination/base_height | 0.000000 | 0.000000 |
| Episode_Termination/base_orientation | 0.005080 | 0.001847 |
| Episode_Termination/calf_height | 0.223843 | 0.194663 |
| Episode_Termination/head_contact | 0.012080 | 0.007416 |
| Episode_Termination/time_out | 0.759970 | 0.796769 |
| Metrics/base_velocity/curriculum_active_bins | 264.000000 | 272.473125 |
| Metrics/base_velocity/curriculum_max_abs_ang_vel_z | 2.000000 | 2.000000 |
| Metrics/base_velocity/curriculum_max_abs_lin_vel_x | 2.142857 | 2.142857 |
| Metrics/base_velocity/error_vel_xy | 0.273517 | 0.259867 |
| Metrics/base_velocity/error_vel_yaw | 0.212812 | 0.200962 |
| Metrics/base_velocity/tracking_reward_ang_vel | 0.786644 | 0.804334 |
| Metrics/base_velocity/tracking_reward_lin_vel | 0.703207 | 0.725374 |
| Metrics/success_rate | 0.434099 | 0.478137 |
| Train/mean_episode_length | 1302.569201 | 1333.354469 |
| Train/mean_reward | 9.159598 | 9.753661 |

## X1-Opposed

| Reward component | PPO | PPO-TRS | Difference | Change / absolute PPO |
|---|---:|---:|---:|---:|
| track_lin_vel_xy_exp | 0.291069 | 0.300614 | +0.009544 | +3.28% |
| foot_periodicity_velocity | -0.218312 | -0.208989 | +0.009322 | +4.27% |
| track_ang_vel_z_exp | 0.224051 | 0.232973 | +0.008923 | +3.98% |
| foot_clearance | -0.068857 | -0.063301 | +0.005556 | +8.07% |
| action_smoothness | -0.028107 | -0.024379 | +0.003729 | +13.27% |
| foot_periodicity_force | -0.074852 | -0.072225 | +0.002628 | +3.51% |
| base_height | 0.131378 | 0.133870 | +0.002492 | +1.90% |
| action_rate_l2 | -0.016353 | -0.014180 | +0.002173 | +13.29% |
| dof_acc_l2 | -0.025158 | -0.023580 | +0.001578 | +6.27% |
| lin_vel_z_l2 | -0.011671 | -0.011050 | +0.000621 | +5.32% |
| joint_target_limits | -0.003867 | -0.003672 | +0.000196 | +5.06% |
| alive_bonus | 0.000000 | 0.000000 | +0.000000 | -- |
| termination_penalty | 0.000000 | 0.000000 | +0.000000 | -- |
| dof_torques_l2 | -0.038467 | -0.038730 | -0.000263 | -0.68% |
| smoothness | -0.070691 | -0.071155 | -0.000465 | -0.66% |
| dof_vel_l2 | -0.033723 | -0.034432 | -0.000709 | -2.10% |
| hip_action_penalty | -0.011204 | -0.012221 | -0.001016 | -9.07% |
| base_roll_exp | -0.026463 | -0.027618 | -0.001155 | -4.37% |
| leg_permutation_symmetry | -0.133287 | -0.134854 | -0.001568 | -1.18% |

| Training diagnostic | PPO | PPO-TRS |
|---|---:|---:|
| Derived/foot_clearance_mean_overshoot | 0.014000 | 0.009593 |
| Diagnostics/foot_clearance_mean_height_error | 0.020133 | 0.015671 |
| Diagnostics/foot_clearance_mean_shortfall | 0.006133 | 0.006078 |
| Diagnostics/foot_clearance_mean_swing_height | 0.075051 | 0.070692 |
| Diagnostics/foot_clearance_mean_target_height | 0.067184 | 0.067177 |
| Diagnostics/ji22_multiplier_mean | 0.457669 | 0.473689 |
| Diagnostics/ji22_negative_reward_mean | -0.016519 | -0.015768 |
| Diagnostics/ji22_positive_reward_mean | 0.014023 | 0.014211 |
| Diagnostics/joint_target_clipped_fraction | 0.019658 | 0.012292 |
| Episode_Termination/base_contact | 0.000000 | 0.000000 |
| Episode_Termination/base_height | 0.019399 | 0.015647 |
| Episode_Termination/base_orientation | 0.034475 | 0.025122 |
| Episode_Termination/calf_height | 0.084375 | 0.069169 |
| Episode_Termination/time_out | 0.857782 | 0.887423 |
| Metrics/base_velocity/curriculum_active_bins | 334.849125 | 331.000000 |
| Metrics/base_velocity/curriculum_max_abs_ang_vel_z | 2.000000 | 2.000000 |
| Metrics/base_velocity/curriculum_max_abs_lin_vel_x | 2.428571 | 2.428571 |
| Metrics/base_velocity/error_vel_xy | 0.274367 | 0.258339 |
| Metrics/base_velocity/error_vel_yaw | 0.224218 | 0.208115 |
| Metrics/base_velocity/tracking_reward_ang_vel | 0.774560 | 0.796675 |
| Metrics/base_velocity/tracking_reward_lin_vel | 0.729313 | 0.750930 |
| Metrics/success_rate | 0.424463 | 0.461355 |
| Train/mean_episode_length | 1383.849488 | 1408.680533 |
| Train/mean_reward | 9.049350 | 9.637022 |

## X1-Aligned

| Reward component | PPO | PPO-TRS | Difference | Change / absolute PPO |
|---|---:|---:|---:|---:|
| track_lin_vel_xy_exp | 0.280114 | 0.285841 | +0.005727 | +2.04% |
| track_ang_vel_z_exp | 0.211402 | 0.215349 | +0.003947 | +1.87% |
| foot_periodicity_velocity | -0.221347 | -0.217468 | +0.003879 | +1.75% |
| action_smoothness | -0.029141 | -0.025580 | +0.003561 | +12.22% |
| foot_clearance | -0.069440 | -0.066378 | +0.003062 | +4.41% |
| action_rate_l2 | -0.017897 | -0.015578 | +0.002319 | +12.96% |
| foot_periodicity_force | -0.085014 | -0.083143 | +0.001871 | +2.20% |
| dof_acc_l2 | -0.026186 | -0.024770 | +0.001416 | +5.41% |
| leg_permutation_symmetry | -0.132012 | -0.131095 | +0.000917 | +0.69% |
| base_roll_exp | -0.025545 | -0.024651 | +0.000895 | +3.50% |
| dof_vel_l2 | -0.034493 | -0.033880 | +0.000612 | +1.78% |
| base_height | 0.129338 | 0.129902 | +0.000564 | +0.44% |
| hip_action_penalty | -0.010814 | -0.010453 | +0.000361 | +3.34% |
| joint_target_limits | -0.002964 | -0.002609 | +0.000355 | +11.98% |
| dof_torques_l2 | -0.038665 | -0.038323 | +0.000342 | +0.88% |
| alive_bonus | 0.000000 | 0.000000 | +0.000000 | -- |
| termination_penalty | 0.000000 | 0.000000 | +0.000000 | -- |
| smoothness | -0.070940 | -0.071096 | -0.000156 | -0.22% |
| lin_vel_z_l2 | -0.012565 | -0.013841 | -0.001276 | -10.16% |

| Training diagnostic | PPO | PPO-TRS |
|---|---:|---:|
| Derived/foot_clearance_mean_overshoot | 0.014491 | 0.011591 |
| Diagnostics/foot_clearance_mean_height_error | 0.021894 | 0.018922 |
| Diagnostics/foot_clearance_mean_shortfall | 0.007403 | 0.007331 |
| Diagnostics/foot_clearance_mean_swing_height | 0.074257 | 0.071443 |
| Diagnostics/foot_clearance_mean_target_height | 0.067169 | 0.067183 |
| Diagnostics/ji22_multiplier_mean | 0.439566 | 0.448358 |
| Diagnostics/ji22_negative_reward_mean | -0.017366 | -0.016935 |
| Diagnostics/ji22_positive_reward_mean | 0.013893 | 0.014074 |
| Diagnostics/joint_target_clipped_fraction | 0.031038 | 0.022183 |
| Episode_Termination/base_contact | 0.000000 | 0.000000 |
| Episode_Termination/base_height | 0.007771 | 0.016183 |
| Episode_Termination/base_orientation | 0.012202 | 0.010959 |
| Episode_Termination/calf_height | 0.170665 | 0.158916 |
| Episode_Termination/time_out | 0.806727 | 0.811713 |
| Metrics/base_velocity/curriculum_active_bins | 335.000000 | 342.758500 |
| Metrics/base_velocity/curriculum_max_abs_ang_vel_z | 2.000000 | 2.000000 |
| Metrics/base_velocity/curriculum_max_abs_lin_vel_x | 2.714286 | 2.714286 |
| Metrics/base_velocity/error_vel_xy | 0.280921 | 0.271211 |
| Metrics/base_velocity/error_vel_yaw | 0.239775 | 0.231176 |
| Metrics/base_velocity/tracking_reward_ang_vel | 0.753926 | 0.764099 |
| Metrics/base_velocity/tracking_reward_lin_vel | 0.718020 | 0.728877 |
| Metrics/success_rate | 0.356896 | 0.397844 |
| Train/mean_episode_length | 1341.915889 | 1344.191692 |
| Train/mean_reward | 8.344060 | 8.630788 |

