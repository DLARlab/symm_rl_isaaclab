<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Time-reversal symmetry and discounted state values

Saved on November 4, 2026.

This note records the current working result for comparing \(V^\pi(s)\) with \(V^\pi(\mathcal T_{\mathcal S}(s))\). It includes the continuous-space proposition, its proof, and the explanation of the residual \(b(s)\). The mathematical argument is a working draft; priority and theoretical novelty have not been established.

## Notation and scope

Use the MDP notation \((\mathcal S,\mathcal A,\rho,P,R,\gamma)\) and the transformations \(\mathcal T_{\mathcal S}\) and \(\mathcal T_{\mathcal A}\). Write \(\cev{s}:=\mathcal T_{\mathcal S}(s)\) and \(\cev{a}:=\mathcal T_{\mathcal A}(a)\). The arrow notation uses the paper's existing LaTeX macro.

The result concerns equality between values at conjugate states. A separate comparison between \(V^\pi(s_t)\) and \(V^\pi(\cev{s}_{t+1})\) was deferred. The successor state still appears in the transition-reversal assumptions because time reversal exchanges the transition's endpoints.

## Proposition

Consider an infinite-horizon DARMDP with Borel state and action spaces \(\mathcal S\subseteq\mathbb R^n\) and \(\mathcal A\subseteq\mathbb R^m\), bounded measurable transition rewards \(R\), and discount factor \(\gamma\in(0,1)\). Let \(\pi\) be a stationary policy. Assume that \(P\) and \(\pi\) admit measurable probability densities with respect to sigma-finite reference measures \(\mu\) and \(\nu\), respectively.

Let \(\mathcal T_{\mathcal S}\) and \(\mathcal T_{\mathcal A}\) be measurable involutions that preserve their corresponding reference measures. Suppose that, for every feasible transition \((s_t,a_t,s_{t+1})\),

\[
\begin{aligned}
P(s_{t+1}\mid s_t,a_t)
&=P(\cev{s}_t\mid\cev{s}_{t+1},\cev{a}_t),\\
R(s_t,a_t,s_{t+1})
&=R(\cev{s}_{t+1},\cev{a}_t,\cev{s}_t),\\
\pi(a_t\mid s_t)
&=\pi(\cev{a}_t\mid\cev{s}_{t+1}).
\end{aligned}
\tag{1}
\]

Here, feasibility means positive transition density. The pointwise conclusions below assume versions of the densities satisfying the stated identities pointwise on that support. If the assumptions are imposed only almost everywhere, the conclusions are likewise interpreted almost everywhere.

Define the on-policy transition density and expected immediate reward by

\[
k(s,y)=\int_{\mathcal A}\pi(a\mid s)P(y\mid s,a)\,\nu(\mathrm da),
\]

\[
r(s)=\int_{\mathcal A}\int_{\mathcal S}\pi(a\mid s)P(y\mid s,a)R(s,a,y)\,\mu(\mathrm dy)\nu(\mathrm da).
\]

For bounded measurable functions \(f\), define

\[
(Kf)(s)=\int_{\mathcal S}k(s,y)f(y)\,\mu(\mathrm dy),
\qquad
(K^\leftarrow f)(s)=\int_{\mathcal S}k(y,s)f(y)\,\mu(\mathrm dy).
\]

Let

\[
r^\leftarrow(s):=r(\cev{s}),
\]

and define the residual function \(b:\mathcal S\to\mathbb R\) by

\[
b(s):=r(s)-r^\leftarrow(s)+\gamma\big[(K-K^\leftarrow)V^\pi\big](s).
\tag{2}
\]

Then

\[
V^\pi(s)-V^\pi(\cev{s})
=
\left[\sum_{j=0}^{\infty}\gamma^j(K^\leftarrow)^j b\right](s),
\tag{3}
\]

where the series converges uniformly. Consequently,

\[
V^\pi(s)=V^\pi(\cev{s})\ \text{for every }s
\quad\Longleftrightarrow\quad
b(s)=0\ \text{for every }s.
\tag{4}
\]

The symmetry conditions in Eq. (1) do not generally guarantee this additional condition, and therefore do not generally guarantee value invariance.

## Proof

Changing the action variable through \(\mathcal T_{\mathcal A}\) and applying Eq. (1) to the transition \((y,a,s)\) gives

\[
\begin{aligned}
k(\cev{s},\cev{y})
&=\int_{\mathcal A}\pi(\cev{a}\mid\cev{s})P(\cev{y}\mid\cev{s},\cev{a})\,\nu(\mathrm da)\\
&=\int_{\mathcal A}\pi(a\mid y)P(s\mid y,a)\,\nu(\mathrm da)\\
&=k(y,s).
\end{aligned}
\tag{5}
\]

The dynamics condition and involutivity ensure that feasibility is preserved in both directions. Thus, the identities multiplied by transition densities also hold outside their positive support.

Equation (5) implies that \(k^\leftarrow(s,y):=k(y,s)\) is a normalized transition density:

\[
\begin{aligned}
\int_{\mathcal S}k^\leftarrow(s,y)\,\mu(\mathrm dy)
&=\int_{\mathcal S}k(\cev{s},\cev{y})\,\mu(\mathrm dy)\\
&=\int_{\mathcal S}k(\cev{s},z)\,\mu(\mathrm dz)
=1.
\end{aligned}
\tag{6}
\]

Both transition operators therefore map bounded measurable functions to bounded measurable functions and satisfy

\[
\|Kf\|_\infty\leq\|f\|_\infty,
\qquad
\|K^\leftarrow f\|_\infty\leq\|f\|_\infty,
\tag{7}
\]

where \(\|f\|_\infty:=\sup_{s\in\mathcal S}|f(s)|\).

Changing variables in the expected reward similarly gives

\[
r^\leftarrow(s)=r(\cev{s})
=\int_{\mathcal A}\int_{\mathcal S}\pi(a\mid y)P(s\mid y,a)R(y,a,s)\,\mu(\mathrm dy)\nu(\mathrm da).
\tag{8}
\]

The forward Bellman equation is

\[
V^\pi=r+\gamma KV^\pi.
\tag{9}
\]

Define \(\overline V^\pi(s):=V^\pi(\cev{s})\). Evaluating Eq. (9) at \(\cev{s}\), changing the successor variable through \(\mathcal T_{\mathcal S}\), and applying Eq. (5) yields

\[
\begin{aligned}
\overline V^\pi(s)
&=r^\leftarrow(s)+\gamma\int_{\mathcal S}k(\cev{s},\cev{y})V^\pi(\cev{y})\,\mu(\mathrm dy)\\
&=r^\leftarrow(s)+\gamma(K^\leftarrow\overline V^\pi)(s).
\end{aligned}
\tag{10}
\]

Let \(\Delta^\pi(s):=V^\pi(s)-\overline V^\pi(s)\). Subtracting Eq. (10) from Eq. (9), then adding and subtracting \(\gamma K^\leftarrow V^\pi\), gives

\[
\Delta^\pi=b+\gamma K^\leftarrow\Delta^\pi.
\tag{11}
\]

Repeated substitution gives

\[
\Delta^\pi
=
\sum_{j=0}^{N-1}\gamma^j(K^\leftarrow)^j b
+\gamma^N(K^\leftarrow)^N\Delta^\pi.
\tag{12}
\]

Bounded rewards imply

\[
\|V^\pi\|_\infty\leq\frac{\|R\|_\infty}{1-\gamma},
\]

so \(\Delta^\pi\) is bounded. By Eq. (7),

\[
\left\|\gamma^N(K^\leftarrow)^N\Delta^\pi\right\|_\infty
\leq\gamma^N\|\Delta^\pi\|_\infty
\longrightarrow0.
\tag{13}
\]

Taking \(N\to\infty\) establishes the uniformly convergent expansion in Eq. (3). If \(b\equiv0\), this expansion gives \(\Delta^\pi\equiv0\). Conversely, if \(\Delta^\pi\equiv0\), Eq. (11) gives \(b\equiv0\). This proves Eq. (4).

To establish that Eq. (1) does not automatically enforce \(b\equiv0\), consider a continuous family with \(\mathcal S=\mathcal A=[-1,1]\), Lebesgue reference measures, and

\[
\mathcal T_{\mathcal S}(s)=-s,\qquad
\mathcal T_{\mathcal A}(a)=-a,\qquad
P(y\mid s,a)=\frac12,\qquad
\pi(a\mid s)=\frac12.
\]

Choose any bounded measurable odd function \(\phi\) satisfying

\[
\int_{-1}^{1}\phi(s)^2\,\mathrm ds>0,
\]

and define

\[
R(s,a,y)=\phi(s)-\phi(y).
\]

The constant transition and policy densities satisfy their reversal conditions. The reward condition also holds:

\[
R(-y,-a,-s)=\phi(-y)-\phi(-s)=\phi(s)-\phi(y).
\]

Since \(\phi\) is odd, its integral over \([-1,1]\) vanishes. Hence \(r=\phi\) and \(K\phi=0\), so \(\phi\) satisfies the Bellman equation. By uniqueness of the bounded discounted Bellman solution, \(V^\pi=\phi\). Consequently,

\[
V^\pi(s)-V^\pi(\cev{s})
=\phi(s)-\phi(-s)
=2\phi(s),
\]

which is nonzero on a set of positive measure. Exact symmetry of the dynamics, rewards, and policy therefore does not generally imply invariance of the discounted state-value function.

## What b(s) means

The residual can be written explicitly as

\[
b(s)
=r(s)-r(\cev{s})
+\gamma\int_{\mathcal S}\big[k(s,y)-k(y,s)\big]V^\pi(y)\,\mu(\mathrm dy).
\]

Using the forward Bellman equation, the same expression becomes

\[
b(s)
=V^\pi(s)-\left[r^\leftarrow(s)+\gamma(K^\leftarrow V^\pi)(s)\right].
\]

Thus, \(b(s)\) measures how much the original value function fails to satisfy the reversed Bellman equation. It is a state-dependent function, so pointwise definitions should include \(s\). Equations between whole functions may use \(b\) as shorthand.

The symmetry assumption applies to the transition reward \(R(s,a,y)\), not directly to the policy-averaged reward \(r(s)\). Equation (8) averages rewards over transitions arriving at \(s\), while the definition of \(r(s)\) averages rewards over transitions leaving \(s\). Equality of rewards on conjugate transitions does not require these two conditional averages to agree.

Similarly, the dynamics and policy assumptions imply \(k(\cev{s},\cev{y})=k(y,s)\), which does not generally imply \(k(s,y)=k(y,s)\). Therefore, \(K\) and \(K^\leftarrow\) can differ. A nonzero \(b(s)\) is compatible with all three symmetry assumptions.

The condition in Eq. (4) is global: \(b(s)=0\) at one state alone does not ensure equality of the two values there, because the expansion in Eq. (3) also involves residuals at states reached by \(K^\leftarrow\).

## Continuous-space interpretation

The proof acts on bounded measurable functions equipped with the supremum norm. It does not require a finite-dimensional matrix or an inverse. The notation \((I-\gamma K^\leftarrow)^{-1}\), if used later, abbreviates the bounded operator series \(\sum_{j=0}^{\infty}\gamma^j(K^\leftarrow)^j\); it does not assume that \(K^\leftarrow\) itself is invertible.

## Context for resuming the paper

The active training configuration inspected in train2.sh used policy regularization with coefficient 0.1 and TRS value regularization with coefficient 0. The actor loss uses temporally aligned reversed histories and a detached reference action mean. Its transition-level interpretation involves the conjugate successor state. The current proposition only addresses the separate question of value invariance at conjugate states; the successor-based value comparison was deliberately set aside.

The policy condition in Eq. (1) is an edgewise condition involving the two endpoints of a transition. It differs from ordinary pointwise policy equivariance. The implementation encourages consistency of actor means on sampled history pairs; the theoretical condition should be presented as an idealized assumption.

The result does not claim that conjugate values must always differ, that critic regularization is always harmful, or that policy regularization is necessarily optimal.

Writing preferences: use natural academic language, preserve the established notation, and keep each prose paragraph on one source line. Use rendered mathematics while the content is being developed; produce LaTeX source when requested.

## References to revisit before making a novelty claim

* [Barkley et al., An Investigation of Time Reversal Symmetry in Reinforcement Learning](https://proceedings.mlr.press/v242/barkley24a/barkley24a.pdf).
* [Jiang et al., Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](https://arxiv.org/abs/2505.13925).
* [Zhang et al., Learning Retrospective Knowledge with Reverse Reinforcement Learning](https://arxiv.org/abs/2007.06703).
* [Avrachenkov et al., Linking PageRank, Time Reversal, and Policy Evaluation](https://arxiv.org/abs/2605.00532).

The exact gap characterization, its application to time-reversal assumptions, and the negative result should be distinguished from standard Bellman contraction and conditional-expectation arguments when evaluating novelty.
