# Why the design is as it is

**Background, not frozen.** This file explains or compares; it does not define the test and is
not part of freeze 1 or 2. Where it and [DESIGN.md](../DESIGN.md) differ, DESIGN.md governs.

## Why planar pushing

- **Exact, cheap counterfactuals.** MuJoCo can save its full integration state, restore it and
  step again. From any recorded state we can apply the logged action or any other action and get
  the true outcome of each. A test costs microseconds of simulation.
- **A built-in hard case.** The box moves only when the pusher touches it. Without contact,
  copying the box pose is exactly right, as in-tick swaps are nearly linear in evm-intervene.
  Contact is the analogue of crossing a liquidity cliff: rare in routine data, where the action
  matters most.
- **A shortcut by construction.** The scripted policy makes actions predictable from the state
  and recent history. A model can therefore reach a low validation loss while relying on the
  action less than it should. The state is fully observed, so P(s' | s, a) is the same whether a
  comes from the policy or is forced. Any gap measures what each model learned from correlated,
  unevenly covered data, not missing inputs.
- **Small.** Low-dimensional state, no pixels. Everything trains on a CPU or a free Colab or
  Kaggle GPU.

## Why grazes are not gated

Grazes are excluded from the gated stratum because they are easy for every model: a prediction of
"barely moves" is close to right, so they add a similar small error to both models in a pair and
dilute the relative gap. Example: real pushes with errors 2.0 vs 2.5 (a 25% gap), pooled with 6,000 grazes at 0.5 each out
of 9,000 contacts, give 1.00 vs 1.17 (17%, below the 20% threshold). The effect is small when
models are nearly exact on grazes and larger otherwise. The push thresholds use only the
simulator's true outcome, so they treat every model alike.

## Why `contact_wm`

`mlp`, `gru` and `mlp-ms` are general-purpose networks with the same inputs, trained on the same
data, and may learn nearly the same function; in evm-intervene's primary arm its three
general-purpose models were within 0.6% of each other under intervention, and every difference
came from its structured model, `conservation_wm`. `contact_wm` is this task's counterpart: the
form of pushing physics is written into the model and only its numbers are learned, so it is the
one main model whose design differs in kind.

Unlike `conservation_wm`, `contact_wm` is expected to have the *lowest* observational loss, not
the highest: this scene's mechanisms have the form of its law, and in Phase 0 smoke runs its
L_obs was about a third of the general-purpose models'. So matching stops it early, not the
others, and its G1 pairs are read asymmetrically (DESIGN.md, *The structured model*).

## Why `mlp-ms` is the model with a different objective

The issue asks for a model with a different objective as well as one with a different structure.
The first draft used an RSSM trained on the ELBO (PlaNet). In Phase 0 smoke runs it learned
slowest, ended with the highest L_obs, and so set V* for every pair, pulling every other model to
an early checkpoint; it also changed several things at once (objective, sequence training, a
stochastic latent, sampled predictions), so a gap could not be put down to the objective. Rescoring
showed its gap was not sampling noise but a prior that did not learn what its latent encoded.
DreamerV3's KL terms did not help, and a Gaussian negative log-likelihood network (PETS without
the ensemble) fitted far worse. `mlp-ms` keeps `mlp`'s network and changes only the objective, to a
4-step rollout loss: deterministic, as cheap as `mlp`, and in Phase 0 about 35% above `mlp` on
L_obs, close enough that matching stays late for every model. Its contrast is milder than an
ELBO model's, which DESIGN.md's *Limits* states.

How to read a G1 pass:

A G1 pass from a pair with `contact_wm` says a model given the form of the rules
  differs under intervention from one that had to learn them, at matched loss: evm-intervene's
  finding, in this task. A pass from a pair of general-purpose models says that similar learned
  models can differ. The result states which pairs pass (`gate_g1_passing_pairs`).

## Why the positive control's flaw is known

Given the state and history, a'_t has the same distribution as a_t,
  so no input reveals whether the action was replaced: every input is genuine with probability
  1 − p. The best predictor the model can learn is then
  (1 − p) · f(s, h, a) + p · m(s, h), where f is the true outcome and m its average over the
  behaviour policy's actions. Its extra error is p² · (f − m)², small for actions the policy would
  take and large for actions it would not. So its L_obs rises slightly while its E_int on
  push interventions rises by a predictable amount.

## Why p is computed after the main runs

`mlp-shortcut` must reach the final V*, or the positive control fails the match screen and a G1
*fail* becomes unreadable. Its extra loss comes from the replaced actions, about p² · D_obs, and
does not shrink with more data.

**Why not set p in the pilot.** The pilot trains on 200 episodes and the main runs on 1,000. The
main models therefore reach a lower L_obs, and the final V* is very likely lower than the pilot's.
A p that just reaches the pilot's V* would then be too large for the final one, by an amount
nobody can know in advance. A safety margin (for example aiming at 0.9 × the pilot's V*) would be
a guess at that amount.

**The rule.** After the medium-width main runs, p is the largest grid value with
L_mlp + p² · D_obs ≤ V*, where L_mlp is `mlp`'s seed-median best L_obs, V* the final matching
level and D_obs computed by the simulator on the validation transitions. `mlp-shortcut` is a
copy of `mlp` with the replacements, so its loss is about `mlp`'s plus the built-in flaw. The
formula holds for a well-fitted model, so the runs may still miss V* slightly. V*'s 10% margin
leaves room, and a miss is caught by the match screen and reported, as for any run.

**Why the experiment stays valid.** A pre-registered choice is safe when it cannot depend on the
outcome being tested, here intervention error on the test states, whenever it is made:

- the rule is fixed before freeze 1; only its value is filled in, mechanically;
- its inputs are validation loss on recorded actions and the simulator; no intervention error of
  any model is involved;
- it is made before freeze 2, which hashes p and the `mlp-shortcut` checkpoints; the test states
  are generated from that hash, so they do not exist yet.

V*, each run's matched checkpoint and the exclusions are chosen after training in the same way.
Taking the largest p that reaches V* does not steer the control: whether it then separates on
the test states is unknown when p is chosen.

## Why 10 seeds

G1 resamples each architecture's seeds as well as the held-out states, so that a difference
between training seeds cannot pass as a difference between architectures. With 1,000 states and
thousands of push transitions, the state part of the interval is small; its width is set mostly
by how much an architecture's seeds disagree. G1 is then in effect a comparison of n seed-level
values against n, and the bootstrap is its estimate of how much those values vary.

**Why not 5, evm-intervene's number.** evm-intervene uses seeds 11, 22, 33, 44, 55. It does not
resample seeds, and it compares each model with its structured anchor, where the gap is about 20
times, so seed noise has little effect on its result. Among its general-purpose models, at matched checkpoints in its crash-exposed arm,
`latent_jepa`'s 5 seeds range from 54.0 to 59.0 in cliff std-MSE, while the three general-purpose
models' means lie between about 55 and 57. Seed spread is as large as the differences between
architectures. This design compares general-purpose models with each other, so it is exposed to
exactly that.

**What resampling n seeds does.** The bootstrap treats the n observed seeds as the whole
population. Its variance uses 1/n rather than 1/(n − 1), shrinking it by (n − 1)/n, and the
percentile interval behaves roughly like a normal interval where a t interval with n − 1 degrees of
freedom is correct:

| | 5 seeds | 10 seeds |
| --- | --- | --- |
| correct multiplier, t with n − 1 df | 2.78 | 2.26 |
| bootstrap, roughly 1.96 × √((n − 1)/n) | 1.75 | 1.86 |
| seed part of the interval too narrow by | about 37% | about 18% |
| distinct resamples per architecture | 126 | 92,378 |
| smallest exact two-sided p, n vs n (permutation) | 2/252 ≈ 0.008 | 2/184,756 ≈ 0.00001 |

With 5 seeds a correct exact test, after Holm over 6 pairs, can reach p < 0.05 only if every
seed of one architecture beats every seed of the other (0.008 × 6 ≈ 0.048). The bootstrap's p can
go down to 1/5,001, but that precision comes from treating 5 seeds as the population, not from
the data. A 6th seed would fall outside the range of the first 5 one time in three, and the
bootstrap never produces one that does.

**Why 10 and not more.** 10 seeds halves the narrowing and gives the seed resampling enough
distinct outcomes to be meaningful. The rest is checked rather than removed: Phase 0's repeated
stand-in twins (now with 10 "seeds") measure the false-alarm rate of the full procedure, and the
real twin pairs show it on trained models. Going to 20 would double the medium-width runs again
for a further narrowing from about 18% to about 9%.

**Why only at medium width.** G1, the gated test, uses the medium width. Small and large widths
enter only secondary analyses and G2, which ranks runs rather than resampling seeds, so they keep
5 seeds. Total: 80 main runs and 60 control runs, against 60 and 30 before, within the compute
fallback in DESIGN.md (*Compute*).

**Why the twins also have 10.** A negative control checks the same procedure on a pair with no
real difference. It must be run at the same size as the comparison it guards, 10 seeds against
10, or its false-alarm rate is not the main test's.

**Comparability with evm-intervene.** The first five seeds are evm-intervene's, and every run's
results are in `seed_checkpoints`, so a 5-seed, states-only gap (evm-intervene's procedure) can be
recomputed from the result file.

## Why one matched level, with full training

**What G1 answers.** forward#5 asks whether models with similar validation loss differ under
intervention: whether validation loss alone tells you how good a world model is at "what if". A
pair of models with the same L_obs and different E_int answers yes, whatever the reason for the
difference. A checkpoint chosen by validation loss is an ordinary model; it is how checkpoints are
usually chosen. So G1 compares at one shared level, V*, as evm-intervene does.

**How V* is chosen.** evm-intervene matches every model to the 5-seed mean best L_obs of its
structured model, `conservation_wm`, the model with the highest loss floor, which is compared at
its own best checkpoint. This design does not fix the anchor on one model, because a fixed anchor
works only if it is the worst model: matching stops runs early, so it can only reach losses above
a model's best, and in this task the structured model is expected to be the best, not the worst.
It uses the same idea for all four architectures: V* is 1.1 ×
the largest architecture median of best L_obs. It is the lowest level every architecture can
reach, so models are compared as fully trained as the worst one allows. The median keeps one bad
seed from moving it. The 10% margin is needed because the worst architecture is screened too:
with no margin, about half its seeds could not reach its own median and would be excluded.

**The cost: the matched point favours learning speed.** For every architecture except the worst,
V* is an earlier checkpoint. In classification, rare, atypical and hard examples are learned last,
by every measure of difficulty used (Arpit et al., ICML 2017; Toneva et al., ICLR 2019; Jiang et
al., ICML 2021; Baldock et al., NeurIPS 2021); in regression, networks fit smooth structure
before sharp structure (Rahaman et al., ICML 2019). Pushes are rare, nearly discontinuous
transitions, so they are likely still being learned at V*. A gap there may be a difference in
learning speed, and the bias can also hide a gap. In vision, gaps that come from a built-in prior
(pretraining) appear during training and fade at convergence (Andreassen et al., TMLR 2022);
`contact_wm`, whose prior is the form of the pushing law, is the closest case here.

**Why not match by model size instead.** Training every architecture to convergence and shrinking
the better ones until their best L_obs equals the worst's removes the training-stage handicap but
adds a capacity handicap, which falls on the same rare examples: pruning a ResNet-50 to 90%
sparsity cost 4 points of ImageNet accuracy overall but dropped accuracy to about 40% on the 10%
of images most affected (Hooker et al. 2019); at equal loss, larger models transfer better (Liu
et al., ICML 2023). In vision, size and training time move models along the same line of
in-distribution against out-of-distribution accuracy (Miller et al., ICML 2021), so neither route
is neutral.

**What the design does instead.**

- *Full training, constant learning rate.* No run stops early, so every run reaches its own floor
  and the best-checkpoint comparison and the curves are complete. A constant rate makes every
  checkpoint a fair "stopped here" model: checkpoints in the middle of a decaying schedule are not
  comparable with runs whose schedule ends there (Hoffmann et al. 2022).
- *Dense early checkpoints*, so matching does not overshoot V* where L_obs falls fastest.
  evm-intervene's `mlp_direct` missed its match screen by 15% for this reason.
- *Gap along training*, reported. Comparing models along their whole curve of out-of-distribution
  against in-distribution performance, not at one point, is how transient gaps are caught
  (Andreassen et al.; Teney et al., NeurIPS 2023; checkpoints of one run lie on the same line as
  separately trained models, Baek et al., NeurIPS 2022). *Overlap* means L_obs determines E_int
  in that range; *persistent* means the gap does not depend on the level chosen; *closing* means
  it is a difference in learning speed.
- *`matched_only`*, reported beside the G1 verdict when a pass at V* is not seen at best
  checkpoints. It does not change the verdict, since the issue's question does not depend on the
  cause.

No study found compares world models this way; the evidence above is from vision, language and
model-based RL, so these choices rest on analogy, and the curves measure the point directly.

## Notes on G1

- A state with more push actions weighs more in E_int, as in evm-intervene.
- The reference is chosen after scoring so that Δ_int ≥ 0. Because the test is two-sided, this
  does not change p.
- Seeds are resampled so that a difference between two training seeds of the same architecture
  cannot pass as an architecture difference.
- Everything except the seed resampling and the choice of reference follows evm-intervene; both
  differences are in *Deliberate differences*.
