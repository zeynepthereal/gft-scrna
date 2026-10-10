# Pre-registration: state-matched control (GSE178341)

Written 2026-10-10, before any residual, AUROC, or CNV computation for this specific test.
No analysis below this document has been run; the eligible-pair COUNTS reported here are
pure bookkeeping (patient/state/tissue cell counts already on disk in the metatable and
cluster CSVs) and required no residual, CNV, or AUROC computation to produce.

## Motivation

Tumor-channel epithelium is dominated by stem/TA-like states (cE01, cE03), while
normal-channel epithelium is dominated by differentiated states (enterocytes, goblet
cells). The near-perfect tumor-vs-normal residual AUROC reported in
`results/gse178341_prereg_results.txt` (median ~0.99) could in principle reflect this
differentiation-state composition difference rather than malignancy per se — a cell's
residual could be driven by "which state am I in" more than by "am I malignant." This
test isolates malignancy from state by comparing tumor and normal cells WITHIN the same
authors'-assigned epithelial state.

## Data and pipeline (frozen, no parameter changes)

- **Patients:** the same 34 chemistry-matched, >=50/>=50 eligible patients from
  `docs/prereg_gse178341.md` (Amendment 1), as implemented in
  `scripts/gse178341_prereg.py`.
- **Residual pipeline:** identical to `scripts/gse178341_prereg.py` — binomial thinning
  to the 10th percentile of total counts, label-blind top-3000 HVGs, 50-D PCA per
  patient (fit on ALL of that patient's included normal+tumor epithelial cells, label-
  blind), q=20 normal subspace fit on **all** of that patient's normal epithelium
  (unchanged from the main pre-registered run — not refit per state), 5-fold held-out
  residual for normal cells (seed 0), full-fit residual for tumor cells. The ONLY new
  step relative to the main run is which cells are included in the AUROC computation
  (restricted to one state at a time) — the subspace fit itself, thinning, HVG
  selection, and PCA are unchanged and still use the full patient-level normal/tumor
  epithelium, exactly as frozen.
- **CNV score:** the cross-fitted CNV score from `scripts/gse178341_posthoc.py` Part B
  (5-fold cross-fitted normal-epithelium reference, non-epithelial cells always in the
  reference, window_size=250), not the Amendment-2 (non-cross-fitted) score, since the
  cross-fitted version is the fairer comparison established in the post-hoc analysis.

## State labels

Pelka et al.'s epithelial subtype clusters `cE01`-`cE11` (`cl295v11SubShort` in
`GSE178341_crc10x_full_c295v4_submit_cluster.csv`), assigned to every epithelial cell
regardless of tumor/normal origin (see `docs/prereg_gse178341.md`'s original data
inspection — these are differentiation-state labels, not a malignancy flag).
**Caveat, stated before any analysis:** the authors' cell-state classifier for
**tumor**-channel cells was trained on **normal** epithelium's reference states; a
tumor cell's state label is therefore itself a classifier output, not a direct
observation, and could be systematically biased for malignant cells whose
transcriptomes don't cleanly match any normal reference state. State-matching cannot
fully rule out this residual confound — it narrows the comparison to pairs the authors'
own classifier called the same state, which is a weaker guarantee than true matching on
an independent, non-expression-derived cell property.

## Analysis unit

A **(patient, state)** pair is eligible if that patient has **>=30 tumor-channel AND
>=30 normal-channel** epithelial cells assigned to that state (`cE01`-`cE11`).

## Eligible-pair counts (computed now, bookkeeping only — no residual/AUROC/CNV run)

**Total eligible (patient, state) pairs: 136**, drawn from all 34 eligible patients
(every one of the 34 contributes at least one eligible pair) and 9 of the 11 states
(`cE07` and `cE11` have zero eligible pairs — too rare/small in this cohort).

By MMR status: **MMRd = 75 pairs, MMRp = 61 pairs.**

By state:

| State | Description | MMRd | MMRp | Total |
|---|---|---|---|---|
| cE01 | Stem/TA-like | 19 | 14 | 33 |
| cE02 | Stem/TA-like/Immature Goblet | 10 | 7 | 17 |
| cE03 | Stem/TA-like prolif | 17 | 11 | 28 |
| cE04 | Enterocyte 1 | 8 | 7 | 15 |
| cE05 | Enterocyte 2 | 8 | 9 | 17 |
| cE06 | Immature Goblet | 8 | 9 | 17 |
| cE08 | Goblet | 3 | 2 | 5 |
| cE09 | Best4 | 1 | 1 | 2 |
| cE10 | Tuft | 1 | 1 | 2 |
| cE07 | Goblet/Enterocyte | 0 | 0 | 0 |
| cE11 | Enteroendocrine | 0 | 0 | 0 |

136 >= 10, so **the test is NOT declared underpowered** by the pre-specified rule below;
all 136 pairs will be used. Note the count is heavily dominated by the three stem/TA-like
states (cE01+cE02+cE03 = 78/136, 57%) and by cE01 in particular (33 pairs) — any
state-level breakdown in the eventual results should be read with this imbalance in
mind, since the rarer differentiated states (cE08-cE10, 9 pairs combined) will
contribute little statistical weight to the pooled median.

## Hypotheses

**H-SM1 (residual detects malignancy beyond differentiation state).** Supported if,
across the 136 eligible (patient, state) pairs: the median within-state residual AUROC
is **> 0.75** AND within-state residual AUROC is **> 0.6** in **>= 70%** of eligible
pairs.

**H-SM2 (state-matched residual vs CNV, descriptive, no threshold).** Report median
within-state (residual AUROC − cross-fitted CNV AUROC), overall and split by MMR status.

**Underpowered rule (not triggered here, stated for completeness):** if fewer than 10
eligible pairs existed, the test would be declared underpowered and reported
descriptively only, with no pass/fail verdict for H-SM1. This does not apply — 136
pairs are eligible.

## What is NOT pre-registered here

No fallback or secondary threshold is specified if H-SM1's two conditions disagree
(e.g., median > 0.75 but < 70% of pairs > 0.6, or vice versa) beyond reporting both
numbers and declaring H-SM1 "not supported" whenever either condition fails, consistent
with how `docs/prereg_gse178341.md`'s AND-combined rules have been read throughout this
project. No new pipeline parameters are introduced; everything is inherited frozen from
`scripts/gse178341_prereg.py` and `scripts/gse178341_posthoc.py`.

## Commit scope

This document only. No residual, CNV, or AUROC script for the state-matched test has
been written or run.
