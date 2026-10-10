# Session summary: curvature and covariance-geometry analyses of the projection residual

Dates: 2026-09-27 to 2026-09-30. Nothing was committed to git. All new scripts and results are untracked.

## 1. What we set out to do

We started from a discussion of blind spots in genomics AI and a physics analogy (Einstein: interaction as geometry). The goal was a stronger thesis axis built on the existing **projection residual**:

> residual = distance of a cell from the 20-D PCA subspace of the same donor's normal epithelial cells

Two lines were tested in turn:

1. **Local graph geometry.** Ollivier-Ricci curvature (ORC) on cell kNN graphs, and whether it relates to the residual.
2. **Covariance geometry.** Three questions:
   - Is the residual the dominant term of a probabilistic PCA (PPCA) potential?
   - Do the principal angles between normal and malignant subspaces show changed covariance?
   - Is there a dynamic network biomarker (DNB) signature along residual quantiles?

Every test had a decision rule written into its output before the numbers.

## 2. Data and common setup

- **Data:** `data/crc_epithelial.h5ad` (13,843 cells). The GSE161277 data was not used in this session.
- **Groups:**
  - normal = `disease=='normal' & cell_type!='malignant cell'`
  - malignant = `disease=='colorectal cancer' & cell_type=='malignant cell'`
  - tumour-non-malignant = `disease=='colorectal cancer' & cell_type!='malignant cell'`
- **Donors:** 6 donors with ≥50 normal and ≥50 malignant cells.

| donor | N normal | N malignant |
|---|---|---|
| HTA8_6002 | 264 | 60 |
| HTA8_6009 | 822 | 761 |
| HTA8_6013 | 769 | 667 |
| HTA8_6016 | 1510 | 782 |
| HTA8_6018 | 1256 | 2577 |
| HTA8_6028 | 206 | 1234 |

- **Majority rule:** unless noted otherwise, a claim counts as supported when its criterion holds in ≥4/6 donors.
- **PCA cache:** `scripts/extract_pca50.py` writes `_ricci_tmp/crc_pca50.npz`. It holds a 50-D PCA from an implicitly centred sparse SVD of log-normalised X over all genes, plus n_genes (raw nnz>0) and obs labels.
- **Sanity check:** the in-sample residual ratios computed from this cache matched `results/all_samples_residual.txt` exactly.

| donor | residual ratio (cache = published) |
|---|---|
| 6002 | 3.58x |
| 6009 | 2.97x |
| 6013 | 4.81x |
| 6016 | 3.66x |
| 6018 | 5.51x |
| 6028 | 4.18x |

## 3. Analyses run

### 3.1 `scripts/ricci_curvature_pilot.py` → `results/ricci_curvature_pilot.txt`

**Setup:**
- Graph: cosine kNN, k=20, symmetrised and unweighted, on 50-D PCA of normal+malignant cells.
- Curvature: ORC with alpha=0.5, exact W1 (POT), hop-distance cost.
- Residual: q=20, with normal cells scored 5-fold held-out.

**Tests:**
- T1: OLS `node ORC ~ is_mal + log deg + log n_genes`.
- T2: partial Spearman(residual, ORC) within malignant cells.
- T3: cross-edge vs within-edge ORC, 1000 label permutations.

| donor | T1 coef_mal | T1 p | T2 rho_mal | T2 p | rho_normal | T3 gap | n_cross | perm p |
|---|---|---|---|---|---|---|---|---|
| 6002 | -0.0697 | 2.8e-03 | -0.166 | 2.1e-01 | -0.437 | -0.2318 | 285 | 0.0010 |
| 6009 | -0.0829 | 1.6e-39 | +0.061 | 9.0e-02 | -0.029 | -0.2003 | 505 | 0.0010 |
| 6013 | -0.0989 | 8.7e-61 | +0.148 | 1.3e-04 | +0.037 | -0.3168 | 150 | 0.0010 |
| 6016 | -0.0321 | 8.6e-10 | -0.017 | 6.4e-01 | +0.204 | -0.1449 | 424 | 0.0010 |
| 6018 | -0.1076 | 6.4e-191 | +0.033 | 9.5e-02 | +0.125 | -0.3354 | 78 | 0.0010 |
| 6028 | -0.1595 | 6.8e-98 | +0.022 | 4.4e-01 | -0.207 | -0.1462 | 171 | 0.0010 |

**Outcome:**
- T2 criterion met in **0/6** donors: NOT SUPPORTED.
- T3 met in **6/6**. It was noted afterwards that T3 is expected by construction when normal and malignant cells form separate clusters, so it is not treated as evidence.

### 3.2 `scripts/ricci_t1_robustness.py` → `results/ricci_t1_robustness.txt`

**Tests:**
- 1a: separate graphs per group, equal N, 20 seeds.
- 1b: OLS with added covariates log kNN-distance and local intrinsic dimension (ID, Levina-Bickel MLE).
- 1c: k = 10, 20, 30.
- 1d: tumour-non-malignant cells as a third group.

**Results:**
- 1a: malignant < normal in 100% of seeds, in 6/6 donors, at every k.
- 1b coef_mal at k=20:

| donor | 1b coef_mal | p |
|---|---|---|
| 6002 | +0.0756 | 8.6e-05 |
| 6009 | +0.0190 | 8.8e-04 |
| 6013 | -0.0429 | 2.7e-10 |
| 6016 | +0.0587 | 1.4e-21 |
| 6018 | -0.0717 | 2.8e-120 |
| 6028 | -0.0848 | 1.3e-36 |

- 1d mean ORC (normal / malignant / tumour-non-malignant):

| donor | normal | malignant | tumour-non-malignant |
|---|---|---|---|
| 6002 | 0.3831 | 0.2682 | n/a |
| 6009 | 0.1294 | 0.0465 | 0.1435 |
| 6013 | 0.1411 | 0.0330 | 0.1984 |
| 6016 | 0.0878 | 0.0471 | 0.1238 |
| 6018 | 0.0657 | -0.0215 | 0.2832 |
| 6028 | 0.1765 | 0.1011 | 0.1644 |

**Outcome:**
- All three conditions held in **3/6** donors (6013, 6018, 6028). T1 is treated as an artefact.
- Step 2 (the CNV/ASCL2 analysis) was **not run**, per the decision rule.

### 3.3 `scripts/ricci_exploratory_phase_genes.py` → `results/ricci_exploratory_phase_genes.txt`

This analysis is labelled EXPLORATORY.

**Setup:**
- Graph: malignant-only kNN graph per donor.
- κ_resid: residual of OLS `ORC ~ log deg + log n_genes + local_ID`.
- Gene correlations: partial Spearman over 3,517 genes.
- Enrichment: preranked GSEA.

**R² of `ORC ~ geometry`:**

| donor | R² |
|---|---|
| 6002 | 0.583 |
| 6009 | 0.310 |
| 6013 | 0.336 |
| 6016 | 0.291 |
| 6018 | 0.287 |
| 6028 | 0.353 |

**Outcome:**
- **Phase-portrait rule:** met in 0/6. The largest κ_resid ε² across clusters was 0.0457 (HTA8_6013).
- **Gene rule:**
  - (a) 9 consistent κ_resid genes, all negative, against a threshold of 20: **FAIL**. The genes are mostly ribosomal (RPS21, RPL36, RPS19, RPL37, RPS18, RPS15, RPL28) plus NOP53 and SNHG32.
  - (b) Spearman(κ_resid median rho, ID median rho) = −0.388: pass.
  - (c) Oxidative Phosphorylation, TNF-α/NF-κB, Wnt-β-catenin and EMT had FDR<0.05 for κ_resid and FDR≥0.25 for ID: pass.
- **Verdict:** "curvature reduces to local geometry; the Ricci line closes as a negative result."
- GO BP terms "cell-cell adhesion" and "actin cytoskeleton organization" were not found under exact names and were skipped.

### 3.4 `scripts/geom_ppca_potential.py` → `results/geom_ppca_potential.txt` (B1)

**Setup:**
- Model: PPCA on normal cells, Tipping & Bishop closed form.
- Decomposition: E = const + ½·M_in + ½·r²/σ². Checked numerically: max error ≤ 1.2e-04.
- q = 10, 20, 30.
- Two spaces: cohort PCA, and a "normal-only" space (R1).

**Results:**
- Cohort PCA, q=20: residual share of the median ΔE was 0.871–0.965 across donors, 6/6 pass.
- Normal-only space: 2/6 pass.
- B1 verdict as run: NOT SUPPORTED.
- **The R1 space was later judged invalid by design** (§4, bug 3).

### 3.5 `scripts/geom_principal_angles.py` → `results/geom_principal_angles.txt` (B2)

- B2a, cohort PCA: 6/6 donors had d(N,M) above both split-half nulls.
- B2a, normal-only space: 3/6 (this space is invalid).
- **B2b, cross-donor consistency of the top-3 "new directions":** observed 0.1284, null median 0.1068, null p95 0.1287. **NOT SUPPORTED.** Enrichment was therefore not interpreted.
- Gene weights saved to `results/geom_new_direction_gene_weights.csv`.

### 3.6 `scripts/geom_dnb.py` → `results/geom_dnb.txt` (B3)

**Setup:**
- DNB score: I = SD_in·PCC_in/PCC_out.
- Counts: binomially thinned.
- Bins: normal reference plus residual quintiles Q1–Q5 (terciles for HTA8_6002).
- Modules: 50 Hallmark sets.
- Test: 200-permutation family-wise test.

**Results:**
- **Transition peak:** best module was E2F Targets, peak in 1/6 donors (HTA8_6018, fw_p=0.005). NOT SUPPORTED.
- **Monotonic increase** (Spearman>0.8) in 5/6 donors: Glycolysis, Hypoxia, TNF-α/NF-κB, UV Response Up, mTORC1.
- Monotonic increase in 4/6 donors: Androgen Response, Apoptosis, Estrogen Response Early, Mitotic Spindle, Heme Metabolism, p53 Pathway.

### 3.7 `scripts/geom_r1_fullgene.py` → `results/geom_r1_fullgene.txt` (B1′, B2a′)

**Setup:**
- Residual and PPCA computed in full gene space, with no 50-D truncation.
- Gene sets:
  - G1: label-blind top-3000 HVG.
  - G2: genes expressed in ≥10% of cells.
  - G3: normal-only HVG, reference only.
- ΔE computed on means.
- Test A run on raw G1 and on G1 binomially thinned to the 10th-percentile depth.

| donor | share G1 | share G2 | share G3 | Test A raw G1: median-quintile ratio / OLS coef | Test A thinned G1: median-quintile ratio / OLS coef | (a) | (b) | (c) | B2a′ |
|---|---|---|---|---|---|---|---|---|---|
| 6002 | 1.000 | 0.999 | 1.001 | 1.19x (1/5 quintiles) / 5.2541 | no quintile usable / 5.1797 | Y | N | N | no |
| 6009 | 0.999 | 0.998 | 0.998 | 1.46x / 8.1382 | 1.43x / 5.8587 | Y | Y | Y | yes |
| 6013 | 0.995 | 0.992 | 0.994 | 1.39x / 4.6479 | 1.05x / 0.7159 | Y | N | N | yes |
| 6016 | 0.998 | 0.998 | 0.998 | 1.60x / 6.4950 | 1.34x / 4.5374 | Y | Y | Y | yes |
| 6018 | 0.981 | 0.931 | 0.920 | 1.27x / 3.5213 | 1.14x / 1.5493 | Y | N | N | yes |
| 6028 | 0.984 | 0.994 | 0.976 | 1.11x (3/5) / 2.0788 | 0.87x / −2.0329 | Y | N | N | yes |

**Outcome:**
- **B1′: 2/6, NOT SUPPORTED.**
- **B2a′: 5/6, SUPPORTED.**
- B2b was not retested.
- HTA8_6002 and HTA8_6028 already failed Test A on raw G1, before thinning. Only 6013 and 6018 went from passing (raw) to failing (thinned).

### 3.8 `scripts/geom_dnb_control.py` → `results/geom_dnb_control.txt` (B3′)

**Setup:**
- Each bin's module-score distribution was matched to Q1's (10 deciles, 20 reps).
- Criterion: Spearman(matched PCC_in, bin order) > 0.8.
- Confirmatory modules: 5.

**Donors passing, per module:**

| module | donors passing |
|---|---|
| Glycolysis | 1 (6013) |
| Hypoxia | 1 (6016) |
| TNF-α/NF-κB | 3 (6009, 6013, 6016) |
| mTORC1 | 2 (6013, 6016) |
| UV Response Up | 2 (6009, 6016) |

**Outcome:** 0/5 confirmatory modules pass B3′. Verdict: the monotonic increase is explained by heterogeneity in module activity.

HTA8_6002 used 3 bins of 11 cells each. With 3 bins, only Spearman = 1.0 exceeds 0.8.

### 3.9 `scripts/geom_thinning_diagnostic.py` → `results/geom_thinning_diagnostic.txt`

**Setup:** the same thinned counts were used for three diagnostics.
- **D1:** fresh label-blind HVG, then 50-D PCA, then held-out residual, then Test A.
- **D2:** noise-normalised effect and Cliff's delta, in 3000-D.
- **D3:** delta-method Poisson noise floor.

| donor | depth target | D1 median-quintile ratio | D1 OLS coef | D1 pass | D2 Cliff's δ | D3 noise share |
|---|---|---|---|---|---|---|
| 6002 | 1518 | no quintile usable | 7.8830 | NO | +0.8795 | 177.7% |
| 6009 | 1176 | 1.84x | 5.7793 | YES | +0.8465 | 259.0% |
| 6013 | 4056 | 2.43x | 7.1010 | YES | +0.9033 | 40.6% |
| 6016 | 1752 | 2.04x | 6.0427 | YES | +0.9131 | 126.9% |
| 6018 | 6363 | 3.69x | 9.1043 | YES | +0.9891 | 23.4% |
| 6028 | 2464 | 1.97x | 5.9900 | YES | +0.3701 | 68.9% |

**Outcome:**
- **D1: 5/6.** The pre-stated interpretation rule gives: "B1′(b) collapse most likely noise dilution in high dimension." B1′ remains NOT SUPPORTED.
- D3 values above 100% are flagged in the output as a failure of the delta-method approximation for sparse genes.
- D2 quintile-level estimates were mostly skipped because too few cells fell in each bin.

## 4. Bugs and design errors found and fixed

1. **Staging failure.** Staging the 224 MB h5ad to the cloud failed. Fix: the PCA was extracted on the device into a 2.6 MB npz (`extract_pca50.py`). Temporary split parts remain in `_ricci_tmp/crc_part_*` (the folder is 218 MB).
2. **svds broadcasting error.** The `LinearOperator` in `extract_pca50.py` caused a (25344×25344) allocation. Fix: `np.ravel` in matvec/rmatvec.
3. **R1 "normal-only 50-D" space invalid by design.** Projecting malignant cells into a PCA fitted on normal cells discards the out-of-subspace variance the residual measures. Documented in `results/NOTE_r1_invalid.txt`. The "cohort PCA leakage" claim was withdrawn.
4. **Non-additive ΔE shares.** Median-based shares are not additive (e.g. −4.25). Replaced by mean-based ΔE in 3.7.
5. **Gene identifiers.** Ensembl IDs were used where gene symbols were expected in the exploratory script. Fixed before the final run.
6. **σ² in the PPCA decomposition.** `geom_r1_fullgene.py` used a single σ² for normal cells that had been scored with per-fold σ². Fixed so the decomposition is exactly additive.
7. **Cosmetic.** Column spacing in the `ricci_t1_robustness` decision table was fixed in both the script and the output.

## 5. Decisions made

- The Ricci/ORC line is closed as a negative result (T2 0/6; T1 3/6; κ_resid gene rule failed).
- Step 2 of the ricci robustness plan (CNV/ASCL2) was not run. Relaxing the ≥4/6 rule to the 3 passing donors was declined.
- DNB is closed on both the transition-peak test (B3) and the monotonic-coordination test (B3′).
- B2b is closed (0.1284 < 0.1287) and will not be retested in another space.
- B1′ stays NOT SUPPORTED. D1 is reported as a diagnostic only.
- Next step agreed: replicate the frozen pipeline (thinning → label-blind HVG → 50-D PCA → q=20 held-out residual → Test A) on GSE161277 Patients 1–3. It is framed as a replication, not blind validation, because those patients were used during exploratory development. **Not yet run.** Neither `scripts/replicate_gse161277.py` nor its results exist.
- Prompts written for Claude Code are in English.

## 6. Open questions

- **D3 does not line up with the D1 interpretation.** The donors whose 3000-D thinned Test A failed after passing on raw data (6013, 6018) have the lowest noise-floor shares (40.6%, 23.4%). The donors flagged for high noise (6009, 6016) passed the 3000-D test. The "noise dilution" explanation is not directly supported by D3 for 6013 and 6018.
- **HTA8_6028:** thinned full-gene OLS coef = −2.0329 (p=2.3e-41), and D2 Cliff's δ = +0.3701, much lower than the other donors (0.85–0.99). Not explained.
- **HTA8_6002:** too few malignant cells remain after thinning (33) for quintile stratification. This donor cannot pass Test A as specified.
- **n_genes:** its role as a confound versus biology (e.g. stemness, RNA content) is unresolved. It is used as a covariate throughout.
- **Blind validation:** replication on a truly unseen CRC cohort is still needed. Candidates mentioned, accessions to be verified: GSE132465, GSE144735, GSE178341.
- **Housekeeping:** `_ricci_tmp/` (218 MB) should be cleaned up and added to `.gitignore`.
