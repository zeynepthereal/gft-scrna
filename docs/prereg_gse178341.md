# Pre-registration: GSE178341 (Pelka et al. 2021, Cell)

Written 2026-10-10, before any residual or CNV analysis has been run on this dataset.
Data acquisition and inspection only preceded this document (see "Data inspection" below);
no hypothesis test in this document has been run yet.

## Dataset

GSE178341, colorectal cancer scRNA-seq, Pelka et al. 2021 *Cell*. 62 patients (28 MMRp,
34 MMRd), paired tumor and adjacent-normal tissue, 10x 3' scRNA-seq.

Files (the complete GEO supplementary set — confirmed against the FTP directory listing,
nothing else is published for this series):
- `GSE178341_crc10x_full_c295v4_submit.h5` — 1.1 GB, 10x CSR count matrix,
  43,113 genes x 370,115 cells (gene-major on disk; transpose on load).
- `GSE178341_crc10x_full_c295v4_submit_metatables.csv` — per-cell clinical/sample metadata
  (370,115 rows). Columns include `cellID`, `SPECIMEN_TYPE` (T/N), `PID` (patient ID),
  `MMRStatus` (MMRp/MMRd, defined for tumor rows), plus histology, staging, demographics.
- `GSE178341_crc10x_full_c295v4_submit_cluster.csv` — per-cell cluster annotation
  (370,115 rows, row-aligned with the metatable by `cellID`/`sampleID`). Columns:
  `clTopLevel` (7 lineages: Epi, TNKILC, Myeloid, Plasma, B, Strom, Mast), `clMidwayPr`
  (finer lineage), `cl295v11SubShort`/`cl295v11SubFull` (finest subtype, e.g. `cE01`
  = "Stem/TA-like").

## Data inspection (completed; no analysis beyond this)

- Total: 370,115 cells x 43,113 genes.
- `SPECIMEN_TYPE`: T=257,251, N=112,864. `MMRStatus` (per patient, from tumor rows):
  34 MMRd, 28 MMRp — matches the published cohort size exactly.
- Epithelial lineage (`clTopLevel=='Epi'`): 168,295 cells total (T=113,216, N=55,079),
  entirely covered by 11 subtypes `cE01`-`cE11` (Stem/TA-like, Stem/TA-like prolif,
  Immature Goblet, Enterocyte 1/2, Goblet, Goblet/Enterocyte, Best4, Tuft,
  Enteroendocrine). **These 11 subtypes are differentiation-state labels, not a
  malignant/non-malignant flag, and they are used identically for both tumor and normal
  specimens.**
- **No separate per-cell "malignant" classifier column exists anywhere in the three
  supplementary files.** All cluster-code prefixes were enumerated and cross-checked
  against `clTopLevel`: `cB*`=B, `cE*`=Epi, `cM*`=Myeloid, `cMA*`=Mast, `cP*`=Plasma,
  `cS*`=Strom, `cTNI*`=TNKILC. None is specific to malignant epithelium. This was
  cross-checked against the GEO FTP listing to confirm no fourth supplementary file
  (e.g. a CNV-based malignant call table) was omitted — these three files are the
  complete published set.
- **Consequence for this pre-registration:** wherever the task description below says
  "the authors' malignant-cell annotation," no such per-cell label exists in the public
  data. "Malignant" is therefore operationalized the same conservative way as
  `scripts/replicate_gse161277.py`'s C1 comparison: **malignant (proxy) = epithelial
  cells (`clTopLevel=='Epi'`) from TUMOR specimens (`SPECIMEN_TYPE=='T'`)**; **normal =
  epithelial cells from NORMAL specimens (`SPECIMEN_TYPE=='N'`)**. This is explicitly a
  proxy, not a ground-truth or even an authors' per-cell malignancy call, and it is
  known to be conservative (diluted by non-malignant epithelium in the tumor sample) in
  exactly the way C1 was for GSE161277. This substitution is stated here, before any
  results, specifically so it cannot be adjusted post hoc to make H2/H3 look better or
  worse.
- Per-patient normal/tumor epithelial counts (all 62 patients) are in
  `/private/tmp/claude-501/gse178341_per_patient.csv` (scratch; not committed). Patients
  with ≥50 normal epithelial AND ≥50 tumor-epithelial(-proxy) cells: **35/62** total —
  **20/34 MMRd**, **15/28 MMRp**. These 35 are the eligible-patient pool referenced below.

## Frozen pipeline

Identical to `scripts/replicate_gse161277.py` (itself frozen from
`scripts/geom_thinning_diagnostic.py` D1, validated 5/6 donors on the CRC epithelial
cohort). No parameter is changed for this dataset:
1. Binomial thinning of raw counts to a common depth per patient (target = 10th
   percentile of that patient's included cells' total counts; cells below target
   dropped).
2. `normalize_total` + `log1p` on thinned counts.
3. Label-blind top-3000 HVGs (`flavor='seurat'`), selected per patient on ALL of that
   patient's included cells (normal + malignant-proxy together, no group label used).
4. 50-D PCA fit per patient on those 3000 HVGs (label-blind, all included cells).
5. Normal-cell subspace, q=20, fit by SVD on normal cells only.
6. Normal cells: 5-fold held-out residual (seed 0). Malignant-proxy cells: residual
   against the full-normal-cell fit.
7. Test A (`confound_and_detector.test_a_sample`, imported unmodified): n_genes quintile
   stratification + OLS `residual ~ is_malignant + log(n_genes)`; "survives" = stratified
   median ratio > 1.2x AND OLS coefficient > 0 with p < 0.05.

## Hypotheses

**H1 (blind validation).** Malignant-proxy cells have higher held-out-normal-referenced
residual than held-out normal cells, across GSE178341 as a whole (both MMR strata
pooled). *Supported if Test A's "survives" rule passes in ≥70% of the 35 eligible
patients.*

**H2 (MMRd).** Among MMRd patients, residual separates authors'-proxy-malignant from
normal epithelial cells better than an infercnvpy CNV score does (AUROC comparison).
Rationale: MMRd (hypermutated) tumors are not expected to carry the same burden of large
copy-number alterations as MMRp tumors, so a CNV-score classifier is expected to be
comparatively weak there — this is the dataset's built-in test of whether the residual
captures something beyond CNV burden. *Supported if residual AUROC > CNV-score AUROC in
≥70% of eligible MMRd patients (of the 20 MMRd patients in the eligible pool) AND the
median AUROC difference (residual − CNV) across those patients is > 0.05.*

**H3 (control).** Among MMRp patients, CNV-score AUROC is expected to be high (MMRp CRC
is CNV-driven). Residual vs CNV-score AUROC will be reported for comparison, with no
pass/fail threshold — this is a sanity check on the CNV score itself, not a test of the
residual.

**Baselines, for every AUROC reported in H2/H3:** `n_genes_by_counts`, and Euclidean
distance to the normal-cell centroid (same two baselines used throughout this project).

## What "CNV score" means here

`infercnvpy.tl.infercnv` + `infercnvpy.tl.cnv_score`, run per patient, same convention as
`scripts/cnv_clone_vs_residual.py`: reference = normal epithelial cells (and, where
available, non-epithelial/immune cells from the same patient); observation = malignant-
proxy cells. Window size and other infercnvpy parameters will be fixed before H2/H3 are
run and reported alongside the results, not tuned afterward.

## Ground-truth caveat (stated before any results)

Both "malignant" labels used in this study — the tumor/normal-proxy label here, and the
CNV score computed for the AUROC comparison — are derived from the same or closely
related single-cell expression data as the residual itself. **The authors' own cell
lineage/subtype classifier (`clTopLevel`/`cl295v11SubShort`) is also expression-based,**
not DNA-level ground truth (no matched WES/WGS malignant calls are part of this public
release). All AUROC and Test A results in this line of work, including H1–H3, are
internal-consistency checks between different expression-derived measurements, not
validation against an independent (e.g., genomic) ground truth. This applies with extra
force here since there is no authors' per-cell malignant flag at all (see "Data
inspection" above) — H2/H3's "malignant" group is one proxy (tumor-sample epithelium)
compared against another expression-derived score (CNV), not two independent measurements.

## Commit scope

This document and `.gitignore` only. `data/GSE178341/` is excluded via `.gitignore` and
was not and will not be committed. No residual or CNV analysis script for this dataset
has been written yet.
