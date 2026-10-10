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

---

## Amendment 1, before any analysis (2026-10-10)

Written after the original pre-registration above, still before any residual, CNV, or
AUROC analysis has been run on GSE178341. Nothing above is deleted or altered; this
section adds channel/chemistry controls that the original pre-registration omitted.

### Why

Pelka et al.'s malignant-cell classifier was trained on **source channel** (tumor-channel
vs. adjacent-normal-channel), not on a per-cell molecular malignancy signature, and ~95%
of tumor-channel epithelium was called malignant. Our own "malignant" proxy (established
in the original pre-registration, since no authors' per-cell label exists at all) is
`SPECIMEN_TYPE=='T' & clTopLevel=='Epi'` — i.e., it is *also* channel of origin. Any
technical difference between a tumor channel and its paired normal channel (processing
run, 10x chemistry, capture batch) can therefore inflate "malignant vs normal" residual
or AUROC without reflecting disease biology, and would mechanically favor an
expression-based score (the residual) over a CNV-based score, since CNV is specifically
built to be robust to non-genomic technical variation. This amendment adds the controls
needed to tell these apart before any such analysis is run.

### STEP 1 — Channel and chemistry inspection (report only, no analysis run)

Channel identity: `batchID` in the cluster-annotation file (format
`{PID}_{T|N}_..._c{capture}_v{chemistry}`), 180 channels total across 62 patients.
Chemistry: the `SINGLECELL_TYPE` metatable column (`SC3Pv2` / `SC3Pv3`); cross-checked
against the `v2`/`v3` suffix of every `batchID` — **the two sources agree on all 370,115
cells with no exceptions.**

Per-patient channel listing (tissue, chemistry, epithelial-cell count per channel) is
saved to `/private/tmp/claude-501/gse178341_channels.csv` (scratch, not committed; 180
rows, one per channel) and the per-patient summary to
`/private/tmp/claude-501/gse178341_patient_channel_summary.csv` (62 rows). Key findings:

- **Same chemistry on both sides.** 35/62 patients have at least one T and one N channel;
  of those, **35/35 use a single, matching chemistry for every T and every N channel**
  except **C162**, whose T channels use `{SC3Pv2, SC3Pv3}` (mixed) and whose N channels
  also use `{SC3Pv2, SC3Pv3}` (mixed) — present in both chemistries on both sides, but
  not a clean single-chemistry match, so it is excluded from the "same chemistry" count
  below out of caution (per-channel analysis could still use it if a future script
  explicitly restricts each comparison to a single shared chemistry, but it is not
  counted as eligible here). All 27 remaining patients have a T channel but no N channel
  at all (chemistry match is undefined, not violated, for these — they were already
  excluded from the original pre-registration's eligible pool for lacking normal cells).
  **No patient in this dataset has tumor and normal channels on genuinely different,
  non-overlapping chemistries** — the chemistry confound is present as a possibility in
  principle (SC3Pv2 vs SC3Pv3 differ across the cohort) but does not co-occur with the
  T/N split within any single patient's paired channels, apart from C162's ambiguous case.
- **Multi-channel patients with >=50 epithelial cells per channel** (needed for the
  channel-effect control, C-ctrl, below):
  - **>=2 normal channels, each >=50 epithelial cells: 15/62 patients (8 MMRd, 7 MMRp).**
  - >=2 tumor channels, each >=50 epithelial cells: 44/62 patients (22 MMRd, 22 MMRp)
    (reported for completeness; C-ctrl as specified uses normal channels).

### STEP 2 — Pre-registration amendments

**Ground truth (supersedes the "Ground-truth caveat" section above, does not delete it).**
Labels in this dataset are channel of origin: our malignant proxy is tumor-channel
epithelium, and the authors' own malignant classifier was *trained on* tumor-channel vs.
normal-channel origin. The authors' classifier is therefore **not independent of channel
effects** — a technical difference between a patient's tumor and normal channel (capture
run, chemistry, any batch effect) can be absorbed into "malignant" status by
construction, for both our proxy and the published classifier. No result in this dataset
should be read as validating the residual against molecularly-defined malignancy; at
best it can show the residual separates tumor-channel from normal-channel epithelium,
which is a weaker and more confoundable claim.

**Eligibility (supersedes the original "35/62" eligible-patient pool).** Patients are
eligible for H1/H2/H3 only if their tumor and normal epithelium come from the same 10x
chemistry. Applying this on top of the original >=50-normal-epithelial AND
>=50-tumor-epithelial(-proxy) requirement: **34/62 patients remain eligible (20 MMRd, 14
MMRp)** — only C162 is newly excluded (its T/N chemistry split is ambiguous, see above);
every other patient that met the original >=50/>=50 criterion already had matching T/N
chemistry.

**Channel-effect control (C-ctrl), for the 15 patients with >=2 normal channels each
with >=50 epithelial cells (8 MMRd, 7 MMRp) — to be run together with H1/H2/H3, not
before:**
1. Fit the normal subspace (q=20, same frozen pipeline) on normal channel A only.
2. Held-out residual for channel A cells (5-fold CV, as in the frozen pipeline).
3. Residual for normal channel B cells, scored against the channel-A fit (no held-out
   needed for B since it was never in the training fit).
4. Channel ratio = median(residual, channel B) / median(held-out residual, channel A).
5. Residual AUROC for separating channel B from channel A (channel B = positive class).
   Baselines (n_genes, Euclidean-to-normal-centroid) reported alongside, as elsewhere.
6. When a patient has more than 2 qualifying normal channels, every ordered pair (A, B)
   is used (same convention as `scripts/heldout_crossdonor_control.py`'s cross-donor
   test), pooled into one channel-ratio / channel-AUROC distribution across all 15
   patients for the thresholds below.

**H1 (revised).** Supported only if, in >=70% of the 34 eligible patients: (a) Test A
passes (unchanged from the original), **AND** (b) the tumor/normal residual ratio for
that patient exceeds the 95th percentile of the pooled C-ctrl channel-ratio distribution
(i.e., the tumor-vs-normal deviation must clearly exceed what two normal channels from
the same patient already show each other, not just be numerically greater than 1).

**H2 (revised).** The residual-vs-CNV AUROC comparison in MMRd patients is interpretable
only if the **median C-ctrl residual AUROC (normal channel B vs. normal channel A,
across the pooled 15-patient distribution) is below 0.6**. If the median C-ctrl AUROC is
>=0.6, H2 is declared **untestable in this dataset**, regardless of what the
malignant-vs-normal residual/CNV AUROC numbers turn out to be — a residual that already
separates two normal channels from the same patient at AUROC>=0.6 cannot be trusted to
report a malignant-vs-normal AUROC as a disease signal. The original H2 thresholds
(residual AUROC > CNV AUROC in >=70% of eligible MMRd patients, median difference >0.05)
apply only if this untestability gate is cleared first.

**H3.** Unchanged (control comparison, no pass/fail threshold); will additionally report
where each MMRp patient's tumor/normal residual ratio falls relative to the C-ctrl
channel-ratio distribution, for the same reason as H1(b).

**All other pipeline parameters remain frozen** exactly as in the original
pre-registration (thinning to the 10th percentile, 3000 label-blind HVGs, 50-D PCA,
q=20, 5-fold held-out, seed 0, Test A as implemented in
`confound_and_detector.test_a_sample`). Nothing in this amendment changes those
parameters; it adds a control and two gating conditions on top of them.

### Commit scope (Amendment 1)

This amendment, appended to `docs/prereg_gse178341.md` only. No residual, CNV, or AUROC
script has been written or run for GSE178341 — C-ctrl, H1, H2, and H3 are still
descriptions of analyses to be run, not results.
