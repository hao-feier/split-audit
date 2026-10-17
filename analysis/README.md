# analysis/

Post-hoc analysis, table & figure generation, claim verification. 94 scripts,
flat by design (referenced by name from the root `README.md` and the paper's
reproduction commands); grouped below by role. Run from the repo root.

## Claim verification & manuscript consistency (6)

- `abstract_words.py` — Word count for the MDPI abstract, counting a math span as ONE word.
- `bib_audit.py` — Audit paper/refs.bib entries against CrossRef.
- `dump_audit.py` — Audit helper: dump ground-truth aggregates from results/metrics_summary.json
- `experiment_inventory.py` — Full inventory of every experiment run for this study.
- `verify_analysis_index.py` — Audit results/README.md: every row must point at a real script and a real JSON.
- `verify_manuscript_numbers.py` — Verify every headline number in the manuscript against the result JSONs.

## Table generation (paper LaTeX blocks) (7)

- `aggregate_results.py` — Stage 5 (PLAN 6): aggregate all per-(model, seed) metrics into paper tables.
- `make_factorial_tables.py` — Regenerate tab:factorial in place in paper/main.tex from results/uniform/ir_factorial.json (also rewrites the rand-sweep table if present).
- `make_latex_tables.py` — Emit the main result tables (tab:main, tab:rand) from results/metrics_summary.json into the AUTO-GENERATED marker block of paper/main.tex.
- `make_mosaic_summary.py` — Propagate the de-duplicated (mosaic) evaluation into the table pipeline.
- `make_random_split_comparison.py` — Build <root>/random_split/comparison.json for tab:rand (uniform protocol).
- `make_scene_table.py` — Emit the per-scene acquisition table (tab:scenes) into the marked block of paper/main.tex, from results/scene_dates_tides.json.
- `make_supplementary_tables.py` — Emit the appendix supporting tables into the marked block of paper/main.tex (from results/metrics_summary.json and results/leave_one_cluster_out.json).

## Figure generation & QA (7)

- `ascii_view.py` — ASCII luminance preview of a PNG (for text-only QA of figures).
- `export_figure_inputs.py` — Export the fig1 zone input (results/fig1_zones.json) that make_figures.py reads.
- `figstyle.py` — Shared publication-quality matplotlib style for all paper figures.
- `figure_qa.py` — Re-measure the real textwidth and fail if a figure canvas drifts from it (figure QA gate).
- `make_figures.py` — Generate all seven paper figures (vector PDF + 300-DPI PNG).
- `make_roc_pr_curves.py` — Compute ROC/PR curves + pAUC@FPR<=0.01 for honest vs leaky arms (inference only).
- `plot_distance_response.py` — Plot fig_distance_response from results/distance_response.json.

## GT interpretation workbench (2,400-pt field pass) (6)

- `gt_interpret_assemble.py` — Assemble interpretation.json from per-montage judgments.
- `gt_interpret_crops.py` — Build interpretation montages for the 2400 GT points.
- `make_groundtruth_samples.py` — stratified random sample design for visual interpretation.
- `make_gt_tool.py` — Build the ground-truth interpretation workbench (gt_tool/).
- `merge_gt_interpretation.py` — Merge interpreter output (gt_tool/interpretation.json) into the sample GeoJSONs.
- `oloofsson_area_accuracy.py` — Olofsson et al. (2014) area-based accuracy estimation.

## E3 overlap-free block controls (12)

- `aggregate_block_split.py` — Aggregate the R1 non-overlapping block counterfactual (M3-7a).
- `e3b_analyze.py` — E3-B analysis: overlap-free block-interleaved contrast, 16 stratified splits.
- `e3b_composition_penalty.py` — E3-B composition penalty C -- measurement protocol and PRE-COMMITTED USE RULE.
- `e3b_make_splits.py` — E3-B: emit the 16 locked block-interleaved splits + their matched controls.
- `e3b_preflight.py` — E3-B pre-flight: matched-arm feasibility + locked candidate list.
- `e3b_repair_matched.py` — E3-B defect-1 repair: redraw the matched-arm test set disjoint from the block test set.
- `e3c_analyze.py` — Aggregate E3-C: does residual proximity matter when overlap is zero?
- `e3c_extract_256.py` — E3-C: a full-scale, overlap-free interleaved control.
- `e3c_make_splits.py` — E3-C split list: scattered vs separated, both overlap-free, size-matched.
- `e3c_roc_gap.py` — Prevalence-free re-scoring of the E3-C proximity control.
- `footprint_overlap.py` — Geometric audit of the 2x2 factorial claim: edge distance and footprint overlap of the random test patches relative to the training zone.
- `nonoverlap_eval.py` — evaluation on the NON-OVERLAPPING patch subset.

## Inflation & decomposition (9)

- `aggregate_ir_factorial.py` — Aggregate the I/R factorial cell sweep into results/ir_factorial.json.
- `argmax_floor_effect.py` — The F1 = 0 runs are a decision-rule artifact, not training failures.
- `collapse_reconcile.py` — Collapse reconciliation for the split-protocol sweep.
- `decompose_f1_gap.py` — Re-derive the honest-vs-leaky change-F1 decomposition (Fig. 5 / Sec. prevalence).
- `inflation_decomposition.py` — same-test-set inflation decomposition.
- `loss_gain_decomposition.py` — Per-class split-protocol contrast (loss vs gain) into results/loss_gain_decomposition.json.
- `protocol_factorial.py` — B factorial: controlled single-scenario evaluation-protocol effect (design-based).
- `random_split_contrast.py` — Compare the spatially-isolated protocol (exp3) vs the random-split control.
- `random_split_sweep.py` — robustness of the spatial-shortcut quantification across 10 independent random split seeds.

## CIs, resampling & discrimination controls (11)

- `ap_exact_recompute.py` — Exact AP at target prevalence for k1/k2/k3 splits (90-run set).
- `ap_pr_auc.py` — prevalence-invariant ranking control (AP/PR-AUC arm).
- `block_bootstrap.py` — cluster / spatial-block bootstrap CIs.
- `bootstrap_cell_sensitivity.py` — Cluster-bootstrap sensitivity to the spatial cell size (why 1024 px?).
- `bootstrap_ci.py` — patch-level bootstrap CIs + paired model comparisons.
- `cluster_bootstrap_v2.py` — conservative cluster bootstrap + multiple-comparison correction + seed-replication audit.
- `dump_predictions.py` — dump per-patch argmax predictions for bootstrap CI.
- `prevalence_control.py` — prevalence control.
- `roc_auc_prevalence.py` — Prevalence-free discrimination check for the split-protocol contrast.
- `split_replicate_ci.py` — Partition-level replication CIs for the spatial-interleaving change-F1 gap (10 partitions).
- `threshold_f1_gap.py` — Threshold/FPR-optimized change-F1 gap: interleaved vs buffer-separated arms.

## Sensitivity sweeps & confounders (36)

- `area_uncertainty.py` — uncertainty on the reported mangrove area-change figures.
- `audit_regions.py` — region vs patch-domain statistics, split fractions, distances.
- `autocorr_decay.py` — Change-label spatial autocorrelation decay (mechanism figure).
- `boundary_dongzhai.py` — Boundary-distance / reference-resolution analysis for all test regions including Dongzhai Bay.
- `boundary_tolerance.py` — E7: boundary-tolerance sweep -- the direct test of the label-resolution tax.
- `fringe_band_reference_error.py` — Reference-error x tested-pixel-distribution channel (Limitations): fraction of each arm's change pixels in the 30 m GMW fringe band.
- `buffer_sensitivity.py` — buffer-width sensitivity of the within-region (honest) numbers.
- `cross_region_degradation.py` — PLAN 7: quantify cross-region generalization drop.
- `cw_compare.py` — class-weight sensitivity, uniform (none) vs sqrt, full sweep.
- `cw_schemes.py` — class-weight scheme comparison (inv / sqrt / uniform).
- `deterministic_audit.py` — deterministic re-run -> noise floor of the training pipeline.
- `diag_seed2026.py` — Why do 4 architectures predict exactly zero change pixels on the isolated test set of split k3 with run-seed 2026, while training looked healthy?
- `distance_covariates.py` — distance-vs-patch-difficulty control.
- `distance_covariates_clustered.py` — cluster-robust inference for the distance covariate.
- `dualsource_boundary_strat.py` — Second-source check stratified by distance to the reference mangrove fringe.
- `edge_band.py` — Edge-band analysis: quantify how much of the model's predicted change sits in the 10-30 m boundary band around the reference mangrove fringe.
- `error_autocorr.py` — spatial autocorrelation of the MODEL ERROR field.
- `eval_distance_response.py` — Distance-response under the honest (spatially separated) protocol.
- `eval_dongzhai.py` — Evaluate existing uniform-sweep checkpoints on the Dongzhai Bay test region.
- `gmw2020_v4112.py` — extract GMW v4.1.12 2020 layer (stack band 36) + three-way source-vs-time agreement.
- `independent_validation.py` — PLAN 2.2: independent-source validation.
- `label_noise_sensitivity.py` — Label-noise sensitivity of the interleaved-vs-honest change-F1 gap.
- `leave_one_cluster_out.py` — leave-one-spatial-cluster-out sensitivity.
- `mislocalization.py` — Mislocalization analysis: distance of missed / spurious change to the model's own (or the reference's) change pixels.
- `mosaic_eval.py` — evaluate on a de-duplicated mosaic, not on the patch union with 50% overlap.
- `nested_cv_cw.py` — nested-CV class-weight scheme selection.
- `prior_shift.py` — Class-prior shift diagnostic: quantify how much cross-regional degradation is driven by the difference in change prevalence between training and test regions.
- `prior_shift_oos.py` — Out-of-sample within-region threshold gain.
- `radiometric_consistency.py` — Radiometric consistency between the two epochs, beyond a single deep-water pair.
- `resolution_control.py` — label-resolution control experiment + edge-band null baselines.
- `scene_dates_tides.py` — acquisition dates + tidal-phase metadata.
- `scl_frac.py` — SCL classes 1-3 (saturated/dark/shadow) kept as "clear" contribute a fraction of possibly-contaminated observations to the composites.
- `split_audit.py` — Split-geometry audit.
- `strict_scl_sensitivity.py` — strict-SCL-mask sensitivity.
- `zhanjiang_subset.py` — robustness of the model ranking where the reference is least reliable.
- `zone_covariates.py` — Zone-level covariate comparison (is the test zone simply harder?).
