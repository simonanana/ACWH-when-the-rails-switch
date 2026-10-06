# Pipeline and module map

All commands run from the repository root. Modules keep their original names so that they
match the paper's appendix: `u0x` are the main pipeline steps, `r1x`/`r2x` audit and
robustness modules, `h4x` hypothesis modules.

## Main pipeline — `u00`–`u10`

`src/run_upgrade.py` runs these in order. The default output directory is
`paper_tables_upgrade`; pass `--out outputs --tex-out outputs/tables` to keep output
separate from the committed `results/` and `tables/`. `--quick` uses small bootstrap and
permutation counts; `--only u03 u08` runs selected steps.

| Module | What it does | Paper | Inputs | Outputs |
|---|---|---|---|---|
| `u00_common.py` | Shared engine: data loading, QR-cached fixed-effects projection, CR1, cluster jackknife (CV3), restricted wild cluster bootstrap-*t* with Webb weights, three permutation schemes, Imbens–Manski bands | all | — | — |
| `u01_icsid_reference.py` | Rebuilds the 171-State ICSID Contracting-State reference table from the embedded ICSID/3 transcription | §3.4 | — | `data/reference/icsid_convention_status_icsid3.csv` |
| `u02_gsdb_treatment.py` | Builds the seven treatment definitions from the GSDB case file | §3.2, Table 3 | GSDB ✱, panel ✱ | `data/derived/gsdb_treatment_cohorts.csv`, `tab_treatment_family.csv`, `tab_switchers_gsdb.csv`, `tab_treatment_overlap.csv` |
| `u03_core_table.py` | Core routing table under four eligibility codings and five inference procedures; paired cluster bootstrap for the attenuation | §5.3–5.4, Tables 1 and 7 | cases, cohorts, reference | `tab_core_inference.csv`, `tab_attenuation_paired.csv` |
| `u04_interpretive_bounds.py` | Article 72 interpretive range over strict, permissive, 8 country-level and 3 × 1,000 case-level mixed codings | §5.6, Table 8, Figure 4 | same | `tab_interpretive_bounds.csv`, `tab_interpretive_assignments.csv` |
| `u05_influence.py` | Leave-one-respondent-State-out, with and without the eligibility control | §5.5, Figure 3a | same | `tab_influence_loo.csv`, `tab_country_effects.csv` |
| `u06_event_study.py` | Event study, Borusyak–Jaravel–Spiess imputation, Rambachan–Roth relative-magnitude bounds | §5.7, Table 9, Figure 3b | same | `tab_event_study.csv`, `tab_imputation.csv`, `tab_honest_did_forum.csv` |
| `u07_ovb_sensitivity.py` | Cinelli–Hazlett omitted-variable bounds benchmarked on eligibility | §5.8, Table 10, Figure 5 | same | `tab_ovb_sensitivity.csv` |
| `u08_spec_curve.py` | 224-specification curve with joint permutation inference, full and eligibility-controlled subsets | §5.12, Table 14, Figures 6–7 | same | `tab_spec_curve.csv`, `tab_spec_curve_joint.csv`, `spec_null_draws_*.csv` |
| `u09_incidence_etwfe.py` | Incidence: Wooldridge extended TWFE Poisson with a quasi-separation guard | §4, Table 5 | panel ✱, cohorts | `tab_incidence_etwfe.csv` |
| `u10_paper_tables.py` | Writes LaTeX table fragments from the CSVs above | — | `results/` | `tables/tab_u_*.tex` |
| `r27_reference_tables.py` | Treaty-layer decomposition and parallel-claims robustness under the reference definition, both outcomes; genuine-choice subsample. Composes existing functions only and checks each number against the paper | §5.10–5.11, Tables 11 and 13 | cases, merged Navigator file, cohorts, reference, treaty menus | `tab_two_layer_reference.csv`, `tab_parallel_claims_reference.csv`, `tab_parallel_claims_groupcluster.csv`, `tab_genuine_choice.csv` |
| `build_exhibits.py` | Headline table, switching-States table, two-panel core table, and Figures 1, 2, 3, 8, 9 | Tables 1, 7, 12; Figures 1–3, 8–9 | `results/`, cases, cohorts, reference | `figures/*.pdf`, `tables/*.tex` |

✱ Not redistributable; see `data/README.md`.

`build_exhibits.py` takes its paths as flags:

```bash
python src/build_exhibits.py --results results --cases data/cases_with_navigator.parquet \
    --cohorts data/derived/gsdb_treatment_cohorts.csv \
    --icsid data/reference/icsid_convention_status_icsid3.csv --src src
```

## Shared infrastructure

| Module | Role |
|---|---|
| `config.py` | Paths (`data/`, `outputs/`) and the legacy treatment threshold (`SANCTION_INTENSITY_THRESHOLD = 0.10`) |
| `io_load.py` | Loads the case file and the panel; builds legacy cohorts |
| `legal_eligibility.py` | Converts ICSID Contracting-State dates into case-level eligibility (Convention / Additional Facility / neither) and the enforcement regime |
| `r26_eligibility_audit.py` | Three-valued eligibility coding (open / contested / closed) and the falsification test against observed forum |
| `menu_logic.py` | Attaches treaty forum menus to cases |
| `_data.py` | Data entry point for `r22`, `r23`, `r25`; accepts `cohort_def=` to use a definition from the cohort file |
| `_compat.py`, `estimators.py` | Fixed-effects OLS, PPML, wild cluster bootstrap and permutation routines used by the `r` and `h` modules |
| `honest_did.py` | Rambachan–Roth moment-inequality implementation by linear programming |

## Supporting analyses

These need the country–year panel. Several resolve their default paths relative to `src/`
rather than the repository root, so pass `--panel` and `--cases` explicitly.

| Module | What it does | Paper | Command |
|---|---|---|---|
| `r13_bloc_choice.py` | Supply channel, volume margin: PPML structural gravity under three sanctioning-bloc definitions | §7 | `python src/r13_bloc_choice.py --panel data/analysis_panel_v4.parquet --cases data/cases_caselevel_final.parquet --out outputs` |
| `r14_forum_channels.py` | Supply channel, forum margin: claimant-bloc differential with respondent-by-year fixed effects | §7 | same flags as `r13` |
| `h4_moderation.py`, `h4b_moderation_rw.py`, `pnci_rebuild.py`, `r12_pnci_rebuild.py` | Alternative-rails moderators (CIPS, CPMI, RMB infrastructure, UNGA alignment, PNCI variants) with Romano–Wolf correction; PNCI normalisation audit | §8 | `python src/h4b_moderation_rw.py` (writes to `outputs/`); `r12` takes the same flags as `r13` |
| `r22_forum_eligibility.py` | Eligibility decomposition and audit (earlier formulation of §5.4, on the legacy index) | — | `python src/r22_forum_eligibility.py --panel data/analysis_panel_v4.parquet --out outputs` |
| `r23_claim_group.py` | Parallel-claim grouping (`detect_claim_groups`, `collapse_to_groups`, used by `r27`); its command line runs an earlier, legacy-index UNCITRAL version of the robustness table | §5.11 (via `r27`) | `python src/r23_claim_group.py --out outputs` |
| `r25_iia_choiceset.py` | Navigator merge (`--stage worklist`, which produced `data/cases_with_navigator.parquet`) and an earlier, legacy-index version of the two-layer decomposition (`--stage merge`); `attach_treaty_choice_set` is used by `r27` | §5.9–5.10 (via `r27`), §6 | `python src/r25_iia_choiceset.py --stage merge --navigator <Navigator.xlsx> --coding data/reference/iia_forum_menu_v4.csv --out outputs` |
| `feasibility.py` | Feasibility gate: coverage and switching-cluster checks run before estimation | §3.5 | imported by other modules |

`r26_eligibility_audit.py` runs standalone on the case file that `r22` produces:
`python src/r26_eligibility_audit.py --cases outputs/cases_with_eligibility.parquet --out outputs`.

## Reproduction status of each result

Checked by running this repository's code from a clean copy.

| Paper result | Status |
|---|---|
| Core routing table, attenuation (Tables 1, 7) | Reproduced to 10⁻¹⁴ from the data in this repository |
| Leave-one-out, event study, OVB bounds (Figures 3, 5; Tables 9, 10) | Reproduced to 10⁻¹³ from the data in this repository |
| Article 72 range, specification curve (Tables 8, 14) | Run from the data in this repository; point estimates deterministic, bands depend on draws |
| ICSID reference table | Rebuilt by `u01` with zero date disagreements |
| Post-award review, Figures 8–9 | Computed by `build_exhibits.py` from `data/cases_with_navigator.parquet` |
| §3.4 falsification (two remaining violations) | Reproduced by `r26`; `results/supporting/tab_elig_consistency_violations.csv` |
| §7 volume margin (−0.486, −0.646, −0.586; p = .237, .132, .128) | Reproduced exactly by `r13`; `results/supporting/b2_supply_by_bloc.csv` |
| §7 forum margin (+6.6, p = .644; −4.6, p = .610) | Reproduced exactly by `r14`; `results/supporting/f1_forum_channels.csv` |
| §8 Romano–Wolf family (33 tests, none survives; CPMI 0.498) | Reproduced exactly by `h4b`; `results/supporting/h4b_moderation_rw.csv` |
| Incidence (Table 5) | Committed output of `u09`; needs the panel to rerun |
| Treaty-menu decomposition (Table 11), genuine-choice subsample | Reproduced exactly by `r27` from the data in this repository (all 8 effects, standard errors and p-values; 281 cases, 23.1% vs 22.0%) |
| Parallel claims (Table 13) | Reproduced exactly by `r27` from the data in this repository (all 10 effects and standard errors; N = 958, 948, 958, 777, 666) |

`results/supporting/` holds the outputs of the supporting modules, generated from this
code with the full inputs.
