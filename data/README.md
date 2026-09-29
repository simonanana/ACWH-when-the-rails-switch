# Data

This directory contains only files that can be published. Two inputs used in the paper
cannot be redistributed and must be obtained separately; they are listed at the end.

## Included

### `cases_caselevel_final.parquet` — 958 rows × 24 columns

Every treaty-based investor–State dispute initiated between 2010 and 2023 in the UNCTAD
Investment Dispute Settlement Navigator (full data release as of 31 December 2023), one row
per case. Key fields: `year` (year of initiation), `respondent_iso3`, `claimant_iso3`,
`rules`, `institution`, `icsid_flag`, `admin_class` (Western / non-Western / ad hoc),
`amount_claimed_usd`, and keyword flags for payment-barrier and expropriation claims.
The `seat_*` fields are recorded for only 65 cases (6.8%) and are not used as outcomes.

Source: UNCTAD, Investment Dispute Settlement Navigator,
<https://investmentpolicy.unctad.org/investment-dispute-settlement>. Reorganised and coded by
the author. Please cite UNCTAD as the original source.

### `cases_with_navigator.parquet` — 958 rows × 54 columns

The same 958 cases merged back to the Navigator release to recover the applicable
international investment agreement (`iia`), the outcome of the original proceedings, and
follow-on proceedings (`fo_annulment`, `fo_setaside`, `fo_upheld`, `fo_annulled_full`,
`fo_annulled_part`, `fo_pending`, …). Produced by `src/r25_iia_choiceset.py --stage worklist`.
Used for the post-award analysis and the treaty-menu layer. All 958 cases matched.

### `reference/icsid_convention_status_icsid3.csv` — 171 rows

ICSID Contracting-State status by date: `signature`, `in_force_from`, `out_of_force_from`
(for the denunciations by Bolivia, Ecuador and Venezuela), with a `source` for every row.
Transcribed from the official list *ICSID/3, List of Contracting States and Other
Signatories of the Convention* (as of 9 June 2020), supplemented by ICSID announcements for
later accessions and by *ICSID/8* (28 October 2022) for territorial designations.
`src/u01_icsid_reference.py` regenerates this file from the transcription embedded in the
script.

On territories: Hong Kong, Bermuda and Macao are coded as non-parties, which is the
conservative choice. Under Article 70 the better view is that Hong Kong claimants are
nationals of a Contracting State (China filed no exclusion and designated the HKSAR under
Article 25(1) on 26 August 2022); switching the coding affects two control cases and moves
the eligibility-conditioned estimates by about 0.1 percentage points. The file's
`status_note` column documents this.

### `reference/iia_forum_menu_v4.csv` — 54 treaties

For each coded investment treaty, the dispute-settlement forums its investor–State clause
offers (ICSID Convention, Additional Facility, UNCITRAL, SCC, ICC, other). Coverage was
prioritised by identifying weight: the 54 treaties cover most identifying cases but only a
minority of the 470 distinct instruments in the case file.

Read the provenance columns before relying on any row:

| `coder` | `verification_level` | Meaning |
|---|---|---|
| `derived` | 0 | Observational lower bound: a forum is recorded as offered because a case under the treaty actually used it |
| `LLM_assisted` | 1 | Drafted with a large language model from the treaty text; not independently verified |
| `LLM_assisted` | 2 | Checked by a human coder against a cited primary source |

Five entries are at level 2, twenty-seven at level 1 and twenty-two at level 0. Results that
depend on this file are reported in the paper as limited by the coding.

### `derived/gsdb_treatment_cohorts.csv` — 170 economies

Onset year of financial-sanctions treatment for each economy under each of the seven
treatment definitions in the paper (`bloc1_k2` is the reference definition: at least two of
the EU, United States, United Kingdom, Switzerland and Norway imposing financial sanctions,
the EU counted as one). A missing value means the economy is never treated under that
definition inside the window; the `*_left_censored` columns flag economies already treated
before 2010.

This file contains **onset years only** — a derived research variable, not Global Sanctions
Data Base records. It is produced by `src/u02_gsdb_treatment.py` from the GSDB, and is
everything the core routing analysis needs from the sanctions data.

## Not included — obtain separately

### Global Sanctions Data Base (GSDB), Release 4

`Global_Sanctions_Data_Base__GSDB__2023.csv`, the case-level file. Required only to rebuild
the treatment cohorts with `u02_gsdb_treatment.py`.

The GSDB is free for non-commercial research but is distributed individually on request, and
its terms ask users **not to pass the data on**. Request it from
<https://www.globalsanctionsdatabase.com/> and place the file at
`data/Global_Sanctions_Data_Base__GSDB__2023.csv`. Please cite:

- Felbermayr, G., Kirilakha, A., Syropoulos, C., Yalçın, E. and Yotov, Y. V. (2020). The Global Sanctions Data Base. *European Economic Review* 129, 103561. <https://doi.org/10.1016/j.euroecorev.2020.103561>
- Yalçın, E., Felbermayr, G., Kariem, H., Kirilakha, A., Kwon, O., Syropoulos, C. and Yotov, Y. V. (2025). The Global Sanctions Data Base—Release 4: The Heterogeneous Effects of the Sanctions on Russia. *The World Economy* 48(9), 2003–2017. <https://doi.org/10.1111/twec.13732>

For the same reason three files that `u02` writes alongside the cohorts file are not
included, because they contain GSDB case identifiers, sanction objectives or entity names:
`derived/gsdb_fin_country_year.csv`, `derived/gsdb_name_iso3_crosswalk.csv` and
`results/tab_switchers_gsdb.csv`. The paper's Appendix Table 17 reports the senders and GSDB
case identifiers for the twelve switching States under the reference definition.

### Country–year panel

`analysis_panel_v4.parquet` (2,380 rows = 170 economies × 14 years, 243 columns). Required by
the incidence analysis (`u09`), by `r22`, `r23` and `r25` through `_data.py`, and by the
supply-channel and alternative-rails modules (`r12`, `r13`, `r14`, `h4`, `h4b`).

It is not published because it combines series from many third-party sources with their own
terms — among them GSDB-derived sanctions intensities, World Bank macro indicators, SWIFT
RMB Tracker figures, CIPS participation, UN General Assembly voting ideal points and a
Chainalysis index — and most of its columns are not used in this paper. The columns the
included code reads can be identified from `src/io_load.py`, `src/config.py` and the modules
listed above.

### UNCTAD Navigator Excel release

Only needed to re-run `src/r25_iia_choiceset.py --stage worklist`, which produced
`cases_with_navigator.parquet`. It is freely downloadable from UNCTAD (full data release as
of 31 December 2023, Excel format).
