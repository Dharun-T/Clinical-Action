# MIMIC-IV Step 2 Data Dictionary

**Status:** Schema review of the supplied MIMIC-IV Clinical Database Demo 2.2.
This document contains no patient values or experimental results. The demo is
a 100-patient development subset and must not be presented as the full
credentialed MIMIC-IV research dataset.

## Access boundary

MIMIC-IV is marked **Credentialed Access** on PhysioNet. Before using it, the
researcher must create a PhysioNet account, complete the required human
subjects and data-use training, and obtain the dataset access agreement. The
student should also follow their college supervisor's ethics and research
approval process. Do not download or copy restricted data into this repository
before authorization.

Official sources:

- MIMIC-IV on PhysioNet: https://physionet.org/content/mimiciv/
- Hospital module documentation: https://mimic.mit.edu/docs/iv/modules/hosp/
- ICU module documentation: https://mimic.mit.edu/docs/iv/modules/icu/

## Candidate tables

| Table | Purpose | Key identifiers and joins | Relevant timestamps | Important fields | Known issues and limitations |
|---|---|---|---|---|---|
| `hosp.labevents` | Patient laboratory results, including chemistry and hematology | `subject_id`; `hadm_id` when assigned; `itemid` joins `hosp.d_labitems` | `charttime`, `storetime` | `labevent_id`, `specimen_id`, `itemid`, `value`, `valuenum`, `valueuom`, `ref_range_lower`, `ref_range_upper`, `flag`, `priority`, `comments` | Results can be inpatient or outpatient/ED. `hadm_id` is not present for every observation and may require time-aware joins. Duplicate-looking rows, repeated tests, textual values, flags, and unit differences require inspection. |
| `hosp.d_labitems` | Definitions for laboratory `itemid` values | `itemid` joins `labevents.itemid` | None | `itemid`, `label`, `fluid`, `category`, `loinc_code` where available | Labels and mappings must be inspected in the authorized version. One clinical concept may have multiple item IDs or specimen contexts. |
| `hosp.patients` | Deidentified patient demographics | `subject_id` joins `admissions`, `labevents`, and diagnoses | `anchor_year`, `anchor_year_group`, `dod` | `subject_id`, `gender`, `anchor_age`, `anchor_year`, `anchor_year_group` | Dates are deidentified and shifted consistently within a patient. Age is anchored and should be interpreted according to MIMIC documentation, not as exact birth data. |
| `hosp.admissions` | Hospital admission and discharge context | `subject_id`; unique hospital visit key `hadm_id` | `admittime`, `dischtime`, `deathtime`, `edregtime`, `edouttime` | `subject_id`, `hadm_id`, `admission_type`, `admission_location`, `discharge_location`, `insurance`, `language`, `marital_status`, `race` | One patient can have multiple admissions. Organ-donor records and unusual lengths of stay require review. |
| `hosp.diagnoses_icd` | Diagnoses assigned to a hospital admission | `subject_id`, `hadm_id`; ICD code joins dictionary tables | No event timestamp in the core table | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` | A diagnosis code is not necessarily a diagnosis made before the prediction time. It must not be used unless its availability timing is defensible. |
| `hosp.prescriptions` | Medication prescription orders | `subject_id`, `hadm_id`; medication identifiers and order fields | `starttime`, `stoptime`, `entertime` | `drug`, `formulary_drug_cd`, `dose_val_rx`, `dose_unit_rx`, `route`, `poe_id`, `poe_seq` | A prescription order is not proof of administration or of a response to a laboratory result. Start/stop changes need a precise operational definition and temporal exclusion rules. |
| `hosp.emar` / `hosp.emar_detail` | Medication administration records and details | `subject_id`, `hadm_id`, `emar_id`; medication identifiers and order links | `charttime`, `scheduletime`, `storetime` where available | Medication, administration status, route, dose, `poe_id` where available | Administration is stronger evidence of an event than a prescription, but it still does not prove the lab caused the medication event. |
| `hosp.poe` / `hosp.poe_detail` | Provider order entries and order details | `subject_id`, `hadm_id`, `poe_id`; detail rows join by `poe_id` | `ordertime` | `poe_id`, `poe_seq`, `order_type`, `order_subtype`, `order_status`, `transaction_type` | Order categories and lab-order linkage must be audited in the actual release. An order near a lab does not prove that the lab caused it. |
| `hosp.procedures_icd` | ICD-coded hospital procedures | `subject_id`, `hadm_id`; code joins procedure dictionary | `chartdate` | `subject_id`, `hadm_id`, `icd_code`, `icd_version`, `chartdate`, `seq_num` | Date-level timing may be too coarse to establish that a procedure followed a specific lab report. |
| `icu.icustays` | ICU stay definitions | `subject_id`, `hadm_id`, unique `stay_id` | `intime`, `outtime` | `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `last_careunit` | Use only when ICU context is needed. Patients and admissions can have multiple stays. |
| `icu.chartevents` | ICU charted measurements and observations | `stay_id` and `itemid`; `itemid` joins `icu.d_items` | `charttime`, `storetime` | `subject_id`, `hadm_id`, `stay_id`, `itemid`, `value`, `valuenum`, `valueuom` | Very large event table. It may provide vital signs, but concepts and units must be selected from `d_items` after access. |
| `icu.inputevents` / `icu.procedureevents` | ICU inputs and documented ICU procedures | `stay_id`, `itemid` and event identifiers | Event-specific start/end or chart timestamps | Event IDs, `stay_id`, `itemid`, values, rates, and times | These are measurable events, but attribution to a future laboratory report is not automatic. |

## Five laboratory concepts

The required tests are **Glucose, Creatinine, Potassium, Sodium, and
Hemoglobin**. The implementation must not guess numeric `itemid` values. After
authorized access, inspect `d_labitems` and create a versioned mapping using
the actual `itemid`, `label`, and `fluid` and `category`, plus `loinc_code` only
if it exists in the selected release. The supplied 2.2 demo has four
`d_labitems` columns and no `loinc_code`. Chemistry and blood-gas measurements may have different units
or specimen types.

For every selected item, the audit must record:

1. Exact label and `itemid`.
2. Specimen/fluid and category.
3. Whether `valuenum` is usable.
4. Observed `valueuom` values and any conversion rule.
5. Missing, nonnumeric, flagged, and duplicate-record counts.
6. The timestamp used for ordering and why it is appropriate.

The demo audit found multiple label matches for every requested test: 13 for
Glucose, 20 for Creatinine, 13 for Potassium, 13 for Sodium, and 25 for
Hemoglobin. These are not final research mappings. They include blood, urine,
body-fluid, blood-gas, chemistry, and hematology variants. Step 3 must select
and justify the exact blood-test concepts using a reproducible mapping table.

## Research-question feasibility

MIMIC-IV can provide laboratory results, patient/admission context, diagnoses,
medication orders/administration, and procedures/events. However, the dataset
does not provide a ready-made label saying that a particular future laboratory
report caused a clinical action.

Therefore the final binary target is **not approved in Step 2**. Before Step 3,
we must inspect the authorized data and select one reproducible downstream
event definition. Candidate definitions may include a new order, medication
prescription change, or documented procedure/event within a fixed observation
window, but each must be shown to be timestamped and identifiable. An
association in time is not proof of clinical causation.

The intended temporal unit is a laboratory report at prediction time `t0`:

```text
features available before t0 -> future selected lab report -> observation window -> measurable event
```

The future result itself and any information recorded after `t0` must be
excluded from model features. If no downstream event can be operationalized
reliably, the original research question cannot be implemented exactly with
MIMIC-IV and the paper must state the narrower alternative.
