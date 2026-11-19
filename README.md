# Maximizing modelled votes over aggregation rules sacrifices proportional representation in real participatory budgets

Core computational code accompanying the paper of the same title.

**Paper authors:** Tianyu Shi; Yiwen Huang; Shuhao Chen; Ziheng He; Chengyi Tu (corresponding author)  
**Affiliation:** School of Economics and Management, Zhejiang Sci-Tech University  
**Code license:** MIT  
**Release prepared:** 7 October 2026

## Purpose and scope

This repository implements model-implied elections between participatory-budgeting aggregation rules. It computes eight incumbent rules, trains a 97-parameter sequential rule by evolution strategies, evaluates learned rules and non-learned selectors, and audits proportionality and minority representation under count, cost and square-root utility models. These are counterfactual preferences inferred from utility models, not observed votes over rules.

The paper's primary analysis covers 1,069 budgeting elections and 3,673,063 non-empty ballots. Computation initially retains 1,083 records; the statistical analysis removes 14 paired Stanford knapsack records to obtain the primary sample. Preserve this distinction when comparing stage outputs with the paper.

This is a **minimal, source-preserving release**. All seven Python files are copied byte-for-byte from the supplied research source. No algorithm, seed, hyperparameter or statistical calculation has been changed. The package includes the fixed catalogue and input checksums because they are needed to select, group and verify the original inputs. It does not include ballots, trained parameters, generated results, plotting code, manuscript sources, submission correspondence or PDF-building tools. The numerical data underlying the analyses and figures are generated when the pipeline is run; the manuscript and its visual presentation are not rebuilt by this release.

## Files

```text
real-ballot-rule-elections-code/
├── README.md
├── requirements.txt
├── LICENSE
├── catalogue.json                  # fixed 2,052-entry Pabulib metadata catalogue
├── input_manifest_sha256.json      # hashes of the 1,209 eligible input files
└── code/
    ├── pbcore.py                   # parsing, rules, utilities, social choice and audits
    ├── stage1_rules.py             # instance selection, rule outcomes, grouped partition
    ├── validate_pabutools.py        # comparison with the reference implementations
    ├── train_dai.py                # evolution-strategies training
    ├── stage3_eval.py              # rule elections, selectors and representation audits
    ├── stage1b_bootstrap.py        # bootstrap with incumbent outcomes recomputed
    └── analysis.py                 # numerical summaries and CSV tables
```

`catalogue.json` is input metadata, not a results file. Keeping the complete catalogue is necessary for the reported sample-flow counts. `input_manifest_sha256.json` is copied from the original input manifest; it contains no ballot contents or model outputs.

## Environment and dependencies

The original compute environment reported in the paper was a 192-core Linux server running Python 3.11. Use **64-bit CPython 3.11 on Linux**, or Linux in WSL on Windows, for the documented commands. Bash is used only to organize the runs. The Python programs also support serial execution with one worker; limited Windows multiprocessing checks passed during packaging, but a complete native-Windows reproduction was not performed.

From the package root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The requirements retain the supplied release's versions of NumPy, SciPy, pandas, statsmodels, pabutools and its explicitly pinned support packages. Plotting and PDF dependencies have been removed. pip installs additional dependencies declared by these packages automatically. MILP calculations in `pbcore.py` use SciPy/HiGHS; the research calculations do not require a Gurobi license or a GPU.

The original server's complete environment lock was not supplied. These version pins therefore describe the supplied release requirements, rather than a verified lock of every package on the original compute server.

## Obtain and verify the input data

The paper identifies the Pabulib snapshot dated **25 September 2026**, containing 2,052 files:

[Download the identified Pabulib snapshot](https://pabulib.org/download/snapshot/e67daae7d1ffdcd9).

Download it separately and save the ZIP as `pabulib_snapshot.zip` in the package root. Use the included catalogue, not a newly downloaded live catalogue. The external snapshot download could not be independently completed during packaging; access to these exact files remains a prerequisite for a full reproduction.

The scripts expect a ZIP named `approval.zip` in the **working directory**, with `.pb` files at its top level. The following preparation command selects the 1,209 inputs listed in the checksum manifest, verifies their original bytes, and writes the flat archive. It handles either a flat downloaded ZIP or `.pb` files inside ZIP subfolders. No ballot content is transformed.

```bash
python - <<'PY'
import hashlib
import json
from pathlib import PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

with open("input_manifest_sha256.json", encoding="utf-8") as f:
    expected = json.load(f)
with ZipFile("pabulib_snapshot.zip") as source:
    entries = {}
    for info in source.infolist():
        name = PurePosixPath(info.filename).name
        if not info.is_dir() and name in expected:
            if name in entries:
                raise SystemExit(f"Duplicate input basename in snapshot: {name}")
            entries[name] = info
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise SystemExit(f"Missing {len(missing)} required inputs; first: {missing[:5]}")
    for name in sorted(expected):
        digest = hashlib.sha256(source.read(entries[name])).hexdigest()
        if digest != expected[name]:
            raise SystemExit(f"SHA-256 mismatch: {name}")
    with ZipFile("approval.zip", "w", compression=ZIP_DEFLATED) as target:
        for name in sorted(expected):
            target.writestr(name, source.read(entries[name]))
print(f"Verified and packed {len(expected)} inputs into approval.zip")
PY
```

If the snapshot is unavailable or a checksum differs, obtain the matching original files before claiming reproduction of the paper. Do not silently substitute a newer snapshot. Pabulib data and metadata retain their source terms; the MIT license covers this repository's original software and documentation, not third-party data or dependencies.

## Reproduce the computations

Run **every command below from the package root**, with the virtual environment active. `out/`, `logs/` and `tables/` are created locally and are not distributed. Use a fresh working copy for a new complete run so that stale parameter files cannot be mixed with new runs.

The commands below execute training runs sequentially, with a configurable worker count per run. This keeps the scientific configurations unchanged while avoiding the original server wrapper's simultaneous launch of 21 training jobs.

### 1. Set the environment and compute the incumbent outcomes

```bash
set -euo pipefail
export PYTHONUTF8=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NPROC=4                       # reduce to 1 on a small machine
mkdir -p out logs tables

python code/stage1_rules.py approval.zip catalogue.json "$NPROC" > logs/stage1.log
python code/validate_pabutools.py > logs/validation.log
```

This creates the incumbent outcomes, edition-grouped partition, solver diagnostics and a fresh input manifest. The partition seed is `20260928`. All Warsaw elections are reserved for the municipality holdout. Expected original counts are 1,209 stage-one input records, 1,083 retained records and 2,166 primary optimization problems. Inspect `logs/stage1.log` and the per-record JSON errors before continuing.

The reference validation samples 150 retained elections with **at most 3,000 voters**, as implemented. See the reproducibility notes below for the difference from the manuscript's threshold and for sample-order sensitivity.

### 2. Train the grouped, sensitivity and municipality-holdout models

The following blocks are intended to run in the same Bash session as step 1. Each call keeps the original training settings: 200 generations, 32 antithetic perturbation pairs, batch size 64, logistic temperature 0.05, perturbation scale 0.10 and learning rate 0.05. Training and validation are restricted to elections with at most 5,000 voters.

```bash
run_model() {
    python code/train_dai.py "$1" "$2" "$3" "$4" "$NPROC" "$5" "${6:-}" \
        > "logs/$5.log" 2>&1
}

# Nine league models: three utility models and three seeds.
for K in card cost sqrt; do
    for S in 0 1 2; do
        run_model "$K" league "$S" 0 "${K}_league_s${S}"
    done
done

# Twelve alternative-rival and exclusion-penalty models.
for K in card cost; do
    run_model "$K" league_distinct 0 0 "${K}_distinct_s0"
    run_model "$K" GreedyAV 0 0 "${K}_GreedyAV_s0"
    run_model "$K" MES 0 0 "${K}_MES_s0"
    for L in 0.5 2 8; do
        run_model "$K" league 0 "$L" "${K}_league_lam${L}"
    done
done

# Sixteen leave-one-municipality/platform-out models.
for PAIR in "Gdynia:Gdynia" "Lodz:Łódź" "Stanford:Stanford Dataset" \
            "Amsterdam:Amsterdam" "Lublin:Lublin" "Toulouse:Toulouse" \
            "Poznan:Poznań" "Olsztyn:Olsztyn"; do
    TAG_CITY="${PAIR%%:*}"
    HOLD_CITY="${PAIR#*:}"
    for K in card cost; do
        run_model "$K" league 0 0 "${K}_lomo_${TAG_CITY}" "$HOLD_CITY"
    done
done
```

This produces 37 `out/dai_*.json` files. Training uses seed `1000 + S`; model selection uses validation support. The main grouped models in the supplied results are `card_league_s1`, `cost_league_s2` and `sqrt_league_s0`. They are selected by the code rather than hard-coded as winners.

### 3. Train the two temporal models

```bash
run_model card league 0 0 card_temporal TEMPORAL
run_model cost league 0 0 cost_temporal TEMPORAL
```

Temporal development uses non-Warsaw elections from 2022 or earlier. Evaluation uses non-Warsaw elections from 2023 onwards. There should now be 39 trained-parameter files.

### 4. Evaluate the rules and recompute the voter bootstrap

```bash
MAIN_MODELS=()
for MODEL in out/dai_*.json; do
    if [[ "$MODEL" != *_temporal.json ]]; then
        MAIN_MODELS+=("$MODEL")
    fi
done
[[ ${#MAIN_MODELS[@]} -eq 37 ]]

python code/stage3_eval.py "$NPROC" "${MAIN_MODELS[@]}" > logs/stage3.log
S3_MINYEAR=2023 S3_OUT=out/stage3_temporal.json \
    python code/stage3_eval.py "$NPROC" \
    out/dai_card_temporal.json out/dai_cost_temporal.json > logs/stage3_temporal.log
python code/stage1b_bootstrap.py approval.zip "$NPROC" 100 > logs/bootstrap.log
```

The last command performs 100 full-recomputation bootstrap samples per election, reduced to 50 for electorates with more than 20,000 voters. Stage three separately uses 200 samples for fixed-outcome election bootstrap summaries. Bootstrap random seeds are derived from filenames. The temporal output should contain 302 elections; the main evaluation and full bootstrap each contain 1,083 records before primary-sample deduplication.

Before aggregation, check for errors and input changes:

```bash
python - <<'PY'
import json
def load(name):
    with open(name, encoding="utf-8") as f:
        return json.load(f)
for name in ("stage1_instances.json", "stage3_results.json",
             "stage3_temporal.json", "bootstrap_full.json"):
    rows = load("out/" + name)
    failures = [r for r in rows if "error" in r]
    if failures:
        raise SystemExit(f"{name}: {len(failures)} failed records; inspect before aggregation")
if load("out/manifest_sha256.json") != load("input_manifest_sha256.json"):
    raise SystemExit("Generated input manifest differs from the supplied manifest")
solver = load("out/solver_log.json")
if any(r.get("status1") != 0 or r.get("status2") != 0 or not r.get("primary_kept")
       for r in solver):
    raise SystemExit("A solver result needs inspection")
print("No per-record errors; input hashes match; solver checks passed")
PY
```

### 5. Generate the numerical summaries

```bash
python code/analysis.py . catalogue.json tables > logs/analysis.log
```

The analysis seed is `20261003`. The supplied implementation uses 2,000 resamples for its default clustered analyses, 500 for the league-share summary intervals and 1,000 for demographic-group differences. These quantities must not be described as uniformly using 2,000 resamples.

## Generated outputs

| Location | Contents |
|---|---|
| `out/stage1_instances.json`, `out/stage1_outcomes.pkl` | Selected instance metadata and incumbent outcomes |
| `out/split.json`, `out/solver_log.json`, `out/manifest_sha256.json` | Grouped partition, solver records and input hashes |
| `out/validation_pabutools.json` | Reference-library comparisons |
| `out/dai_*.json` | 39 model parameter files with configurations and training histories |
| `out/stage3_results.json`, `out/stage3_temporal.json` | Pairwise comparisons, selectors, representation audits and temporal evaluation |
| `out/bootstrap_full.json` | Recomputed-bootstrap summaries |
| `tables/numbers.json` | Aggregated statistics used in the paper |
| `tables/T_*.csv` | Election, pairwise, held-out, sensitivity and municipality-holdout tables |
| `logs/` | Local run logs |

Do not distribute a generated output as if it were an input. Load `stage1_outcomes.pkl` only when it was produced by your own trusted run. The local aggregation step creates numerical tables, not the typeset tables or plots.

The paper reports about one minute for stage one with 96 processes and 10–30 minutes per training run with four processes on the original server. These are author-reported measurements, not benchmarks of this package. A sequential run on a personal computer can take substantially longer; full bootstrapping is also computationally demanding.

## Verification and known limitations

Packaging checks on 7 October 2026 established the following:

- All seven packaged Python files match their originals byte-for-byte and parse successfully.
- A pip dry-run successfully resolved the nine pinned requirements and their declared dependencies on Python 3.12 / Windows. It did not install packages or constitute a Linux 3.11 environment test.
- Re-running `analysis.py` on the supplied, privately retained stage outputs reproduced all recorded aggregate values within numerical comparison tolerance and passed all 51 checks in the supplied manuscript-number verifier. Those stage outputs and the manuscript-dependent verifier are not in this minimal release.
- Small synthetic tests covered budget feasibility, utilitarian optima against exhaustive enumeration, EJR audits against direct enumeration, maximal lotteries, and stage-one/bootstrap/stage-three execution. They do not replace a full run on the Pabulib snapshot.
- Full retraining, a fresh 150-election pabutools validation, and a complete end-to-end run from downloaded raw ballots were not performed during packaging. Local checks used Python 3.12.10, NumPy 2.4.5, SciPy 1.17.1, pandas 3.0.3 and statsmodels 0.14.6, distinct from the original release pins.

The source-preserving review identified issues that remain in this unmodified release:

1. **Reference-validation sampling:** `validate_pabutools.py` uses a 3,000-voter limit; the supplied manuscript says 2,500. It shuffles stage-one records whose order can depend on parallel completion, so the validation sample can change despite its fixed random seed.
2. **Sign-test ties:** the Equal Shares versus GreedyAV sign test in `analysis.py` includes exact pairwise ties in its trial count. This differs from the conventional sign test that removes ties. It affects the reported `p_sign`, not the computed mean support.
3. **Exact-audit coverage:** `stage3_eval.py` reuses identical outcomes through `same_as` links. If an eligible main model aliases an earlier non-main model without an exact audit, its all-subset EJR audit is omitted. Restricted audits are still present. Missing exact-audit records must not be interpreted as certified passes or as coverage of every small election for each learned rule.
4. **Methods descriptions:** the bootstrap counts vary by analysis as documented above; the learned approval-to-cost feature uses `1e-6` stabilizers, not the manuscript's description of adding one approval.

The computations are released as supplied so that these distinctions remain inspectable. Resolving the statistical and exact-audit issues would require a separately versioned code revision and renewed checks; this archive does not silently apply such changes. Solver versions, parallel record ordering and floating-point behavior can also prevent byte-identical regenerated outputs. Reproduction of existing calculations is not itself proof that every method is free from error.

## Acknowledgement and license

Please acknowledge the accompanying paper when reusing the code. No publication DOI is asserted in this release. Pabulib supplies the underlying public research data; pabutools supplies the reference implementations. See `LICENSE` for the MIT terms applying to the original code and documentation.
