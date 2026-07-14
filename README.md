# GLIDER: Graph-Learned Inductive Distributed Edge Routing for LEO Constellations

Code and paper for an IEEE INFOCOM 2027 submission. GLIDER learns a congestion-aware
**cost-to-go** function over dynamic LEO constellation snapshots with an inductive
message-passing GNN, and forwards greedily via
`argmin_v [ c(u,v) + Q(v,d) ]`. Because the cost-to-go is produced by an inductive
GNN, a single trained model transfers **zero-shot** to unseen constellation
geometries and to in-orbit link failures, and it executes in a **distributed** way
(each hop needs only neighbour embeddings plus the destination embedding).

```
infocom-conf/
├── paper/            # LaTeX source (IEEEtran conference, 10-page limit)
│   ├── main.tex
│   ├── refs.bib
│   └── figs/         # figures are generated from result CSVs
└── code/
    ├── glider/       # library: simulator, baselines, model, training, eval
    ├── configs/      # smoke / main / ablation YAMLs
    ├── scripts/      # run_smoke.sh, run_full.sh, make_figures.py
    └── tests/        # pytest unit + end-to-end smoke tests
```

## Status (read before submitting)

The simulator, baselines, training/eval pipeline, and tests are complete and green
(24 tests). The **SP-vs-CA-Global opportunity is real and reproducible**: under
skewed demand the centralized congestion-aware reference delivers about ten points
more carried demand than shortest path at lower peak utilization.

The **learned GLIDER policy is not yet competitive at the small CPU "anchor"
budget** used during development. Fixing the training objective (adding the
temperature-scaled next-hop imitation loss) and deepening the network moved carried
demand from ~0.18 to ~0.77 of shortest path's level on a tiny constellation, and the
next-hop loss was still decreasing, but reaching the shortest-path envelope needs a
receptive field near the network diameter (10+ message-passing rounds) and a large
training budget. That is cheap on the 4090 but exceeds a CPU anchor. **Run
`scripts/run_full.sh` on the GPU and inspect `results/*.csv` before trusting any
GLIDER number in the paper.** The paper's result macros for GLIDER are marked
preliminary and must be regenerated.

## What is (and is not) modelled

* **Constellation** — analytic Walker-delta on circular Keplerian orbits; +Grid ISL
  topology (degree 4) with an optional polar cut-off for inter-plane links; ground
  stations with an elevation-mask visibility model. This matches the constellation
  model used by Hypatia (Kassing et al., IMC 2020) but is fully self-contained.
* **Traffic / queueing** — gravity-model GS-to-GS demands scored with an M/M/1
  flow-delay model (propagation + queueing), reporting mean/p95 latency, peak link
  utilisation, overloaded-link count, and carried-demand fraction.
* **Not modelled** — packet-level ns-3 dynamics, TCP control loops, and antenna
  handover scheduling. The flow-level model is a deliberate, documented tradeoff
  that keeps the whole pipeline reproducible on a single GPU. See the paper's
  limitations section.

## Setup (RTX 4090 host, Ubuntu + conda)

```bash
git clone https://github.com/haodpsut/glider-leo-routing.git
cd glider-leo-routing
bash code/scripts/setup_conda.sh     # creates env 'glider', checks the GPU, runs tests
conda activate glider
```

`setup_conda.sh` is idempotent: re-running it updates the environment in place. It
installs PyTorch with a bundled CUDA 12.1 runtime from the `pytorch`/`nvidia`
channels, so **no system CUDA toolkit is needed**, only a recent NVIDIA driver. If
your driver is too old, change `pytorch-cuda=12.1` to `11.8` in
[`environment.yml`](environment.yml). The script prints `cuda avail` and the GPU
name; if it says CUDA is not visible, check `nvidia-smi` before running anything.

Manual equivalent, if you prefer:

```bash
conda env create -f environment.yml && conda activate glider
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

<details>
<summary>Alternative: plain venv + pip (no conda)</summary>

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
</details>

## Smoke test (seconds)

```bash
conda activate glider && cd code
pytest -q                 # 24 unit + end-to-end tests
bash scripts/run_smoke.sh # tiny constellation, trains a few dozen steps, evaluates
```

## Full experiments (paper numbers)

The run takes hours, so drive it from **tmux** and it survives an SSH drop:

```bash
conda activate glider && cd code
bash scripts/run_tmux.sh                       # start, then Ctrl-b d to detach
SEEDS="1 2 3 4 5" N_EVAL=100 bash scripts/run_tmux.sh   # bigger budget
```

The session opens three panes: the experiment, a live log tail, and `nvidia-smi`.
It activates the conda env for you and tees everything to `code/logs/`.

```bash
bash scripts/run_tmux.sh --status   # progress, without attaching
bash scripts/run_tmux.sh --attach   # reattach after an SSH drop
bash scripts/run_tmux.sh --kill     # stop the run
tail -f code/logs/latest.log        # or just watch the log
```

<details>
<summary>Foreground equivalent (no tmux)</summary>

```bash
cd code
SEEDS="1 2 3" bash scripts/run_full.sh
```
</details>

For **each seed** this trains GLIDER and the no-message-passing ablation from
scratch on the `medium` constellation, then evaluates that model in-distribution and
zero-shot on `starlink_shell1`, `kuiper_shell`, and `telesat_polar`, and sweeps the
ISL failure rate. Finally it emits `paper/tables/*.tex` and `paper/figs/*.pdf`.

**Reporting protocol (this is what the paper claims).** For each trained model we
average each metric over the evaluation scenarios; tables and figures then report
the mean and standard deviation of those per-seed means **across seeds**, so error
bars measure variability across *independently trained models*, not across scenarios
within one model. Scenario sampling uses a fixed held-out seed (`EVAL_SEED`),
identical for every method, so all methods see the same instances. SP and CA-Global
do not depend on the model and therefore have zero spread across seeds by
construction. This protocol is locked by `tests/test_aggregation.py`.

**No number is typed by hand.** `main.tex` `\input`s `paper/tables/*.tex`, which are
generated by `scripts/make_tables.py` straight from `results/*.csv`. After a run,
just rebuild the paper.

Override the budget with env vars, e.g. `SEEDS="1 2 3 4 5" N_EVAL=100 bash scripts/run_full.sh`.

### Performance notes

* Training is GPU-bound in the GNN forward/backward but **scenario labelling
  (CA-Global via iterative Dijkstra) runs on CPU**. On a 4090 host with a modern
  multi-core CPU, expect the `medium` config (3000 steps) to complete in well under
  an hour; the GPU is the model, the CPU is the data. To speed up labelling, lower
  `scenario.ca_iters` or `scenario.n_ground_stations` in the config.
* Peak GPU memory for the `main` config is small (a single graph of a few thousand
  nodes per step), so `hidden` and `num_layers` can be scaled up freely on a 4090.

## Reproducing individual pieces

```bash
conda activate glider && cd code

# Train one seed
python -m glider.train --config configs/main.yaml --seed 1 --out runs/main_s1

# Evaluate a checkpoint on a specific constellation
python -m glider.evaluate --config configs/main.yaml --ckpt runs/main_s1/glider.pt \
    --presets kuiper_shell --n 50 --seed 1000 --run-seed 1 --out results/kuiper_s1.csv

# Regenerate tables + figures from whatever CSVs exist
python scripts/make_tables.py  --results results --out ../paper/tables
python scripts/make_figures.py --results results --runs runs --out ../paper/figs
```

## Building the paper

```bash
cd paper
latexmk -pdf main.tex     # or: pdflatex main.tex && bibtex main && pdflatex ... x2
```

Tables and figures are `\input`/`\includegraphics`-ed from generated files, so after
an experiment run you only need to rebuild.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `setup_conda.sh` prints `cuda avail : False` | Check `nvidia-smi`. If the driver predates CUDA 12.1, set `pytorch-cuda=11.8` in `environment.yml` and re-run the script. |
| `conda: command not found` | Install Miniconda, then `source ~/miniconda3/etc/profile.d/conda.sh`. |
| `CondaError: Run 'conda init'` inside tmux | Not needed: `run_tmux.sh` uses `conda run`, which does not require a shell hook. |
| Training feels slow, GPU is idle in `nvidia-smi` | Expected in bursts: scenario labelling (CA-Global, iterative Dijkstra) is **CPU**-bound; the GPU only runs the model. Lower `scenario.ca_iters` or `scenario.n_ground_stations` in the config to speed up labelling. |
| SSH dropped mid-run | Nothing is lost. `bash scripts/run_tmux.sh --attach`. |
| Want to resume after a crash | Per-seed checkpoints live in `runs/main_s<seed>/`; re-running `run_full.sh` retrains from scratch, so move or delete `runs/` first if you want a clean slate. |
