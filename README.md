# mlx-auto-lora

A minimal autonomous LoRA fine-tuning lab for Apple Silicon (MLX).
Cron-driven bandit HPO, domain-specific eval (compile gate + LLM judge),
git as the experiment ledger.

No MLflow. No Ray. No Kubernetes. ~600 lines of Python plus a few shell
scripts. The kind of thing you can drop on a Mac Studio, point at a corpus,
and let run overnight.

## Why this exists

Automated LoRA HPO is a solved problem on GPU clusters - Axolotl has sweep
support, Ray Tune does bandit search, there are real papers on automated
rank/target-module selection. None of that fits a single M-series Mac:

- The memory envelope problem is specific to Apple Silicon. The raw GPU
  limit is the nominal RAM (96 GB on an M2 Ultra), but in practice you share
  unified memory with the OS and any model server you have running. The real
  unattended ceiling is more like 70 GB. Worse, the safe region is
  **jointly** bounded by `(num_layers, max_seq_length)` - picking each
  axis's safe max independently can still OOM.

- Most LoRA autoresearch tooling uses perplexity or MMLU-style benchmarks as
  fitness. For code, a hard `compile/parse` gate is a much sharper signal -
  it catches hallucinated symbols and bad syntax cheaply, before any judge
  even reads the output.

- The MLX community has the strongest local-LLM story on the planet and
  almost zero "let me fine-tune overnight without babysitting" tooling.

This repo is what I wish I'd had.

## Architecture

```
       cron (the loop)
          v
  bin/lab.sh EXPERIMENT
          v
  ┌───────────────────────────┐
  │ lab/run_phase.py          │
  │  1. lock                  │
  │  2. propose ONE hypothesis│   ← bandit, constrained to memory envelope
  │  3. edit config.yaml      │
  │  4. train.py (mlx-lm)     │   ← hard wall-clock budget
  │  5. eval.py               │   ← compile + lint + LLM judge
  │  6. keep (git commit)     │     ─┐
  │     OR revert (checkout)  │      │  ← the working tree IS the
  │  7. notify, unlock        │     ─┘    current best config
  └───────────────────────────┘
```

The bandit picks from a small menu of hypotheses (`H1-rank`, `H4-lr`,
`H8-seq`, `H11-layers`, `Ha-alpha`, `Hd-dropout`, ...). Each EXPERIMENT
mutates one knob in `lab/config.yaml`. If the new composite score is the
best so far, the change is committed and the adapter moved to
`adapters/keepers/`. Otherwise the config is `git checkout`ed and the
adapter is deleted. The result is appended to `lab/results.tsv` either way.

State that actually matters:

| File                    | Role                                    |
| ----------------------- | --------------------------------------- |
| `lab/config.yaml`       | The current best configuration. Mutated each run; reverted on regression. |
| `lab/results.tsv`       | Append-only log of every experiment.    |
| `lab/explored.json`     | Bandit memory: which (axis, value) pairs have been tried. |
| `logs/mem_envelope.tsv` | Output of `bin/mem_probe.sh`. Defines the feasible config space. |
| `state.json`            | Best run, prod score, pause flag.       |
| Git history             | Every kept change is a commit. Reverts are also commits (so reverts are visible). |

## The memory envelope

The interesting part. Run `bin/mem_probe.sh` once with anything memory-hungry
(model servers, browsers) closed; it does short 2-iter trainings across a
grid of `(num_layers, max_seq_length)` combos and records peak GPU memory.

Below is a representative envelope from a 96 GB M2 Ultra training Qwen3-30B
at 8-bit, batch 1, gradient checkpointing on:

| num_layers | max_seq | result | peak GB |
| ---------: | ------: | :----: | ------: |
|          2 |    1024 |   OK   |    33.0 |
|          4 |    1024 |   OK   |    41.8 |
|          4 |    2048 |   OK   |    52.4 |
|          8 |    1024 |   OK   |    58.7 |
|          8 |    2048 |   OK   |    85.1 |
|         12 |    1024 |   OK   |    71.2 |
|         12 |    2048 |  OOM   |       - |
|         16 |    2048 |  OOM   |       - |

Two things to notice:

1. `(8, 2048)` peaks at 85 GB. That fits the nominal 96 GB *in isolation*
   but reliably OOMs in real conditions when an inference server is up. The
   default budget in `run_phase.py` is `PEAK_BUDGET_GB = 70` for this reason.
   Override with `MLX_AUTO_LORA_PEAK_GB=80` if your machine is dedicated.

2. The envelope is **not separable**. `num_layers=8` is safe and
   `max_seq_length=2048` is safe, but `(8, 2048)` is unsafe. The bandit
   refuses any combo unless some measured-feasible point dominates it on
   both axes. This is the line in `run_phase.py:is_safe` and it's the
   single most useful check in the whole thing.

`mem_probe.sh` regenerates the table any time you change the base model,
the quantization, or you upgrade macOS / mlx-lm.

## Quick start

Requirements: macOS on Apple Silicon (M1/M2/M3 series), Python 3.10+,
a corpus you want to fine-tune on.

```bash
# 1. Clone and set up
git clone https://github.com/stephenlthorn/mlx-auto-lora.git
cd mlx-auto-lora
export MLX_AUTO_LORA_ROOT=$PWD
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configure
cp .env.example .env             # add ANTHROPIC_API_KEY etc. (all optional)
$EDITOR lab/config.yaml          # set base_model, data.sources

# 3. Build the corpus
.venv/bin/python lab/prepare_data.py --out corpus/

# 4. Probe the memory envelope (close other heavy apps first)
bin/mem_probe.sh

# 5. Smoke test: run one experiment manually
bin/lab.sh EXPERIMENT
cat lab/results.tsv              # should have one row

# 6. Let it run
bin/install_cron.sh              # installs the default schedule
crontab -l                       # verify
```

The default cron schedule does six EXPERIMENTs/night between 20:00 and 03:30,
a full EVAL of the keeper at 04:05, a digest at 09:00, and a HEALTH check
every 15 minutes. Edit `bin/install_cron.sh` to suit.

## Swapping the eval domain

`lab/config.yaml`'s `eval.domain` picks a module from `lab/domains/`. Each
domain provides four things:

```python
JUDGE_SYSTEM: str                          # LLM judge system prompt

def extract_code(text: str) -> str: ...    # pull the answer out of raw output
def compile(code: str) -> int | None: ...  # 1/0 or None if unavailable
def lint(code: str) -> int | None: ...     # 1/0 or None if unavailable
```

Shipped:

- **`swift`** - `swiftc -parse` compile gate + optional `swiftlint`. Judge
  prompt graded for "modern Swift/SwiftUI idioms". This is the default and
  what the example prompts in `lab/evals/quick_prompts.jsonl` target.

- **`text`** - judge-only, no compile gate. Useful for prose, dialog,
  summarization, or any domain without a static checker. The composite
  drops to single-component when only judge is measurable.

To add a new domain (Python, Rust, TypeScript, your DSL...), copy
`lab/domains/swift.py` and change the four functions. Then set
`eval.domain: <yourname>` in `config.yaml`. Even a permissive parser
(`python -c "import ast; ast.parse(code)"`, `cargo check`, `tsc --noEmit`)
contributes more fitness signal than judge-only.

## The LLM judge

If `ANTHROPIC_API_KEY` is set in `.env`, the judge calls Claude (defaults to
Opus 4.7; override with `ANTHROPIC_JUDGE_MODEL`). Without it, the judge
component is recorded `null` and the composite is **renormalized** over the
components that *were* measured. This is deliberate - refusing to score
because one helper is missing is brittle. A score with `measured_components:
["compile", "lint"]` is still useful; the JSON reports which components
contributed so you can audit.

Want a different judge (local model via mlx-lm, OpenAI, etc.)? Add a branch
in `lab/eval.py:_judge_client`. The contract is one callable that takes
`(prompt, code)` and returns a float in [0, 1].

## Telegram remote control (optional)

The cron loop runs fine without it. If you want to /pause from your phone,
set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and optionally
`TELEGRAM_THREAD_ID` in `.env`, then:

```bash
bin/install_listener.sh
```

Supported commands (in the configured chat/thread): `/lab_pause`,
`/lab_resume`, `/lab_status`, `/lab_hypothesis H4`, `/lab_deploy <run_id>`,
`/lab_kill`, `/lab_help`. See [bin/tg_listener.py](bin/tg_listener.py) for
the full list.

If you don't set the Telegram vars, notifications fall back to stdout (which
the cron job logs to `logs/cron.log`).

## Honest limitations

- **Apple Silicon only.** Nothing here is hard-tied to MLX in principle,
  but the memory envelope, the cron scheduling, and the launchd listener
  are all M-series shaped. On a GPU cluster you'd use Ray Tune.

- **The bandit is dumb on purpose.** No Thompson sampling, no Bayesian
  surrogate model. It picks an untried (hypothesis, value) pair uniformly
  from those that pass the safety check, biases toward the menu, and stops
  when everything has been tried. Spending compute on smarter selection
  saves at most a few experiments per night - the bottleneck is the
  hour-long train+eval cycle, not the picker.

- **Compile gate is the magic.** If your domain doesn't have a fast static
  checker, you get less discrimination, full stop. The framing is
  "domain-specific eval as fitness", not "judge as fitness".

- **Not novel as "AutoLoRA".** Useful as a minimal pattern: cron + git +
  bandit + compile gate + a tiny memory-envelope check, runnable on one Mac.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This started as an iOS / Swift fine-tuning project (`ios-lora-q36`) for a
specific app stack. The Swift example domain is left in because the
`swiftc -parse` compile gate is genuinely useful and shows what
domain-specific eval looks like in practice. The general pattern - cron is
the loop, git is the ledger, one hypothesis per run, memory envelope before
choice - applies to anything you can score in under a minute.
