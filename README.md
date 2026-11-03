# GPT in a tab: Microscope

Train a small transformer in your browser, on your GPU, with hand-written WebGPU kernels. Watch
the moment it learns to copy. Then take it apart: knock out heads, patch activations between
runs, and read its residual stream through a sparse autoencoder trained in WebGPU. No server, no
model download, no machine-learning library; the model you dissect is the one you just trained.

![The induction moment: the copy loss falls off a cliff at step 1,500 and L2·H4's lane lights up](shots/v3/desktop-05-the-moment.png)

## Ten seconds with it

1. **Open the page.** Within 150 ms a recorded run of this same code is on the trace, dim, marked
   REF. Its blue copy loss falls off a cliff at step 1,500, labelled "recorded: L2·H4 learned to
   copy at step 1,500". Your own run starts training over it in under a second.
2. **Wait about 20–30 seconds.** Your run reaches the same cliff, and the trace writes it in orange:
   *L2·H4 learned to copy at step 1,500 · copy loss 4.06 → 0.97 nats*. The head's lane under the
   trace turns orange, and its partner, a head that attends to the previous character, is outlined.
3. **When the run ends, open Heads.** Shift-click the copying heads to knock them out. On random
   characters shown twice, the trained model continues the copy 40 of 40 characters right; without
   those two heads, 0 of 40.
4. **Open Features.** Pick a feature and pin it at three times its strongest activation. Feature
   #918 fires on the "X" that starts the sonnets' Roman numerals; pinned, the model writes
   XXXXXXXX… and nothing else.

## The techniques

### Induction heads, forming live

[Olsson et al. 2022, *In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

Every 20 training steps the GPU runs the standard test, between training steps and in the same
command stream. It takes 32 rows of 64 random characters, each shown twice. For every head it
measures the **prefix-matching score**: the attention each position of the second copy pays to
the character just after the previous occurrence of its own character. It also measures the
**previous-character score**, and the **loss on each copy**. A head counts as an induction head
when its prefix-matching score reaches 0.3 and the second copy has become at least 1 nat easier
than the first. The first reading where that happens is **the moment**. The trace marks it in
words, draws the cliff in the copy loss (CH4) from the plateau, and lights the head's lane. The
scoring is three small WGSL kernels (`probe_attn`, `probe_loss`, plus a pattern copy) appended to
a forward pass. Their CPU reference is `scoreProbe` in `src/model/probe.ts`.

**What it took to see the moment within a minute.** The finding first:
- **On text alone, no head learned to copy within the budget.** That held for 3,000 steps on the
  sonnets and 3,000 on the five tragedies. A strong previous-character head formed on both, but
  no induction head. Olsson et al. see the phase change after hundreds of millions of tokens; a
  minute here is about 7 million.
- **So one training window in four is random characters**: a run of 12 to 48 characters,
  repeated to fill the window. Nothing in it can be memorised, so the only way to predict a repeat
  is to copy it.
- **The period varies from window to window**, so a head cannot pass by looking a fixed distance
  back. The test uses runs of 64, longer than any repeat the model trained on.

The controls I ran, in this repository's `tests/gpu/induction.html`, with the probe on CPU
snapshots:

| training data | induction head |
| --- | --- |
| text only (sonnets, tragedies), 3,000 steps | none |
| one window in four a repeated passage *of the text* | none in 1,700 steps (the model memorises the text, so copying never pays) |
| one in two: random characters with a fixed period of 64, no dropout, constant learning rate | at about step 2,850 |
| one in eight: random characters, varying period | none in 2,000 steps |
| **one in four: random characters, varying period (shipped)** | **step 1,100–1,600** across the seeds and dropout settings tried; **1,500** for the shipped seed |

Training is deterministic for a seed on a given GPU. The live run and the recorded one both found
L2·H4 at step 1,500, with its partner L1·H2 attending to the previous character (score 0.99).

### Causality: knockouts and activation patching

When the run ends, the GPU repeats the forward pass with a `head_edit` kernel spliced in before
each layer's output projection. The kernel scales each head's output (0 knocks it out) and can
add the same head's output saved from another run.

- **Knockout (zero-ablation):** for each of the 16 heads, the change in loss on held-out text and
  on the test's second copy.
- **Activation patching** ([Meng et al. 2022](https://arxiv.org/abs/2202.05262), and as used by
  Olsson et al.):
  - **Clean run:** the test.
  - **Corrupted run:** the first copy replaced with other random characters, so the second copy
    has nothing to copy from.
  - **Patched runs:** the corrupted run, once per head, with that head's output taken from the
    clean run, measuring how much of the copying returns.

Results on the shipped run (step 2,250):

| head | its job (from the test) | knocked out: copy loss | knocked out: text loss | patched in: copying restored |
| --- | --- | --- | --- | --- |
| L1·H2 | previous character (1.00) | +4.65 nats | +0.84 | −5% |
| L1·H1 | | +2.84 | +0.10 | −2% |
| L2·H4 | induction (0.89) | +1.31 | −0.00 | 79% |
| L2·H3 | induction (0.88) | +0.78 | +0.00 | 69% |

Read together: the induction heads carry the copy, and they depend on the previous-character head.
Patching the previous-character head alone restores nothing, because the head that copies still
reads the corrupted first copy.

The page also lets you knock heads out while the model writes. Generation runs on the CPU in
workers, through a key/value-cache decoder with the same hooks. One demonstration does it on
random characters shown twice, continued greedily: **40 of 40 right as trained, 0 of 40 with
L2·H3 and L2·H4 knocked out.**

### A sparse autoencoder, trained in WebGPU

[Bricken et al. 2023, *Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html);
[Cunningham et al. 2023, *Sparse Autoencoders Find Highly Interpretable Features in Language Models*](https://arxiv.org/abs/2309.08600).

After the run, the trained model reads the whole training text in consecutive windows. The
residual stream entering its last block is kept at every position: 86,016 vectors of 128.
- **Shape:** an autoencoder with 1,024 features (8×) learns to rebuild those vectors:
  `f = ReLU(x·W_enc + b_enc)`, `x̂ = f·W_dec + b_dec`.
- **Loss:** squared error plus λ = 2 times the sum of activations. Inputs are scaled so the average
  vector has squared length 128.
- **Training:** Adam, 3,000 steps of 4,096 vectors, with the decoder rows renormalised to unit
  length after every step. It takes about 9 seconds.

Everything runs on the GPU:
- the existing tiled matmul and AdamW kernels;
- new kernels for the random row gather, the ReLU, the loss gradients, decoder normalisation and
  error sums;
- a top-k kernel that scans every position for each feature's 12 strongest contexts and its
  firing count.

**Measured on the shipped run:** 15.0% of the residual stream's variance unexplained; 14.7
features active per character on average; 0.0% dead features (none that never fires). These are
fidelity numbers, not a quality score. I did not measure the loss with the reconstruction spliced
back into the model ("loss recovered").

**Labels are by rule, never by a model.** Each candidate rule is a test on one context:
- the character itself, or the one before or after it;
- the word it is in, or the word just before it;
- the start or end of a word or line, capitals, vowels, punctuation.

A feature's label is the most specific rule, or pair of rules, that holds for at least 9 of its 12
strongest contexts, with specificity discounted by coverage. It is shown with its count: "holds
for 12 of the 12 strongest contexts". 675 of the 1,024 features get a label; the rest are shown
as examples only. Typical labels: "“t” ending a word", "“u” after “r”", "the space after
“and”", "“X” after a line break".

**Pinning a feature** while the model writes: at the residual stream entering the last block, the
decoder adds `(value − f_j(x)) · d_j`. That sets feature j to the chosen value and leaves
everything the autoencoder does not explain untouched (`clampFeature` in `src/cpu/sae.ts`).

## Live and recorded

**Recorded** (`npm run record`, `scripts/record.mjs`):
- The recording is this code's production build, run headless in Chromium with WebGPU, trained
  on the sonnets and dissected, on 2 Nov 2026 on this machine's GPU (Apple, Metal).
- It ships as `public/recorded/run.json` (92 KB, 17 KB gzipped: trace, induction readings, the
  moment, knockouts, patching, autoencoder summary), `features.json` (218 KB, 75 KB gzipped,
  fetched when Features first opens) and `weights.bin` (2.1 MB of half-precision weights).
  Model weights stay out of the repository: `weights.bin` is not committed, and `npm run record`
  writes it. Without it, the page shows the download's failure state and everything else works.
- It is the REF waveform on the trace from the first frame. In Heads and Features it is behind a
  Live/Recorded switch, and its weights are fetched only when you ask to write with it.
- The download states its size first, shows progress, can be cancelled, is kept in the Cache API,
  and fails with a message while the rest of the page keeps working.

**Live:** everything else. Your run, its moment, its knockouts and patching, its autoencoder,
and everything you write with it.

**Without WebGPU** (Firefox today, many phones):
- A 62,016-parameter model (2 layers of 2 heads) trains live on the CPU in a worker, and the
  induction test runs on it every 20 steps.
- In 3,000 steps (about three minutes) that small model formed no induction head; the page says
  so ("No induction head yet") rather than pretending.
- The microscope shows the recorded GPU run, labelled. With its weights loaded, knockouts and
  pinned features write on the CPU.

## Quickstart

```sh
npm install
npm run dev        # http://localhost:5173
```

A browser with WebGPU (current Chrome or Edge, Safari 26) trains the full model. `?cpu` forces the
fallback, `?text=alice|tragedies|words|kernels` picks another text, and `?hold` loads without
starting. `npm run build` writes a static site to `dist/`.

## Checks

| check | command | result |
| --- | --- | --- |
| CPU reference: gradient checks for every operation in float64, the whole model, AdamW, the decoder, sampling, batching with repeated windows, the schedule, half precision, the attention replay; the induction test on a hand-made pattern and on a real model; finite-difference gradients of the autoencoder; rule labels; knockouts while writing | `npm test` | 48/48 pass |
| Every GPU kernel against the CPU reference, then a training step, six steps of loss, held-out evaluation; the repeated-window gather; the induction test's scores and copy losses; one head knocked out; one head patched; one autoencoder training step; the autoencoder's firing counts over every position | `npm run test:gpu` | 36/36 in Chromium (Metal), 36/36 in WebKit; worst error on the microscope tests 1.2e-7 |
| The page's lifecycle and the microscope in a real browser (production build): reduced motion waits, a double-pressed RUN, replay, inspection holds the text, a prompt during a run, holding during kernel timing, the end of a run; the moment is named; the copying heads are marked; knocking them out stops copying (40 → 0 of 40); pinning a feature changes the writing; 25% more training; a new text | `npm run test:ui` | 17/17 pass |
| Load, train, the recorded run, Heads, Features, scrubbing, inspecting and holding in Chromium, WebKit and Firefox, one at a time | `npm run test:browsers` | pass in all three (Firefox on the CPU path) |
| Every state at 1440×900 and 390×844, no console errors or warnings, no horizontal overflow | `npm run shots` | 36 screenshots in `shots/v3/`, clean |
| Types, lint, dependency licences, audit | `npm run typecheck`, `npm run lint`, `npm run licences`, `npm audit` | clean; see `LICENCES.md` |

## Browsers

| browser | path | result |
| --- | --- | --- |
| Chromium (Playwright 1.63), headless | WebGPU on Metal | full run, moment, dissection; 36/36 kernel parity |
| WebKit (Playwright), headless | WebGPU | trains on the GPU, microscope panes work; 36/36 kernel parity |
| Firefox (Playwright), headless | no WebGPU adapter (blocklisted on headless macOS) | the CPU model trains; the recorded microscope works. Firefox logs its own "WebGPU is disabled by blocklist" warning when the page asks for an adapter; the page cannot probe for WebGPU without it |

## Measured

Apple GPU (Metal), Chromium headless, production build, `npm run perf`. Other sessions shared the
machine (load average around 17), so treat these as a busy laptop's numbers.

| | |
| --- | --- |
| first contentful paint | 36 ms |
| the recorded run on the trace | 144 ms after navigation (run.json, 17 KB gzipped) |
| the model's first text | 0.72 s |
| the induction moment, live | 26.9 s after navigation, at step 1,500 |
| frames through the moment, training at full speed | 357 frames: median 16.7 ms, p95 16.8 ms, max 16.8 ms, none over 50 ms |
| frames while dragging the scrubber, training | median 16.7 ms, p95 16.8 ms, max 50 ms |
| training | 260,000–300,000 tokens/s; the run of 2,250 steps takes about 31–42 s |
| the autoencoder, after the run | 3,000 steps in 8.6 s; knockouts and patching under a second |
| JavaScript at start | 95 KB gzipped; 580 KB in the whole build, almost all of it the other texts |
| downloads | features.json 75 KB gzipped when Features first opens; weights.bin 2.1 MB only on request |

## Development

```sh
npm run typecheck      # tsc, strict
npm run lint           # Biome
npm test               # CPU reference
npm run test:gpu       # kernels against the reference (--browser=webkit for Safari's engine)
npm run test:ui        # lifecycle and microscope, in Chromium
npm run test:browsers  # Chromium, WebKit, Firefox, one at a time
npm run perf           # paints, the opener, the moment, frames, dissection, downloads
npm run shots          # every state at both sizes into shots/v3/
npm run record         # re-record public/recorded/ from this code
npm run icons          # favicons, and og.png from a real run at its moment
```

The experiments behind the run's settings are development pages served by `npm run dev`:
- `tests/gpu/induction.html` (`?repeat=4&dropout=0.2&steps=2250`): the induction test on CPU
  snapshots;
- `tests/gpu/sae.html` (`?lambda=2`): autoencoder sparsity and labels;
- `tests/experiments/cpu-induction.exp.ts`: the CPU model.

Run the pages with `node scripts/run-page.mjs <page> "<query>"`.

```
src/cpu/         reference implementation: ops, the GPT, the trainer, the decoder with hooks, the autoencoder
src/gpu/         WGSL kernels, the GPU model with the induction test, the microscope (knockouts, patching), the autoencoder
src/model/       config, layout, batching (with repeated windows), the induction test, rule labels, sampling with edits
src/app/         engines, the sampler pool, the controller, the recording and its download
src/ui/          the scope with head lanes, the decode readout, the microscope panes, styles
public/recorded/ the recorded run
```

## Credits

- **Methods:** Olsson et al. 2022; Bricken et al. 2023; Cunningham et al. 2023; Meng et al. 2022
  (links above). Model shape after GPT-2 and Andrej Karpathy's nanoGPT and llm.c; dropout's
  integer hash is Chris Wellons's lowbias32.
- **Texts:** Project Gutenberg, public domain: ebooks 1041, 11, 1524, 1533, 1513, 1531 and 1532.
  Each was fetched once by `npm run fetch:texts`, with the header and footer stripped; see
  `src/samples/sources.json`.
- **Type:** B612 and B612 Mono, SIL Open Font Licence 1.1.
- **Licences:** everything is listed in `LICENCES.md`.

## Licence

MIT. Copyright (c) 2026 Harry Freeman.
