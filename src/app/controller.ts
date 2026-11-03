import { CPU_SHAPE, CPU_TRAIN, GPU_SHAPE, GPU_TRAIN, type ModelConfig, type TrainConfig, withVocab } from '../model/config';
import { type Corpus, splitCorpus } from '../model/data';
import { toHalf } from '../model/half';
import { initParams, paramLayout } from '../model/layout';
import { Rng } from '../model/rng';
import { dataSteps, dropoutFor, textTokensPerStep } from '../model/schedule';
import { type Tokenizer, buildTokenizer, normaliseText } from '../model/tokenizer';
import { type Acquired, acquireDevice } from '../gpu/context';
import type { Profile } from '../gpu/profile';
import type { StepStats } from '../gpu/trainer';
import { SAMPLES, type SampleId, sampleById } from '../samples';
import type { Reference } from '../ui/scope';
import { Decode } from '../ui/decode';
import { $, h, int, reducedMotion, seconds, sig3 } from '../ui/dom';
import { renderEngine } from '../ui/engine-panel';
import { Microscope } from '../ui/microscope';
import { Scope } from '../ui/scope';
import { TOP_K } from '../workers/protocol';
import { CpuEngine } from './engine-cpu';
import type { Engine, EngineEvent, Progress, RunState, Snapshot } from './engine';
import { type RecordedRun, loadRecorded, loadRecordedFeatures } from './recorded';
import { type Moment, type ProbeReading, findMoment, headName } from '../model/probe';
import type { GenerationEdit } from '../model/generate';
import { GpuEngine } from './engine-gpu';
import { Cancelled, type HeadAttention, PRIORITY, type Sample, SamplerPool, type Ticket } from './sampler';

/** ?record: run once, dissect, and hand the recording to scripts/record.mjs (no reference shown). */
const RECORDING = new URLSearchParams(location.search).has('record');
/** Characters written for the microscope's comparisons. */
const WRITE_LENGTH = 320;
const MIN_CHARS = 600;
const MAX_CHARS = 5_000_000;
/** Characters drawn at every checkpoint. */
const SAMPLE_LENGTH = 480;
/** Every checkpoint is sampled with the same random numbers. */
const SAMPLE_SEED = 1609;
const INIT_SEED = 1;
const RUN_SEED = 2026;
/** Checkpoints whose weights are kept (half precision, ~1.6 MB each for the GPU model). */
const MAX_CHECKPOINTS = 80;

type Choice = SampleId | 'own';
type KeyState = 'arming' | 'armed' | 'running' | 'held' | 'done' | 'fault';

interface Checkpoint {
  readonly step: number;
  readonly key: string;
  readonly half: Uint16Array;
  sample: Sample | null;
  /** The sampling settings `sample` was drawn with. */
  epoch: number;
  ticket: Ticket<Sample> | null;
  urgent: boolean;
}

interface Session {
  readonly id: number;
  readonly engine: Engine;
  readonly tok: Tokenizer;
  readonly cfg: ModelConfig;
  readonly corpus: Corpus;
  /** The token every sample starts from when there is no prompt: a line break if the text has one. */
  readonly start: number;
  readonly checkpoints: Checkpoint[];
  epoch: number;
  prompt: number[];
  promptChars: string[];
  dropped: number;
  temperature: number;
  live: boolean;
  cursor: number;
  shown: number;
  progress: Progress | null;
  last: StepStats | null;
  heldOut: number | null;
  sixty: { step: number; loss: number; heldOut: number | null; tps: number } | null;
  profile: Profile | null;
  profiling: boolean;
  profileError: string | null;
  inspect: { index: number; ticket: Ticket<HeadAttention> | null; timer: number } | null;
  attention: { key: string; index: number; value: HeadAttention } | null;
  /** The induction test's readings, and the moment a head learned to copy. */
  probes: ProbeReading[];
  moment: Moment | null;
  dissecting: boolean;
  /** Every step's loss and learning rate, and every held-out reading, as [step, value]. */
  trace: { steps: [number, number][]; lr: [number, number][]; evals: [number, number][] };
}

export interface Platform {
  readonly gpu: Acquired | null;
  readonly gpuError: string | null;
  readonly forceCpu: boolean;
}

export class Controller {
  private readonly pool: SamplerPool;
  private readonly scope: Scope;
  private readonly decode: Decode;
  private readonly micro: Microscope;
  private selectedHead: number | null = null;
  private inspectSummary: string | null = null;
  private recorded: RecordedRun | null = null;
  private readonly texts = new Map<SampleId, string>();
  private choice: Choice = 'sonnets';
  private ownText = '';
  private session: Session | null = null;
  private sessions = 0;
  private starting = false;
  private loading = false;
  private useCpu: boolean;
  private gpu: Acquired | null;
  private gpuLost = false;
  private key: KeyState = 'arming';
  private queued = false;
  private settingsTimer = 0;
  private systemTimer = 0;

  constructor(private readonly platform: Platform) {
    this.useCpu = platform.forceCpu || platform.gpu === null;
    this.gpu = platform.gpu;
    this.watchDevice();
    const cores = navigator.hardwareConcurrency || 4;
    const lanes = Number(new URLSearchParams(location.search).get('lanes')) || Math.max(2, Math.min(this.useCpu ? 3 : 4, cores - 2));
    this.pool = new SamplerPool(lanes);
    this.scope = new Scope($<HTMLCanvasElement>('scope'), {
      scrub: (step) => this.scrub(step),
      step: (d) => this.stepCursor(d),
      live: () => this.goLive(),
      scaled: (t) => {
        $('scale').textContent = t;
      },
    });
    this.decode = new Decode($('text-body'), (i) => this.inspect(i));
    this.micro = new Microscope({
      write: (which, edit, weights, kind) => this.writeWith(which, edit, weights, kind),
      copyPrompt: (which) => this.copyPrompt(which),
      selectHead: (k) => {
        this.selectedHead = k;
        this.lightAttention();
      },
      livePatterns: () => this.session?.engine.probePatterns() ?? Promise.resolve(null),
    });
    this.buildSoftkeys();
    this.bind();
    this.describeDevice();
    this.decode.message('Loading the instrument…');
    this.setKey('arming');
    void this.loadRecording();
  }

  // ------------------------------------------------------------------ platform

  private watchDevice(): void {
    const gpu = this.gpu;
    if (!gpu) return;
    void gpu.device.lost.then(() => {
      if (this.gpu === gpu) this.gpuLost = true;
    });
  }

  private describeDevice(params?: number): void {
    const shape = this.useCpu ? CPU_SHAPE : GPU_SHAPE;
    const p = params ?? paramLayout(withVocab(shape, 64)).count;
    const where = this.useCpu
      ? `No WebGPU here · <b>CPU</b> · ${int(p)} parameters`
      : `WebGPU <b>${this.gpu?.description ?? ''}</b> · ${int(p)} parameters`;
    $('device').innerHTML = where;
    $('device').title = this.useCpu ? (this.platform.gpuError ?? '') : '';
    const narrow = window.matchMedia('(max-width: 900px)').matches;
    $('what').textContent = this.useCpu
      ? `A small transformer learning to write like the text ${narrow ? 'below' : 'on the right'}, from scratch. This browser offers no WebGPU, so it trains a smaller model on the CPU.`
      : `A small transformer learning to write like the text ${narrow ? 'below' : 'on the right'}, from scratch, on your GPU.`;
  }

  // ------------------------------------------------------------------ the source

  private buildSoftkeys(): void {
    const box = $('softkeys');
    const keys: [Choice, string][] = [...SAMPLES.map((s) => [s.id, s.key] as [Choice, string]), ['own', 'Paste']];
    for (const [id, label] of keys) {
      const b = h('button', { type: 'button', class: 'key', role: 'radio', 'aria-checked': String(id === this.choice), 'data-id': id }, label);
      b.addEventListener('click', () => void this.choose(id, true));
      box.append(b);
    }
    box.addEventListener('keydown', (e) => {
      const ids = keys.map(([id]) => id);
      const at = ids.indexOf(this.choice);
      const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
      if (!d || this.locked) return;
      e.preventDefault();
      const next = ids[(at + d + ids.length) % ids.length];
      void this.choose(next, true).then(() => (box.querySelector(`[data-id="${next}"]`) as HTMLElement | null)?.focus());
    });
  }

  private get locked(): boolean {
    return this.starting || this.key === 'running';
  }

  private get sourceText(): HTMLTextAreaElement {
    return $<HTMLTextAreaElement>('source-text');
  }

  /** Pick a text. Mid-run the keys are locked; after a run, picking a text starts a new one. */
  async choose(id: Choice, user = false): Promise<void> {
    if (this.locked || this.loading) return;
    if (this.choice === 'own') this.ownText = this.sourceText.value;
    this.choice = id;
    for (const b of $('softkeys').children) b.setAttribute('aria-checked', String((b as HTMLElement).dataset.id === id));
    const ta = this.sourceText;
    if (id === 'own') {
      ta.value = this.ownText;
      ta.readOnly = false;
      ta.placeholder = `Paste any text: a novel, letters, lyrics, code. At least ${int(MIN_CHARS)} characters; the more the better.`;
      if (user) ta.focus();
    } else {
      const sample = sampleById(id);
      if (!sample) return;
      let text = this.texts.get(id);
      if (text === undefined) {
        this.loading = true;
        $('source-credit').textContent = 'Loading the text…';
        this.setKey('arming', 'loading text');
        try {
          text = normaliseText(await sample.load());
          this.texts.set(id, text);
        } catch (e) {
          $('source-credit').replaceChildren(h('span', { class: 'warn' }, `Could not load the text (${String(e)}). Check the connection and pick it again.`));
          this.setKey('armed', 'no text');
          return;
        } finally {
          this.loading = false;
        }
      }
      if (this.choice !== id) return;
      ta.value = text;
      ta.readOnly = false;
      this.scrollSourceToText(text);
    }
    this.describeText();
    // After a run (or before one, when picked by hand) a new text means a new run.
    if (user && id !== 'own') {
      this.endSession();
      if (this.textReady) void this.start();
    }
  }

  /** Show the source from its first real line of text, past any title page. */
  private scrollSourceToText(text: string): void {
    const ta = this.sourceText;
    const lines = text.split('\n');
    const first = lines.findIndex((l) => l.trim().length >= 24);
    requestAnimationFrame(() => {
      const css = getComputedStyle(ta);
      const lh = parseFloat(css.lineHeight) || 16;
      // Put a blank line (or the text itself) flush at the top, so no half-line peeks in above it.
      const top = first > 0 && lines[first - 1].trim() === '' ? first - 1 : Math.max(0, first);
      ta.scrollTop = top === 0 ? 0 : parseFloat(css.paddingTop) + top * lh;
    });
  }

  private get textReady(): boolean {
    return [...normaliseText(this.sourceText.value)].length >= MIN_CHARS;
  }

  private describeText(): void {
    const text = normaliseText(this.sourceText.value);
    const n = [...text].length;
    const tok = buildTokenizer(text.slice(0, MAX_CHARS));
    $('source-count').textContent = `${int(n)} chars · ${tok.size} distinct`;
    const credit = $('source-credit');
    if (this.choice === 'own') {
      credit.textContent = 'Your text. It stays in this tab.';
      if (n < MIN_CHARS) credit.append(' ', h('span', { class: 'warn' }, `It needs at least ${int(MIN_CHARS)} characters.`));
      if (n > MAX_CHARS) credit.append(' ', h('span', { class: 'warn' }, `Only the first ${int(MAX_CHARS)} are used.`));
    } else credit.textContent = sampleById(this.choice)?.credit ?? '';
    if (tok.unknown !== null) credit.append(' The rarest characters share one token.');
    const shape = this.useCpu ? CPU_SHAPE : GPU_SHAPE;
    this.describeDevice(paramLayout(withVocab(shape, Math.max(2, tok.size))).count);
    if (!this.session && !this.starting && !this.loading) {
      this.setKey('armed', n < MIN_CHARS ? `needs ${int(MIN_CHARS)} characters` : undefined);
      // Before a run, the screen already shows its frame: the chance line and the planned length.
      const train = this.useCpu ? CPU_TRAIN : GPU_TRAIN;
      const steps = n >= MIN_CHARS ? dataSteps(Math.floor(n * 0.9), textTokensPerStep(train, shape.context), train) : train.minSteps;
      this.scope.reset(tok.size, steps, train.peakLr);
      $('where').textContent = 'armed';
      this.decode.message(
        n < MIN_CHARS
          ? `Paste at least ${int(MIN_CHARS)} characters into the source ${this.narrow ? 'below' : 'on the right'}, then press RUN.`
          : `Press RUN, or the space bar, to train a new model on ${this.choice === 'own' ? 'your text' : (sampleById(this.choice)?.title ?? 'this text')}. It starts from random numbers; its writing appears here as it learns.`,
      );
    }
  }

  private get narrow(): boolean {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  // ------------------------------------------------------------------ controls

  private bind(): void {
    this.sourceText.addEventListener('input', () => {
      if (this.locked) return;
      if (this.choice !== 'own') {
        this.choice = 'own';
        for (const b of $('softkeys').children) b.setAttribute('aria-checked', String((b as HTMLElement).dataset.id === 'own'));
      }
      this.ownText = this.sourceText.value;
      if (this.session && this.key !== 'running') this.endSession();
      window.clearTimeout(this.settingsTimer);
      this.settingsTimer = window.setTimeout(() => this.describeText(), 250);
    });
    $('runstop').addEventListener('click', () => void this.runStop());
    $('live').addEventListener('click', () => this.goLive());
    document.addEventListener('keydown', (e) => {
      const t = e.target as HTMLElement;
      if (e.key !== ' ' || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      if (t.closest('input, textarea, button, select, [contenteditable], [role="slider"], [role="group"]')) return;
      e.preventDefault();
      void this.runStop();
    });

    const temp = $<HTMLInputElement>('temp');
    temp.addEventListener('input', () => {
      $('temp-out').textContent = Number(temp.value).toFixed(2);
      this.settingsChanged();
    });
    $<HTMLInputElement>('prompt').addEventListener('input', () => this.settingsChanged());
    window.addEventListener('resize', () => this.describeDevice(this.session?.engine.info.paramCount));

    const system = $('system');
    new IntersectionObserver((entries) => {
      window.clearInterval(this.systemTimer);
      if (entries.some((e) => e.isIntersecting)) {
        this.renderSystem();
        this.systemTimer = window.setInterval(() => this.renderSystem(), 700);
      }
    }).observe(system);
  }

  private setKey(state: KeyState, note?: string): void {
    this.key = state;
    const b = $<HTMLButtonElement>('runstop');
    b.dataset.state = state;
    const words: Record<KeyState, [string, string]> = {
      arming: ['Arming', ''],
      armed: ['Run', 'press to train'],
      running: ['Running', 'press to hold'],
      held: ['Held', 'press to run'],
      done: ['Done', 'press to train 25% more'],
      fault: ['Fault', ''],
    };
    const [label, action] = words[state];
    $('runstop-label').textContent = label;
    $('runstop-state').textContent = note ?? action;
    b.disabled = state === 'arming' || state === 'fault' || (state === 'armed' && !this.textReady);
    b.setAttribute('aria-label', `${label}${note ?? action ? `: ${note ?? action}` : ''}. Space bar also runs and holds.`);
    for (const k of $('softkeys').children) (k as HTMLButtonElement).disabled = this.locked;
    this.sourceText.readOnly = this.locked;
  }

  async runStop(): Promise<void> {
    const s = this.session;
    switch (this.key) {
      case 'armed':
        await this.start();
        break;
      case 'running':
        s?.engine.pause();
        break;
      case 'held':
        s?.engine.resume();
        break;
      case 'done':
        if (s?.progress) s.engine.extend(Math.max(300, Math.round(s.progress.totalSteps * 0.25)));
        break;
      default:
        break;
    }
  }

  // ------------------------------------------------------------------ a run

  async start(): Promise<void> {
    if (this.session || this.starting || !this.textReady) return;
    this.starting = true;
    this.setKey('arming', 'compiling kernels');
    try {
      await this.startSession(normaliseText(this.sourceText.value).slice(0, MAX_CHARS));
    } finally {
      this.starting = false;
      const s = this.session as Session | null;
      if (!s) this.describeText();
      else this.setKey(s.engine.state === 'training' ? 'running' : 'armed');
    }
  }

  private async startSession(text: string): Promise<void> {
    const tok = buildTokenizer(text);
    const shape = this.useCpu ? CPU_SHAPE : GPU_SHAPE;
    const cfg = withVocab(shape, tok.size);
    const tokens = tok.encode(text);
    let corpus: Corpus;
    try {
      corpus = splitCorpus(tokens, cfg.context);
    } catch (e) {
      this.fault('That text is too short to train on.', String(e), false);
      return;
    }
    const train: TrainConfig = this.useCpu ? CPU_TRAIN : { ...GPU_TRAIN, dropout: dropoutFor(tokens.length) };
    const total = dataSteps(corpus.train.length, textTokensPerStep(train, cfg.context), train);
    const initial = initParams(paramLayout(cfg), INIT_SEED);
    // Step 0 is the initial weights, already here: start drawing its text while the kernels compile.
    const early = this.sessions + 1;
    this.pool.sample(cfg, { key: `${early}:0`, half: toHalf(initial) }, [tok.encode('\n')[0] ?? tokens[0]], SAMPLE_LENGTH, Number($<HTMLInputElement>('temp').value), SAMPLE_SEED, PRIORITY.interactive).result.then(
      (sample) => {
        const s = this.session;
        const cp0 = s?.id === early ? s.checkpoints.find((c) => c.step === 0) : undefined;
        if (s && cp0 && !cp0.sample && s.epoch === 0 && s.prompt.length === 0) {
          cp0.sample = sample;
          cp0.epoch = 0;
          this.paintMarks(s);
          this.refresh(s);
        } else this.earlySample = { session: early, sample };
      },
      () => undefined,
    );
    let engine: Engine;
    try {
      if (!this.useCpu && (this.gpuLost || !this.gpu)) {
        const acq = await acquireDevice();
        if ('error' in acq) throw new Error(acq.error);
        this.gpu = acq;
        this.gpuLost = false;
        this.watchDevice();
      }
      engine = this.useCpu || !this.gpu ? new CpuEngine(cfg, train, corpus, RUN_SEED, total) : await GpuEngine.create(this.gpu, cfg, train, corpus, initial, RUN_SEED, total);
    } catch (e) {
      this.fault('The GPU would not run the model.', `${String(e)}`, true);
      return;
    }
    const s: Session = {
      id: ++this.sessions,
      engine,
      tok,
      cfg,
      corpus,
      start: tok.encode('\n')[0] ?? tokens[0],
      checkpoints: [],
      epoch: 0,
      prompt: [],
      promptChars: [],
      dropped: 0,
      temperature: Number($<HTMLInputElement>('temp').value),
      live: true,
      cursor: 0,
      shown: -1,
      progress: null,
      last: null,
      heldOut: null,
      sixty: null,
      profile: null,
      profiling: false,
      profileError: null,
      inspect: null,
      attention: null,
      probes: [],
      moment: null,
      dissecting: false,
      trace: { steps: [], lr: [], evals: [] },
    };
    this.session = s;
    this.readSettings(s);
    // The checkpoint at step 0, from the weights the run starts with.
    this.addCheckpoint(s, { step: 0, params: initial });
    $('scope').setAttribute('aria-valuemax', String(total));
    this.scope.reset(cfg.vocabSize, total, train.peakLr);
    this.decode.resetRange();
    this.decode.message('Waiting for the first checkpoint…');
    this.scope.setShape(cfg.nLayer, cfg.nHead);
    this.micro.startLive(cfg, corpus.tokens, tok.chars, engine.info.kind === 'gpu');
    for (const f of document.querySelectorAll('.fault')) f.remove();
    engine.on((e) => this.onEngine(s, e));
    engine.start();
    this.describeDevice(engine.info.paramCount);
    (window as unknown as { __metrics?: unknown }).__metrics = undefined;
  }

  private endSession(): void {
    const s = this.session;
    if (!s) return;
    this.session = null;
    s.engine.dispose();
    for (const cp of s.checkpoints) cp.ticket?.cancel();
    s.inspect?.ticket?.cancel();
    this.micro.endLive();
    this.paintReadouts(null);
    this.describeText();
  }

  private onEngine(s: Session, e: EngineEvent): void {
    if (this.session !== s) return;
    switch (e.type) {
      case 'stats':
        s.progress = e.progress;
        if (e.steps.length) s.last = e.steps[e.steps.length - 1];
        if (e.evals.length) s.heldOut = e.evals[e.evals.length - 1].loss;
        this.scope.setTotal(e.progress.totalSteps);
        this.scope.addSteps(e.steps);
        this.scope.addEvals(e.evals);
        if (e.probes.length) this.onProbes(s, e.probes);
        for (const p of e.steps) {
          s.trace.steps.push([p.step, p.loss]);
          s.trace.lr.push([p.step, p.lr]);
        }
        for (const p of e.evals) s.trace.evals.push([p.step, p.loss]);
        if (!s.sixty && e.progress.elapsed >= 60) s.sixty = { step: e.progress.completed, loss: this.scope.latestLoss, heldOut: s.heldOut, tps: e.progress.tokensPerSecond };
        this.queuePaint();
        break;
      case 'milestone':
        this.addCheckpoint(s, e.snapshot);
        break;
      case 'state':
        this.onState(s, e.state);
        break;
      case 'error':
        this.fault('Training stopped.', e.message, !this.useCpu);
        break;
    }
  }

  private onState(s: Session, state: RunState): void {
    const map: Record<RunState, KeyState> = { training: 'running', paused: 'held', finished: 'done', failed: 'fault' };
    if (!this.starting) this.setKey(map[state]);
    this.queuePaint();
    if (state === 'finished') void this.dissect(s);
  }

  /** The induction test's new readings: onto the trace and lanes, and is this the moment? */
  private onProbes(s: Session, readings: readonly ProbeReading[]): void {
    s.probes.push(...readings);
    this.scope.addProbes(readings);
    this.micro.probes(readings);
    if (!s.moment) {
      const m = findMoment(s.probes, s.cfg);
      if (m) {
        s.moment = m;
        this.scope.setMoment(m, true);
        this.micro.setMoment(m);
        if (this.selectedHead === null) this.selectedHead = m.head;
      }
    }
  }

  /** When the run ends: knockouts, patching and the sparse autoencoder, on the GPU. */
  private async dissect(s: Session): Promise<void> {
    if (s.dissecting) return;
    s.dissecting = true;
    try {
      const d = await s.engine.dissect((p) => {
        if (this.session === s) this.micro.dissectProgress(p);
      });
      if (this.session !== s) return;
      if (d) {
        this.micro.dissected(d);
        if (RECORDING) void import('./record').then((r) => r.record(this.recordInput(s), d));
      }
    } catch (e) {
      if (this.session === s) this.micro.dissectFailed(e instanceof Error ? e.message : String(e));
    } finally {
      s.dissecting = false;
    }
  }

  /**
   * Writes with an intervention, for the microscope: from the live run's newest checkpoint or
   * the recorded model, with the prompt and temperature set above, always the same seed.
   */
  private recordedHalf: { weights: Float32Array; half: Uint16Array } | null = null;

  private async writeWith(which: 'live' | 'recorded', edit: GenerationEdit | null, weights: Float32Array | null, kind: 'free' | 'copy'): Promise<string | null> {
    const copy = kind === 'copy' ? this.copyPrompt(which) : null;
    if (kind === 'copy' && !copy) return null;
    const temperature = copy ? 0 : Number($<HTMLInputElement>('temp').value);
    const length = copy ? copy.expected.length : WRITE_LENGTH;
    const encode = (tok: Tokenizer, text: string) => {
      const index = new Map(tok.chars.map((c, i) => [c, i]));
      return Array.from(text).flatMap((c) => (index.has(c) ? [index.get(c) ?? 0] : []));
    };
    if (which === 'live') {
      const s = this.session;
      const cp = s?.checkpoints[s.checkpoints.length - 1];
      if (!s || !cp) return null;
      const prompt = copy ? encode(s.tok, copy.prompt) : this.promptFor(s);
      const t = this.pool.sample(s.cfg, { key: cp.key, half: cp.half }, prompt, length, temperature, SAMPLE_SEED, PRIORITY.interactive, edit ?? undefined);
      const r = await t.result.catch(() => null);
      return r ? Array.from(r.tokens, (k) => s.tok.chars[k] ?? '').join('') : null;
    }
    const run = this.recorded;
    if (!run || !weights || !this.recordedTok) return null;
    if (this.recordedHalf?.weights !== weights) this.recordedHalf = { weights, half: toHalf(weights) };
    const tok = this.recordedTok;
    const typed = encode(tok, copy ? copy.prompt : $<HTMLInputElement>('prompt').value);
    const prompt = typed.length ? typed : [tok.encode('\n')[0] ?? 0];
    const t = this.pool.sample(run.cfg, { key: 'recorded', half: this.recordedHalf.half }, prompt, length, temperature, SAMPLE_SEED, PRIORITY.interactive, edit ?? undefined);
    const r = await t.result.catch(() => null);
    return r ? Array.from(r.tokens, (k) => tok.chars[k] ?? '').join('') : null;
  }

  /**
   * The copying demonstration, the induction test made readable: a run of random characters
   * (any character of the text but the line break, as in the test), shown twice, then its first
   * few characters again. A model that copies finishes the run; the continuation is known, so
   * every character can be checked.
   */
  private copyPrompt(which: 'live' | 'recorded'): { prompt: string; expected: string } | null {
    const tok = which === 'live' ? this.session?.tok : this.recordedTok;
    const cfg = which === 'live' ? this.session?.cfg : this.recorded?.cfg;
    if (!tok || !cfg) return null;
    const chars = tok.chars.filter((c) => c !== '\n' && c !== '\r');
    if (chars.length < 8) return null;
    const rng = new Rng(31);
    const n = Math.floor(cfg.context / 4);
    let run = '';
    for (let i = 0; i < n; i++) run += chars[rng.int(chars.length)];
    const lead = 5;
    const prompt = run + run + run.slice(0, lead);
    const expected = run.repeat(4).slice(lead, lead + Math.min(n + 8, cfg.context - prompt.length - 1));
    return { prompt, expected };
  }

  private recordedTok: Tokenizer | null = null;

  private recordInput(s: Session): import('./record').RecordInput {
    const last = s.checkpoints[s.checkpoints.length - 1];
    return {
      text: this.choice === 'own' ? 'sonnets' : this.choice,
      info: s.engine.info,
      totalSteps: s.progress?.totalSteps ?? 0,
      seconds: s.progress?.elapsed ?? 0,
      trace: s.trace,
      probes: s.probes,
      moment: s.moment,
      sample: last?.sample ? Array.from(last.sample.tokens, (t) => s.tok.chars[t] ?? '').join('') : '',
      weights: last?.half ?? new Uint16Array(0),
    };
  }

  /** The recorded run: the reference trace at once, then the panes' recorded data. */
  private async loadRecording(): Promise<void> {
    if (RECORDING) return;
    try {
      const run = await loadRecorded();
      const sample = sampleById(run.text);
      if (!sample) return;
      const text = normaliseText(await sample.load());
      const tok = buildTokenizer(text);
      this.recorded = run;
      this.recordedTok = tok;
      const ref: Reference = {
        train: smoothTrace(run.trace.steps),
        copy: run.probes.map((p) => ({ step: p.step, value: p.copySecond })),
        probes: run.probes,
        moment: run.moment,
        total: run.totalSteps,
        heads: { L: run.cfg.nLayer, H: run.cfg.nHead },
      };
      this.scope.setReference(ref);
      if (!this.session) this.scope.setShape(run.cfg.nLayer, run.cfg.nHead);
      $('ref-legend').hidden = false;
      this.micro.setRecorded(run, tok.encode(text), tok.chars, loadRecordedFeatures());
    } catch {
      // Without the recording the page still trains and dissects its own run.
    }
  }

  private fault(title: string, message: string, offerCpu: boolean): void {
    for (const f of document.querySelectorAll('.fault')) f.remove();
    const box = h('div', { class: 'fault', role: 'alert' }, h('b', {}, title), ' ', message);
    if (offerCpu) {
      const b = h('button', { type: 'button', class: 'key' }, 'Train on the CPU instead');
      b.addEventListener('click', () => {
        this.endSession();
        this.useCpu = true;
        box.remove();
        this.describeText();
        void this.start();
      });
      box.append(h('div', {}, b));
    }
    document.querySelector('.decode')?.append(box);
    this.setKey('fault', 'see the message');
    if (!offerCpu) window.setTimeout(() => this.key === 'fault' && !this.session && this.setKey('armed'), 0);
  }

  // ------------------------------------------------------------------ checkpoints and sampling

  private addCheckpoint(s: Session, snap: Snapshot): void {
    if (s.checkpoints.some((c) => c.step === snap.step)) return;
    const cp: Checkpoint = { step: snap.step, key: `${s.id}:${snap.step}`, half: toHalf(snap.params), sample: null, epoch: -1, ticket: null, urgent: false };
    const early = this.earlySample;
    if (snap.step === 0 && early?.session === s.id && s.epoch === 0 && s.prompt.length === 0) {
      cp.sample = early.sample;
      cp.epoch = 0;
      this.earlySample = null;
    }
    s.checkpoints.push(cp);
    s.checkpoints.sort((a, b) => a.step - b.step);
    if (!s.live) s.cursor = s.checkpoints.indexOf(this.cursorCheckpoint(s) ?? cp);
    this.thin(s);
    this.schedule(s);
    this.paintMarks(s);
    this.refresh(s);
  }

  /** Keep memory bounded on very long runs: drop the checkpoint closest to its neighbour. */
  private thin(s: Session): void {
    const cps = s.checkpoints;
    while (cps.length > MAX_CHECKPOINTS) {
      let drop = 1;
      let gap = Infinity;
      for (let i = 1; i < cps.length - 1; i++) {
        const g = (cps[i + 1].step - cps[i - 1].step) / Math.max(1, cps[i].step);
        if (g < gap && i !== s.cursor) {
          gap = g;
          drop = i;
        }
      }
      cps[drop].ticket?.cancel();
      cps.splice(drop, 1);
      if (s.cursor >= drop) s.cursor = Math.max(0, s.cursor - 1);
    }
  }

  private cursorCheckpoint(s: Session): Checkpoint | null {
    return s.live ? (s.checkpoints[s.checkpoints.length - 1] ?? null) : (s.checkpoints[s.cursor] ?? null);
  }

  private promptFor(s: Session): number[] {
    return s.prompt.length ? s.prompt : [s.start];
  }

  /** Draw a sample for every checkpoint that lacks one at the current settings: the one under the cursor first, then the newest. */
  private schedule(s: Session): void {
    const target = this.cursorCheckpoint(s);
    for (const cp of s.checkpoints) {
      const fresh = cp.sample !== null && cp.epoch === s.epoch;
      if (fresh) continue;
      const urgent = cp === target;
      const priority = urgent ? PRIORITY.interactive : PRIORITY.background + cp.step / 1e7;
      if (cp.ticket) {
        // Work already under way always finishes; waiting work follows the cursor.
        if (!cp.ticket.started && cp.urgent !== urgent) cp.ticket.reprioritize(priority);
        cp.urgent = urgent;
        continue;
      }
      const epoch = s.epoch;
      const ticket = this.pool.sample(s.cfg, { key: cp.key, half: cp.half }, this.promptFor(s), SAMPLE_LENGTH, s.temperature, SAMPLE_SEED, priority);
      cp.ticket = ticket;
      cp.urgent = urgent;
      ticket.result.then(
        (sample) => {
          if (this.session !== s || cp.ticket !== ticket) return;
          cp.ticket = null;
          if (epoch !== s.epoch) return;
          cp.sample = sample;
          cp.epoch = epoch;
          this.paintMarks(s);
          this.refresh(s);
        },
        (e: unknown) => {
          if (cp.ticket === ticket) cp.ticket = null;
          if (e instanceof Cancelled || this.session !== s) return;
          $('text-note').textContent = `Could not draw the text for step ${int(cp.step)}: ${String(e)}`;
        },
      );
    }
  }

  private paintMarks(s: Session): void {
    this.scope.setMarks(s.checkpoints.map((c) => ({ step: c.step, ready: c.sample !== null && c.epoch === s.epoch })));
  }

  private readSettings(s: Session): void {
    s.temperature = Number($<HTMLInputElement>('temp').value);
    const text = $<HTMLInputElement>('prompt').value;
    const tokens: number[] = [];
    const chars: string[] = [];
    let dropped = 0;
    for (const ch of normaliseText(text)) {
      const t = s.tok.encode(ch);
      if (t.length && (s.tok.unknown === null || t[0] !== s.tok.unknown || ch === s.tok.chars[s.tok.unknown])) {
        tokens.push(t[0]);
        chars.push(ch);
      } else dropped++;
    }
    s.prompt = tokens;
    s.promptChars = chars;
    s.dropped = dropped;
  }

  /** A new prompt or temperature re-draws every checkpoint, the one in view first. */
  private settingsChanged(): void {
    window.clearTimeout(this.settingsTimer);
    this.settingsTimer = window.setTimeout(() => {
      const s = this.session;
      if (!s) return;
      const before = `${s.temperature}|${s.prompt.join(',')}`;
      this.readSettings(s);
      if (`${s.temperature}|${s.prompt.join(',')}` === before) return;
      s.epoch++;
      s.attention = null;
      for (const cp of s.checkpoints) {
        cp.ticket?.cancel();
        cp.ticket = null;
      }
      this.schedule(s);
      this.paintMarks(s);
      this.refresh(s);
    }, 300);
  }

  // ------------------------------------------------------------------ the cursor

  private scrub(step: number): void {
    const s = this.session;
    if (!s || s.checkpoints.length === 0) return;
    let best = 0;
    for (let i = 1; i < s.checkpoints.length; i++) if (Math.abs(s.checkpoints[i].step - step) < Math.abs(s.checkpoints[best].step - step)) best = i;
    this.moveCursor(s, best);
  }

  private stepCursor(delta: number): void {
    const s = this.session;
    if (!s || s.checkpoints.length === 0) return;
    const from = s.live ? s.checkpoints.length - 1 : s.cursor;
    this.moveCursor(s, Math.max(0, Math.min(s.checkpoints.length - 1, from + delta)));
  }

  private moveCursor(s: Session, index: number): void {
    const last = s.checkpoints.length - 1;
    const live = index === last && this.key === 'running';
    if (live) {
      this.goLive();
      return;
    }
    s.live = false;
    s.cursor = index;
    this.schedule(s);
    this.refresh(s);
  }

  private goLive(): void {
    const s = this.session;
    if (!s) return;
    s.live = true;
    this.schedule(s);
    this.refresh(s);
  }

  /** Show the text for the cursor's checkpoint (or, following the run, the newest one drawn). */
  private refresh(s: Session): void {
    if (this.session !== s) return;
    const cps = s.checkpoints;
    let index = -1;
    if (s.live) {
      // The newest checkpoint that has any text; after a change of settings its old text stays
      // up (marked as being re-drawn) rather than jumping back to an older checkpoint.
      for (let i = cps.length - 1; i >= 0 && index < 0; i--) if (cps[i].sample) index = i;
      // While a character is being inspected the text holds still (training carries on), so the
      // heads and the glyph under the pointer stay about the same text.
      const held = cps[s.shown];
      if (this.decode.inspected !== null && held?.sample && held.epoch === s.epoch) index = s.shown;
    } else index = cps[s.cursor]?.sample ? s.cursor : -1;
    const target = s.live ? cps.length - 1 : s.cursor;
    const cp = cps[index];
    const note = $('text-note');
    const behind = target >= 0 && index !== target ? cps[target] : null;
    if (this.inspectSummary && this.decode.inspected !== null) note.textContent = this.inspectSummary;
    else if (behind && !s.live) note.textContent = `Drawing the text for step ${int(behind.step)}…`;
    else if (cp && cp.epoch !== s.epoch) note.textContent = 'Re-drawing every checkpoint with these settings…';
    else note.textContent = s.dropped ? `${s.dropped} character${s.dropped === 1 ? '' : 's'} of the prompt never appear in the text, so the model cannot read them; left out.` : '';
    note.title = note.textContent ?? '';

    const where = $('where');
    const total = cps.length ? cps[cps.length - 1].step : 0;
    if (!cp) {
      where.textContent = s.live ? 'live · waiting for a checkpoint' : 'replay';
    } else {
      const holding = s.live && index !== cps.length - 1 && this.decode.inspected !== null;
      where.replaceChildren(
        h('span', { class: s.live ? 'live' : 'replay' }, s.live ? 'LIVE' : 'REPLAY'),
        ` step ${int(cp.step)}${s.live ? '' : ` of ${int(total)}`}`,
        holding ? h('span', { class: 'holding' }, ' · text held', h('span', { class: 'l' }, ' while you point')) : '',
      );
    }
    $<HTMLButtonElement>('live').disabled = s.live;
    $('live').setAttribute('aria-pressed', String(s.live));
    const scopeEl = $('scope');
    const cursorStep = (s.live ? cp?.step : cps[s.cursor]?.step) ?? 0;
    scopeEl.setAttribute('aria-valuenow', String(cursorStep));
    scopeEl.setAttribute('aria-valuetext', `step ${int(cursorStep)}${s.live ? ', following the run' : ''}`);
    this.scope.setCursor(cp || cps[s.cursor] ? cursorStep : null, s.live);

    if (!cp?.sample || index === s.shown && s.attention?.key === `${cp.key}|${s.epoch}`) {
      if (!cp) this.decode.message(cps.length ? 'Drawing the first text…' : 'Waiting for the first checkpoint…');
      return;
    }
    if (index !== s.shown || this.decode.length === 0 || cp.epoch !== this.shownEpoch) {
      s.shown = index;
      this.shownEpoch = cp.epoch;
      this.decode.show({
        key: cp.key,
        settings: `${cp.epoch}`,
        prompt: cp.epoch === s.epoch ? s.promptChars : [],
        chars: Array.from(cp.sample.tokens, (t) => s.tok.chars[t] ?? ''),
        probs: cp.sample.probs,
      });
      // Keep the heads showing something live: the last character, unless one is being inspected.
      this.requestAttention(s, this.decode.inspected ?? Math.max(0, (this.decode.length || cp.sample.tokens.length) - 1), 120);
    }
  }

  private shownEpoch = -1;
  private earlySample: { session: number; sample: Sample } | null = null;

  // ------------------------------------------------------------------ attention

  private inspect(index: number | null): void {
    const s = this.session;
    if (!s) return;
    const cp = s.checkpoints[s.shown];
    if (!cp?.sample) return;
    this.requestAttention(s, index ?? Math.max(0, (this.decode.length || cp.sample.tokens.length) - 1), index === null ? 200 : 40);
    // Letting go of the text lets the live view catch up with the run.
    if (index === null && s.live) queueMicrotask(() => this.refresh(s));
  }

  private requestAttention(s: Session, index: number, delay: number): void {
    const cp = s.checkpoints[s.shown];
    if (!cp?.sample) return;
    const sample = cp.sample;
    const key = `${cp.key}|${cp.epoch}`;
    if (s.attention && s.attention.key === key && s.attention.index === index) {
      this.showAttention(s, cp, s.attention.value, index);
      return;
    }
    if (s.inspect) {
      window.clearTimeout(s.inspect.timer);
      s.inspect.ticket?.cancel();
    }
    const job: { index: number; ticket: Ticket<HeadAttention> | null; timer: number } = { index, ticket: null, timer: 0 };
    s.inspect = job;
    job.timer = window.setTimeout(() => {
      const prompt = cp.epoch === s.epoch ? this.promptFor(s) : [s.start];
      job.ticket = this.pool.attention(s.cfg, { key: cp.key, half: cp.half }, prompt, Array.from(sample.tokens), index);
      job.ticket.result.then(
        (value) => {
          if (this.session !== s || s.inspect !== job) return;
          s.attention = { key, index, value };
          if (s.checkpoints[s.shown] === cp) this.showAttention(s, cp, value, index);
        },
        (e: unknown) => {
          if (!(e instanceof Cancelled) && this.session === s) {
            this.inspectSummary = `Could not compute attention: ${String(e)}`;
            this.refresh(s);
          }
        },
      );
    }, delay);
  }

  private showAttention(s: Session, cp: Checkpoint, a: HeadAttention, index: number): void {
    const sample = cp.sample;
    if (!sample) return;
    const prompt = cp.epoch === s.epoch ? this.promptFor(s) : [s.start];
    const trimmed = prompt.slice(-(s.cfg.context - 1));
    const seq = [...trimmed, ...sample.tokens].map((t) => s.tok.chars[t] ?? '');
    const alternatives: [string, number][] = [];
    for (let k = 0; k < TOP_K; k++) {
      const t = sample.tops[(index * TOP_K + k) * 2];
      if (t >= 0) alternatives.push([s.tok.chars[t] ?? '', sample.tops[(index * TOP_K + k) * 2 + 1]]);
    }
    const say = (c: string) => (c === ' ' ? 'a space' : c === '\n' ? 'a line break' : `“${c}”`);
    const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
    const written = s.tok.chars[sample.tokens[index]] ?? '';
    const others = alternatives.filter(([c, p]) => c !== written && p >= 0.005).slice(0, 2);
    const lit = this.selectedHead === null ? 'lit: every head, averaged' : `lit: ${headName(this.selectedHead, s.cfg.nHead)}`;
    this.inspectSummary = `drew ${say(written)} ${pct(sample.probs[index])}${others.length ? ` · others ${others.map(([c, p]) => `${say(c)} ${pct(p)}`).join(' ')}` : ''} · ${lit}`;
    void seq;
    void a;
    this.refresh(s);
    this.lightAttention();
  }

  /** Light the output with the selected head's attention, or the mean of all of them. */
  private lightAttention(): void {
    const s = this.session;
    const a = s?.attention;
    if (!s || !a || this.decode.inspected === null) {
      this.decode.unlight();
      return;
    }
    const heads = s.cfg.nLayer * s.cfg.nHead;
    const k = this.selectedHead;
    const w = new Float32Array(a.value.length);
    for (let j = 0; j < a.value.length; j++) {
      if (k !== null) w[j] = a.value.weights[k * a.value.length + j];
      else {
        let m = 0;
        for (let x = 0; x < heads; x++) m += a.value.weights[x * a.value.length + j];
        w[j] = m / heads;
      }
    }
    const cp = s.checkpoints[s.shown];
    const prompt = cp && cp.epoch === s.epoch ? this.promptFor(s) : [s.start];
    const seen = Math.min(prompt.length, s.cfg.context - 1);
    // The default start token is not shown, so only a typed prompt has glyphs to light.
    this.decode.light(s.prompt.length && cp?.epoch === s.epoch ? seen : seen, a.value.start, w);
  }

  // ------------------------------------------------------------------ readouts

  private queuePaint(): void {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      this.paintReadouts(this.session);
    });
  }

  private paintReadouts(s: Session | null): void {
    const p = s?.progress ?? null;
    const set = (id: string, text: string) => {
      $(id).textContent = text;
    };
    if (!s || !p) {
      for (const id of ['r-tps', 'r-loss', 'r-val', 'r-lr', 'r-grad']) set(id, '–');
      set('r-step', '0');
      set('r-time', '0.0 s');
      set('r-passes', '0.0');
      return;
    }
    const step = $('r-step');
    step.textContent = int(p.completed);
    step.append(h('small', {}, ` / ${int(p.totalSteps)}`));
    set('r-time', seconds(p.elapsed));
    set('r-tps', p.tokensPerSecond > 0 ? sig3(p.tokensPerSecond) : '–');
    set('r-passes', p.epochs.toFixed(1));
    const loss = this.scope.latestLoss;
    set('r-loss', Number.isFinite(loss) ? loss.toFixed(3) : '–');
    set('r-val', s.corpus.val === null ? 'n/a' : s.heldOut !== null ? s.heldOut.toFixed(3) : '–');
    set('r-lr', s.last ? s.last.lr.toExponential(2) : '–');
    set('r-grad', s.last ? s.last.gradNorm.toFixed(3) : '–');
    (window as unknown as { __metrics: unknown }).__metrics = {
      state: s.engine.state,
      kind: s.engine.info.kind,
      completed: p.completed,
      totalSteps: p.totalSteps,
      elapsed: p.elapsed,
      tokensPerSecond: p.tokensPerSecond,
      tokensSeen: p.tokensSeen,
      epochs: p.epochs,
      loss,
      heldOut: s.heldOut,
      sixty: s.sixty,
      lr: s.last?.lr ?? null,
      checkpoints: s.checkpoints.length,
      drawn: s.checkpoints.filter((c) => c.sample && c.epoch === s.epoch).length,
      sessions: this.sessions,
      params: s.engine.info.paramCount,
    };
  }

  // ------------------------------------------------------------------ the system page

  private async measure(): Promise<void> {
    const s = this.session;
    if (!s || s.profiling) return;
    s.profiling = true;
    s.profileError = null;
    this.renderSystem();
    try {
      s.profile = await s.engine.profile();
    } catch (e) {
      s.profileError = String(e);
    } finally {
      s.profiling = false;
      this.renderSystem();
    }
  }

  private renderSystem(): void {
    const body = $('system-body');
    const s = this.session;
    if (!s) {
      const shape = this.useCpu ? CPU_SHAPE : GPU_SHAPE;
      body.replaceChildren(
        h(
          'p',
          { class: 'kernel-note' },
          `Nothing is running. During a run this shows the model (${shape.nLayer} layers, ${shape.nHead} heads, ${shape.dModel} wide, ${shape.context} characters of context), every buffer on the ${this.useCpu ? 'CPU' : 'GPU'}, the optimiser’s state and the time each kernel takes.`,
        ),
      );
      return;
    }
    renderEngine(
      body,
      { info: s.engine.info, progress: s.progress, last: s.last, heldOut: s.heldOut, sixty: s.sixty, profile: s.profile, profiling: s.profiling, profileError: s.profileError },
      () => void this.measure(),
    );
  }

  /** Called once at load: train straight away if the visitor can see the page and has not asked for less motion. */
  autostart(): void {
    const go = () => {
      if (document.visibilityState === 'visible' && !reducedMotion() && this.key === 'armed') void this.start();
    };
    if (document.visibilityState === 'visible') go();
    else document.addEventListener('visibilitychange', go, { once: true });
  }
}

/** The recorded training loss, smoothed as the live trace is (an exponential average, bias-corrected). */
function smoothTrace(points: readonly (readonly [number, number])[]): { step: number; value: number }[] {
  let ema = 0;
  let n = 0;
  return points.map(([step, loss]) => {
    n++;
    ema = 0.96 * ema + 0.04 * loss;
    return { step, value: ema / (1 - 0.96 ** n) };
  });
}
