import { CPU_SHAPE, CPU_TRAIN, GPU_SHAPE, GPU_TRAIN, type TrainConfig, withVocab } from '../model/config';
import { type Corpus, splitCorpus } from '../model/data';
import { toHalf } from '../model/half';
import { initParams, paramLayout } from '../model/layout';
import { dataSteps, dropoutFor, textTokensPerStep } from '../model/schedule';
import { type Tokenizer, buildTokenizer } from '../model/tokenizer';
import { type Acquired, acquireDevice } from '../gpu/context';
import { sampleById } from '../samples';
import { Decode } from '../ui/decode';
import { $, h, int, reducedMotion } from '../ui/dom';
import { Microscope } from '../ui/microscope';
import { Scope } from '../ui/scope';
import { CpuEngine } from './engine-cpu';
import type { Engine, EngineEvent, RunState } from './engine';
import { type RecordedRun, loadRecordedFeatures, openRecording } from './recorded';
import { paintReadouts, renderSystem } from './readouts';
import { Replay } from './replay';
import { type Choice, MIN_CHARS, Source } from './source';
import { SAMPLE_LENGTH, SAMPLE_SEED, type Session, promptFor } from './session';
import { type CopyDemo, WRITE_LENGTH, copyDemo, encodeKnown, write } from './writing';
import { type ProbeReading, findMoment } from '../model/probe';
import type { GenerationEdit } from '../model/generate';
import { GpuEngine } from './engine-gpu';
import { PRIORITY, SamplerPool } from './sampler';

/** ?record: run once, dissect, and hand the recording to scripts/record.mjs (no reference shown). */
const RECORDING = new URLSearchParams(location.search).has('record');
const INIT_SEED = 1;
const RUN_SEED = 2026;

type KeyState = 'arming' | 'armed' | 'running' | 'held' | 'done' | 'fault';

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
  private readonly replay: Replay;
  private readonly source: Source;
  private recorded: RecordedRun | null = null;
  private recordedTok: Tokenizer | null = null;
  private session: Session | null = null;
  private sessions = 0;
  private starting = false;
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
      scrub: (step) => this.replay.scrub(step),
      step: (d) => this.replay.stepCursor(d),
      live: () => this.replay.goLive(),
      scaled: (t) => {
        $('scale').textContent = t;
      },
    });
    this.decode = new Decode($('text-body'), (i) => this.replay.inspect(i));
    this.replay = new Replay({ pool: this.pool, decode: this.decode, scope: this.scope, session: () => this.session, running: () => this.key === 'running' });
    this.micro = new Microscope({
      write: (which, edit, weights, kind) => this.writeWith(which, edit, weights, kind),
      copyPrompt: (which) => this.copyPrompt(which),
      selectHead: (k) => {
        this.replay.selectedHead = k;
        this.replay.lightAttention();
      },
      livePatterns: () => this.session?.engine.probePatterns() ?? Promise.resolve(null),
    });
    this.source = new Source({
      locked: () => this.locked,
      pick: (id) => this.choose(id, true),
      status: (state, note) => this.setKey(state, note),
      edited: () => {
        if (this.session && this.key !== 'running') this.endSession();
        window.clearTimeout(this.settingsTimer);
        this.settingsTimer = window.setTimeout(() => this.describeText(), 250);
      },
    });
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
    const where = this.useCpu ? `No WebGPU here · <b>CPU</b> · ${int(p)} parameters` : `WebGPU <b>${this.gpu?.description ?? ''}</b> · ${int(p)} parameters`;
    $('device').innerHTML = where;
    $('device').title = this.useCpu ? (this.platform.gpuError ?? '') : '';
    const narrow = window.matchMedia('(max-width: 900px)').matches;
    $('what').textContent = this.useCpu
      ? `A small transformer learning to write like the text ${narrow ? 'below' : 'on the right'}, from scratch. This browser offers no WebGPU, so it trains a smaller model on the CPU.`
      : `A small transformer learning to write like the text ${narrow ? 'below' : 'on the right'}, from scratch, on your GPU.`;
  }

  // ------------------------------------------------------------------ the source

  private get locked(): boolean {
    return this.starting || this.key === 'running';
  }

  /** Pick a text. Mid-run the keys are locked; after a run, picking a text starts a new one. */
  async choose(id: Choice, user = false): Promise<void> {
    if (this.locked || !(await this.source.select(id, user))) return;
    this.describeText();
    // After a run (or before one, when picked by hand) a new text means a new run.
    if (user && id !== 'own') {
      this.endSession();
      if (this.source.ready) void this.start();
    }
  }

  /** The text's count and credit, and what the instrument will do with it. */
  private describeText(): void {
    const { chars: n, tok } = this.source.describe();
    const shape = this.useCpu ? CPU_SHAPE : GPU_SHAPE;
    this.describeDevice(paramLayout(withVocab(shape, Math.max(2, tok.size))).count);
    if (!this.session && !this.starting && !this.source.busy) {
      this.setKey('armed', n < MIN_CHARS ? `needs ${int(MIN_CHARS)} characters` : undefined);
      // Before a run, the screen already shows its frame: the chance line and the planned length.
      const train = this.useCpu ? CPU_TRAIN : GPU_TRAIN;
      const steps = n >= MIN_CHARS ? dataSteps(Math.floor(n * 0.9), textTokensPerStep(train, shape.context), train) : train.minSteps;
      this.scope.reset(tok.size, steps, train.peakLr);
      $('where').textContent = 'armed';
      this.decode.message(
        n < MIN_CHARS
          ? `Paste at least ${int(MIN_CHARS)} characters into the source ${this.narrow ? 'below' : 'on the right'}, then press RUN.`
          : `Press RUN, or the space bar, to train a new model on ${this.source.choice === 'own' ? 'your text' : (sampleById(this.source.choice)?.title ?? 'this text')}. It starts from random numbers; its writing appears here as it learns.`,
      );
    }
  }

  private get narrow(): boolean {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  // ------------------------------------------------------------------ controls

  private bind(): void {
    $('runstop').addEventListener('click', () => void this.runStop());
    $('live').addEventListener('click', () => this.replay.goLive());
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
      this.replay.settingsChanged();
    });
    $<HTMLInputElement>('prompt').addEventListener('input', () => this.replay.settingsChanged());
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
    b.disabled = state === 'arming' || state === 'fault' || (state === 'armed' && !this.source.ready);
    b.setAttribute('aria-label', `${label}${(note ?? action) ? `: ${note ?? action}` : ''}. Space bar also runs and holds.`);
    this.source.lock(this.locked);
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
    if (this.session || this.starting || !this.source.ready) return;
    this.starting = true;
    this.setKey('arming', 'compiling kernels');
    try {
      await this.startSession(this.source.text());
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
    this.pool
      .sample(
        cfg,
        { key: `${early}:0`, half: toHalf(initial) },
        [tok.encode('\n')[0] ?? tokens[0]],
        SAMPLE_LENGTH,
        Number($<HTMLInputElement>('temp').value),
        SAMPLE_SEED,
        PRIORITY.interactive,
      )
      .result.then(
        (sample) => {
          const s = this.session;
          const cp0 = s?.id === early ? s.checkpoints.find((c) => c.step === 0) : undefined;
          if (s && cp0 && !cp0.sample && s.epoch === 0 && s.prompt.length === 0) {
            cp0.sample = sample;
            cp0.epoch = 0;
            this.replay.paintMarks(s);
            this.replay.refresh(s);
          } else this.replay.earlySample = { session: early, sample };
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
      engine =
        this.useCpu || !this.gpu
          ? new CpuEngine(cfg, train, corpus, RUN_SEED, total)
          : await GpuEngine.create(this.gpu, cfg, train, corpus, initial, RUN_SEED, total);
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
    this.replay.readSettings(s);
    // The checkpoint at step 0, from the weights the run starts with.
    this.replay.addCheckpoint(s, { step: 0, params: initial });
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
    delete window.__metrics;
  }

  private endSession(): void {
    const s = this.session;
    if (!s) return;
    this.session = null;
    s.engine.dispose();
    for (const cp of s.checkpoints) cp.ticket?.cancel();
    s.inspect?.ticket?.cancel();
    this.micro.endLive();
    paintReadouts(null, this.scope.latestLoss, this.sessions);
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
        if (!s.sixty && e.progress.elapsed >= 60)
          s.sixty = { step: e.progress.completed, loss: this.scope.latestLoss, heldOut: s.heldOut, tps: e.progress.tokensPerSecond };
        this.queuePaint();
        break;
      case 'milestone':
        this.replay.addCheckpoint(s, e.snapshot);
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
        if (this.replay.selectedHead === null) this.replay.selectedHead = m.head;
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

  private recordedHalf: { weights: Float32Array; half: Uint16Array } | null = null;

  /**
   * Writes with an intervention, for the microscope: from the live run's newest checkpoint or
   * the recorded model, with the prompt and temperature set above ('free'), or greedily
   * continuing the copying demonstration ('copy').
   */
  private async writeWith(
    which: 'live' | 'recorded',
    edit: GenerationEdit | null,
    weights: Float32Array | null,
    kind: 'free' | 'copy',
  ): Promise<string | null> {
    const copy = kind === 'copy' ? this.copyPrompt(which) : null;
    if (kind === 'copy' && !copy) return null;
    const temperature = copy ? 0 : Number($<HTMLInputElement>('temp').value);
    const length = copy ? copy.expected.length : WRITE_LENGTH;
    if (which === 'live') {
      const s = this.session;
      const cp = s?.checkpoints[s.checkpoints.length - 1];
      if (!s || !cp) return null;
      const prompt = copy ? encodeKnown(s.tok.chars, copy.prompt) : promptFor(s);
      return write(this.pool, s.cfg, { key: cp.key, half: cp.half }, s.tok.chars, prompt, length, temperature, edit);
    }
    const run = this.recorded;
    const tok = this.recordedTok;
    if (!run || !weights || !tok) return null;
    if (this.recordedHalf?.weights !== weights) this.recordedHalf = { weights, half: toHalf(weights) };
    const typed = encodeKnown(tok.chars, copy ? copy.prompt : $<HTMLInputElement>('prompt').value);
    const prompt = typed.length ? typed : [tok.encode('\n')[0] ?? 0];
    return write(this.pool, run.cfg, { key: 'recorded', half: this.recordedHalf.half }, tok.chars, prompt, length, temperature, edit);
  }

  private copyPrompt(which: 'live' | 'recorded'): CopyDemo | null {
    const tok = which === 'live' ? this.session?.tok : this.recordedTok;
    const cfg = which === 'live' ? this.session?.cfg : this.recorded?.cfg;
    return tok && cfg ? copyDemo(tok.chars, cfg.context) : null;
  }

  private recordInput(s: Session): import('./record').RecordInput {
    const last = s.checkpoints[s.checkpoints.length - 1];
    return {
      text: this.source.choice === 'own' ? 'sonnets' : this.source.choice,
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
      const { run, tok, tokens, reference } = await openRecording();
      this.recorded = run;
      this.recordedTok = tok;
      this.scope.setReference(reference);
      if (!this.session) this.scope.setShape(run.cfg.nLayer, run.cfg.nHead);
      $('ref-legend').dataset.state = 'ready';
      $('ref-legend').title = `A recorded run of this same code: ${run.recorded.slice(0, 10)}, ${run.device}`;
      this.micro.setRecorded(run, tokens, tok.chars, loadRecordedFeatures());
    } catch {
      // Without the recording the page still trains and dissects its own run; the legend says so.
      $('ref-legend').dataset.state = 'missing';
      $('ref-legend').title = 'The recorded run did not load';
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

  // ------------------------------------------------------------------ readouts

  private queuePaint(): void {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      paintReadouts(this.session, this.scope.latestLoss, this.sessions);
    });
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
    renderSystem(this.session, this.useCpu, () => void this.measure());
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
