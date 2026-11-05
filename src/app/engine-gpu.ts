import type { ModelConfig, TrainConfig } from '../model/config';
import type { Corpus } from '../model/data';
import { isCheckpoint, planSteps, textTokensPerStep } from '../model/schedule';
import { type Acquired, Gpu } from '../gpu/context';
import { PIPELINE_VARIANTS } from '../gpu/model';
import { type Profile, profileStep } from '../gpu/profile';
import { GpuTrainer } from '../gpu/trainer';
import { Microscope } from '../gpu/microscope';
import { GpuSae, saeDefaults } from '../gpu/sae';
import { sampleOffsets } from '../model/data';
import { Rng } from '../model/rng';
import {
  Clock,
  type DissectProgress,
  type Dissection,
  Emitter,
  type Engine,
  type EngineEvent,
  type EngineInfo,
  PROBE_EVERY,
  type RunState,
  type Snapshot,
} from './engine';

/** Steps kept queued on the GPU: enough to never starve it, few enough to stay responsive. */
const IN_FLIGHT = 3;
/** Steps measured before the time-based budget is fixed. */
const CALIBRATE_FROM = 10;
const CALIBRATE_TO = 50;

export class GpuEngine implements Engine {
  readonly info: EngineInfo;
  state: RunState = 'paused';
  private readonly trainer: GpuTrainer;
  private readonly emitter = new Emitter();
  private readonly clock = new Clock();
  /** What the person asked for: training or not. */
  private want = false;
  /** Whether the step loop is actually running. */
  private running = false;
  private loopDone: Promise<void> = Promise.resolve();
  private profiling: Promise<Profile> | null = null;
  private dissecting: Promise<Dissection> | null = null;
  private scope: Microscope | null = null;
  private sae: GpuSae | null = null;
  private reading = false;
  private plannedBy: 'data' | 'time' = 'data';
  private calibration: { elapsed: number; completed: number } | null = null;
  private calibrated = false;
  private completed = 0;
  private tokensPerSecond = 0;
  private disposed = false;
  private readonly onGpuError: (e: Event) => void;

  private constructor(
    private readonly gpu: Gpu,
    acq: Acquired,
    cfg: ModelConfig,
    train: TrainConfig,
    corpus: Corpus,
    initial: Float32Array,
    seed: number,
    totalSteps: number,
  ) {
    this.trainer = new GpuTrainer(gpu, cfg, train, corpus, initial, seed, totalSteps);
    const m = this.trainer.model;
    this.info = {
      kind: 'gpu',
      device: acq.description,
      cfg,
      train,
      paramCount: m.layout.count,
      memory: [...gpu.memory.entries()].map(([category, bytes]) => ({ category, bytes })),
      dispatchesPerStep: m.trainProgram.length,
      flopsPerStep: m.flopsPerStep,
    };
    acq.device.lost.then((info) => {
      if (!this.disposed) this.fail(`The GPU device was lost (${info.reason || 'unknown reason'}${info.message ? `: ${info.message}` : ''}).`);
    });
    this.onGpuError = (e) => {
      if (!this.disposed) this.fail(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`);
    };
    acq.device.addEventListener('uncapturederror', this.onGpuError);
  }

  static async create(
    acq: Acquired,
    cfg: ModelConfig,
    train: TrainConfig,
    corpus: Corpus,
    initial: Float32Array,
    seed: number,
    totalSteps: number,
  ): Promise<GpuEngine> {
    const gpu = new Gpu(acq.device);
    await gpu.prepare(PIPELINE_VARIANTS);
    return new GpuEngine(gpu, acq, cfg, train, corpus, initial, seed, totalSteps);
  }

  on(l: (e: EngineEvent) => void): () => void {
    return this.emitter.on(l);
  }

  private setState(s: RunState): void {
    this.state = s;
    this.emitter.emit({ type: 'state', state: s });
  }

  private fail(message: string): void {
    if (this.state === 'failed') return;
    this.want = false;
    this.running = false;
    this.clock.stop();
    this.setState('failed');
    this.emitter.emit({ type: 'error', message });
  }

  private get usable(): boolean {
    return !this.disposed && this.state !== 'failed';
  }

  private startLoop(): void {
    this.running = true;
    this.clock.start();
    this.loopDone = this.loop();
  }

  start(): void {
    this.resume();
  }

  resume(): void {
    if (!this.usable || this.trainer.step >= this.trainer.totalSteps) return;
    this.want = true;
    this.setState('training');
    // While the kernels are being timed, or the model dissected, the loop restarts when that ends.
    if (!this.running && !this.profiling && !this.dissecting) this.startLoop();
  }

  pause(): void {
    if (!this.want) return;
    this.want = false;
    if (this.profiling || this.dissecting) {
      this.setState('paused');
      return;
    }
    this.running = false;
    void this.loopDone.then(() => {
      this.clock.stop();
      if (this.state === 'training' && !this.want) this.setState('paused');
    });
  }

  /** More steps at the final learning rate: the schedule's length does not change. */
  extend(steps: number): void {
    this.trainer.totalSteps = Math.max(this.trainer.totalSteps, this.trainer.step) + steps;
    this.resume();
  }

  private async milestone(): Promise<void> {
    const snap = await this.trainer.snapshot();
    if (!this.disposed) this.emitter.emit({ type: 'milestone', snapshot: snap });
  }

  private async loop(): Promise<void> {
    const t = this.trainer;
    const device = this.gpu.device;
    const inflight: Promise<unknown>[] = [];
    const evalEvery = () => Math.max(25, Math.round(t.scheduleSteps / 50 / 25) * 25);
    let lastRead = performance.now();
    if (t.step === 0) void this.milestone();
    while (this.running && t.step < t.totalSteps) {
      if (t.step % evalEvery() === 0) t.submitEval();
      if (t.step % PROBE_EVERY === 0) t.submitProbe();
      t.submitStep();
      if (isCheckpoint(t.step, t.scheduleSteps) && t.step < t.totalSteps) void this.milestone();
      inflight.push(device.queue.onSubmittedWorkDone());
      if (inflight.length >= IN_FLIGHT) await inflight.shift();
      const now = performance.now();
      if (now - lastRead > 120) {
        lastRead = now;
        void this.read();
      }
    }
    await Promise.all(inflight);
    if (this.disposed) return;
    if (t.step >= t.totalSteps && this.state !== 'failed') {
      t.submitEval();
      t.submitProbe();
      await this.read(true);
      await this.milestone();
      this.running = false;
      this.want = false;
      this.clock.stop();
      if ((this.state as RunState) !== 'failed') this.setState('finished'); // may have failed during the awaits
    } else {
      await this.read(true);
    }
  }

  private async read(force = false): Promise<void> {
    if (this.reading && !force) return;
    while (this.reading) await new Promise((r) => setTimeout(r, 5));
    if (this.disposed) return;
    this.reading = true;
    try {
      const batch = await this.trainer.readStats();
      if (this.disposed) return;
      const bad = batch.steps.find((s) => !Number.isFinite(s.loss));
      if (bad) {
        this.fail(`Training diverged: the loss stopped being a number at step ${bad.step}.`);
        return;
      }
      this.completed = batch.completed;
      const sps = this.clock.mark(this.completed);
      if (sps > 0) this.tokensPerSecond = sps * this.trainer.tokensPerStep;
      this.calibrate();
      const tps = this.trainer.tokensPerStep;
      this.emitter.emit({
        type: 'stats',
        steps: batch.steps,
        evals: batch.evals,
        probes: batch.probes,
        progress: {
          completed: this.completed,
          totalSteps: this.trainer.totalSteps,
          elapsed: this.clock.elapsed,
          tokensPerSecond: this.tokensPerSecond,
          tokensSeen: this.completed * tps,
          // Passes over the text: the repeated random windows are not the text.
          epochs: (this.completed * textTokensPerStep(this.info.train, this.info.cfg.context)) / this.trainer.corpus.train.length,
          plannedBy: this.plannedBy,
        },
      });
    } finally {
      this.reading = false;
    }
  }

  /**
   * Once the speed is known, shorten the plan if the data budget would overrun the clock.
   * Measured on training time (pauses excluded) between two observed step counts.
   */
  private calibrate(): void {
    if (this.calibrated) return;
    const now = { elapsed: this.clock.elapsed, completed: this.completed };
    if (!this.calibration) {
      if (now.completed >= CALIBRATE_FROM) this.calibration = now;
      return;
    }
    const steps = now.completed - this.calibration.completed;
    const secs = now.elapsed - this.calibration.elapsed;
    if (steps < CALIBRATE_TO - CALIBRATE_FROM || secs < 0.25) return;
    const byTime = planSteps(steps / secs, this.info.train);
    if (byTime < this.trainer.totalSteps) {
      const plan = Math.max(byTime, this.trainer.step + 1);
      this.trainer.totalSteps = plan;
      this.trainer.scheduleSteps = plan;
      this.plannedBy = 'time';
    }
    this.calibrated = true;
  }

  async snapshot(): Promise<Snapshot> {
    // Never hand out weights while timing runs are moving them.
    if (this.profiling) await this.profiling.catch(() => undefined);
    return this.trainer.snapshot();
  }

  /**
   * Times every kernel of one step. The loop stops first and the optimiser state is restored
   * afterwards; pause, resume and "keep training" requests made meanwhile take effect then.
   */
  profile(): Promise<Profile> {
    if (this.profiling) return this.profiling;
    if (!this.usable) return Promise.reject(new Error('the engine has stopped'));
    const run = async (): Promise<Profile> => {
      this.running = false;
      await this.loopDone;
      if (this.dissecting) await this.dissecting.catch(() => undefined);
      this.clock.stop();
      try {
        return await profileStep(this.trainer.model);
      } finally {
        this.profiling = null;
        if (this.usable && this.want && !this.dissecting && this.trainer.step < this.trainer.totalSteps) this.startLoop();
      }
    };
    this.profiling = run();
    return this.profiling;
  }

  probePatterns(): Promise<Float32Array | null> {
    if (this.disposed) return Promise.resolve(null);
    return this.trainer.model.readProbePatterns();
  }

  /**
   * Takes the model apart at its current weights. The loop stops first (like kernel timing) and
   * carries on afterwards if training was wanted; nothing here changes the model's weights.
   */
  dissect(onProgress: (p: DissectProgress) => void): Promise<Dissection> {
    if (this.dissecting) return this.dissecting;
    if (!this.usable) return Promise.reject(new Error('the engine has stopped'));
    const run = async (): Promise<Dissection> => {
      this.running = false;
      await this.loopDone;
      if (this.profiling) await this.profiling.catch(() => undefined);
      this.clock.stop();
      try {
        const t = this.trainer;
        const m = t.model;
        const step = t.step;
        onProgress({ stage: 'heads' });
        this.scope ??= new Microscope(m);
        const text = sampleOffsets(new Rng(77), t.corpus.val ?? t.corpus.train, m.B, m.cfg.context);
        const knockouts = await this.scope.knockouts(text);
        const patching = await this.scope.patching();
        const patterns = await this.scope.patterns();
        onProgress({ stage: 'heads-done', knockouts, patching, patterns });
        onProgress({ stage: 'collect' });
        this.sae ??= new GpuSae(m, saeDefaults(m.cfg.nLayer));
        const sae = this.sae;
        const t0 = performance.now();
        await sae.collect(t.corpus);
        const curve = [];
        const total = sae.opts.steps;
        onProgress({ stage: 'sae', at: null, step: 0, total });
        for (let s = 0; s < total; s++) {
          const measure = s % 100 === 99 || s === total - 1;
          sae.submitStep(measure);
          if (measure) {
            const at = await sae.readStep();
            curve.push(at);
            onProgress({ stage: 'sae', at, step: s + 1, total });
          }
          if (this.disposed) throw new Error('the engine was closed');
        }
        onProgress({ stage: 'analyse' });
        const result = await sae.analyse();
        const params = await sae.readParams();
        const seconds = (performance.now() - t0) / 1000;
        return { step, knockouts, patching, patterns, sae: { opts: sae.opts, curve, result, params, seconds } };
      } finally {
        this.dissecting = null;
        if (this.usable && this.want && this.trainer.step < this.trainer.totalSteps) this.startLoop();
      }
    };
    this.dissecting = run();
    return this.dissecting;
  }

  dispose(): void {
    this.disposed = true;
    this.want = false;
    this.running = false;
    this.gpu.device.removeEventListener('uncapturederror', this.onGpuError);
    const busy = Promise.all([this.profiling?.catch(() => undefined), this.dissecting?.catch(() => undefined)]);
    void busy.then(() => this.loopDone).finally(() => this.trainer.destroy());
  }
}
