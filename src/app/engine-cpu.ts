import type { ModelConfig, TrainConfig } from '../model/config';
import type { Corpus } from '../model/data';
import { paramLayout } from '../model/layout';
import { planSteps, textTokensPerStep } from '../model/schedule';
import type { TrainerRequest, TrainerResponse } from '../workers/protocol';
import { Clock, Emitter, type Engine, type EngineEvent, type EngineInfo, type RunState, type Snapshot } from './engine';

const CALIBRATE_FROM = 5;
const CALIBRATE_TO = 25;

/** The fallback engine: the CPU reference implementation, training in a worker. */
export class CpuEngine implements Engine {
  readonly info: EngineInfo;
  state: RunState = 'paused';
  private readonly worker: Worker;
  private readonly emitter = new Emitter();
  private readonly clock = new Clock();
  private readonly waiting = new Map<number, { resolve: (s: Snapshot) => void; reject: (e: Error) => void }>();
  private nextReq = 1;
  private totalSteps: number;
  private completed = 0;
  private plannedBy: 'data' | 'time' = 'data';
  private calibration: { elapsed: number; completed: number } | null = null;
  private calibrated = false;
  private tokensPerSecond = 0;
  private started = false;

  constructor(
    cfg: ModelConfig,
    train: TrainConfig,
    private readonly corpus: Corpus,
    private readonly seed: number,
    totalSteps: number,
  ) {
    this.totalSteps = totalSteps;
    const layout = paramLayout(cfg);
    const floats = layout.total;
    const N = train.batch * cfg.context;
    const C = cfg.dModel;
    const actFloats = cfg.nLayer * (N * C * 15 + train.batch * cfg.nHead * cfg.context * cfg.context) + N * cfg.vocabSize * 2;
    this.info = {
      kind: 'cpu',
      device: `CPU · JavaScript in a worker (${navigator.hardwareConcurrency || '?'} cores available, 1 used)`,
      cfg,
      train,
      paramCount: layout.count,
      memory: [
        { category: 'parameters', bytes: floats * 4 },
        { category: 'gradients', bytes: floats * 4 },
        { category: 'optimiser state', bytes: floats * 8 },
        { category: 'activations', bytes: actFloats * 4 },
        { category: 'corpus', bytes: corpus.tokens.byteLength },
      ],
      dispatchesPerStep: 0,
      flopsPerStep: 6 * layout.count * N,
    };
    this.worker = new Worker(new URL('../workers/trainer.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<TrainerResponse>) => this.receive(e.data);
    this.worker.onerror = (e) => this.fail(`The training worker failed: ${e.message}`);
    this.worker.onmessageerror = () => this.fail('The training worker sent a message that could not be read.');
  }

  on(l: (e: EngineEvent) => void): () => void {
    return this.emitter.on(l);
  }

  private send(m: TrainerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(m, transfer);
  }

  private setState(s: RunState): void {
    this.state = s;
    this.emitter.emit({ type: 'state', state: s });
  }

  private fail(message: string): void {
    if (this.state === 'failed') return;
    this.rejectWaiting(new Error(message));
    this.clock.stop();
    this.setState('failed');
    this.emitter.emit({ type: 'error', message });
  }

  private receive(m: TrainerResponse): void {
    switch (m.t) {
      case 'stats': {
        const bad = m.steps.find((x) => !Number.isFinite(x.loss));
        if (bad) {
          this.send({ t: 'pause' });
          this.fail(`Training diverged: the loss stopped being a number at step ${bad.step}.`);
          return;
        }
        this.completed = m.completed;
        const sps = this.clock.mark(m.completed);
        if (sps > 0) this.tokensPerSecond = sps * this.info.train.batch * this.info.cfg.context;
        this.calibrate();
        const tps = this.info.train.batch * this.info.cfg.context;
        this.emitter.emit({
          type: 'stats',
          steps: m.steps,
          evals: m.evals,
          probes: m.probes,
          progress: {
            completed: m.completed,
            totalSteps: this.totalSteps,
            elapsed: this.clock.elapsed,
            tokensPerSecond: this.tokensPerSecond,
            tokensSeen: m.completed * tps,
            epochs: (m.completed * textTokensPerStep(this.info.train, this.info.cfg.context)) / this.corpus.train.length,
            plannedBy: this.plannedBy,
          },
        });
        break;
      }
      case 'snapshot': {
        const snap = { step: m.step, params: m.params };
        if (m.req === -1) this.emitter.emit({ type: 'milestone', snapshot: snap });
        else {
          this.waiting.get(m.req)?.resolve(snap);
          this.waiting.delete(m.req);
        }
        break;
      }
      case 'finished':
        this.clock.stop();
        this.setState('finished');
        break;
      case 'error':
        this.fail(m.message);
        break;
    }
  }

  /** Speed from training time (pauses excluded) between two observed step counts. */
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
    if (byTime < this.totalSteps) {
      this.totalSteps = byTime;
      this.plannedBy = 'time';
      this.send({ t: 'plan', totalSteps: byTime });
    }
    this.calibrated = true;
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    const tokens = Uint32Array.from(this.corpus.tokens);
    this.send({ t: 'start', cfg: this.info.cfg, train: this.info.train, tokens, totalSteps: this.totalSteps, seed: this.seed }, [tokens.buffer]);
    this.clock.start();
    this.setState('training');
  }

  pause(): void {
    if (this.state !== 'training') return;
    this.send({ t: 'pause' });
    this.clock.stop();
    this.setState('paused');
  }

  resume(): void {
    if (this.state !== 'paused') return;
    this.send({ t: 'resume' });
    this.clock.start();
    this.setState('training');
  }

  extend(steps: number): void {
    this.totalSteps = Math.max(this.totalSteps, this.completed) + steps;
    this.send({ t: 'extend', totalSteps: this.totalSteps });
    this.clock.start();
    this.setState('training');
  }

  snapshot(): Promise<Snapshot> {
    if (this.state === 'failed') return Promise.reject(new Error('the training worker has stopped'));
    const req = this.nextReq++;
    return new Promise((resolve, reject) => {
      this.waiting.set(req, { resolve, reject });
      this.send({ t: 'snapshot', req });
    });
  }

  private rejectWaiting(e: Error): void {
    for (const w of this.waiting.values()) w.reject(e);
    this.waiting.clear();
  }

  async profile(): Promise<null> {
    return null;
  }

  probePatterns(): Promise<null> {
    return Promise.resolve(null);
  }

  /** The CPU fallback is too slow to dissect; the page shows the recorded GPU run instead. */
  dissect(): Promise<null> {
    return Promise.resolve(null);
  }

  dispose(): void {
    this.worker.terminate();
    this.rejectWaiting(new Error('the run was closed'));
  }
}
