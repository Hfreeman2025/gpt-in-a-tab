import type { GenerationEdit } from '../model/generate';
import type { ModelConfig } from '../model/config';
import type { SamplerRequest, SamplerResponse, WeightsRef } from '../workers/protocol';

export interface Sample {
  readonly tokens: Uint32Array;
  readonly probs: Float32Array;
  /** Top alternatives per token, [token, p] pairs, TOP_K per token, -1 where absent. */
  readonly tops: Float32Array;
}

export interface HeadAttention {
  /** Index into [trimmed prompt, sample] of the first visible character. */
  readonly start: number;
  readonly length: number;
  readonly promptOffset: number;
  /** [layer · heads + head][length] */
  readonly weights: Float32Array;
}

export interface Ticket<T> {
  readonly result: Promise<T>;
  /** A worker has begun it (it can no longer be reprioritised, only cancelled). */
  readonly started: boolean;
  /** Move a job that is still waiting up or down the queue. */
  reprioritize(priority: number): void;
  cancel(): void;
}

type Payload = SamplerRequest extends infer R ? (R extends { t: 'cancel' } ? never : Omit<R, 'job'>) : never;

interface Job {
  readonly id: number;
  readonly payload: Payload;
  priority: number;
  resolve(v: unknown): void;
  reject(e: Error): void;
  worker: Lane | null;
}

interface Lane {
  readonly worker: Worker;
  job: Job | null;
  readonly interactive: boolean;
}

export class Cancelled extends Error {
  constructor() {
    super('cancelled');
  }
}

/**
 * A small pool of sampling workers. Samples (one for every checkpoint) spread over the pool, the
 * one under the cursor first; one lane is kept for attention alone, so pointing at a character
 * never waits behind the backlog.
 */
export class SamplerPool {
  private readonly lanes: Lane[];
  private readonly queue: Job[] = [];
  private nextId = 1;
  private broken: Error | null = null;

  constructor(size: number) {
    this.lanes = Array.from({ length: Math.max(2, size) }, (_, i) => {
      const worker = new Worker(new URL('../workers/sampler.worker.ts', import.meta.url), { type: 'module', name: `sampler ${i}` });
      const lane: Lane = { worker, job: null, interactive: i === 0 };
      worker.onmessage = (e: MessageEvent<SamplerResponse>) => this.receive(lane, e.data);
      worker.onerror = (e) => this.fail(new Error(`a sampling worker failed: ${e.message}`));
      return lane;
    });
  }

  get size(): number {
    return this.lanes.length;
  }

  private fail(e: Error): void {
    this.broken = e;
    for (const lane of this.lanes) {
      lane.job?.reject(e);
      lane.job = null;
    }
    for (const j of this.queue.splice(0)) j.reject(e);
  }

  private receive(lane: Lane, m: SamplerResponse): void {
    const job = lane.job;
    lane.job = null;
    if (job && job.id === m.job) {
      if (m.t === 'error') job.reject(new Error(m.message));
      else if (m.t === 'sample') job.resolve({ tokens: m.tokens, probs: m.probs, tops: m.tops } satisfies Sample);
      else job.resolve({ start: m.start, length: m.length, promptOffset: m.promptOffset, weights: m.weights } satisfies HeadAttention);
    }
    this.pump();
  }

  private submit<T>(payload: Payload, priority: number): Ticket<T> {
    const id = this.nextId++;
    let job!: Job;
    const result = new Promise<T>((resolve, reject) => {
      job = { id, payload, priority, resolve: resolve as (v: unknown) => void, reject, worker: null };
    });
    if (this.broken) job.reject(this.broken);
    else {
      this.queue.push(job);
      this.queue.sort((a, b) => b.priority - a.priority);
      this.pump();
    }
    return {
      result,
      get started() {
        return job.worker !== null;
      },
      reprioritize: (priority: number) => {
        if (job.worker !== null || job.priority === priority) return;
        job.priority = priority;
        this.queue.sort((a, b) => b.priority - a.priority);
        this.pump();
      },
      cancel: () => {
        const q = this.queue.indexOf(job);
        if (q >= 0) this.queue.splice(q, 1);
        else if (job.worker?.job === job) {
          job.worker.worker.postMessage({ t: 'cancel', job: id } satisfies SamplerRequest);
          job.worker.job = null; // the worker skips the rest; its late reply is ignored
          this.pump();
        }
        job.reject(new Cancelled());
      },
    };
  }

  private pump(): void {
    for (const lane of this.lanes) {
      if (lane.job || this.queue.length === 0) continue;
      // The attention lane takes only attention; the others take samples, best first.
      const i = lane.interactive ? this.queue.findIndex((j) => j.payload.t === 'attention') : this.queue.findIndex((j) => j.payload.t === 'sample');
      if (i < 0) continue;
      const job = this.queue.splice(i, 1)[0];
      job.worker = lane;
      lane.job = job;
      const msg = { ...job.payload, job: job.id } as SamplerRequest;
      lane.worker.postMessage(msg);
    }
  }

  sample(
    cfg: ModelConfig,
    weights: WeightsRef,
    prompt: readonly number[],
    length: number,
    temperature: number,
    seed: number,
    priority: number,
    edit?: GenerationEdit,
  ): Ticket<Sample> {
    return this.submit<Sample>({ t: 'sample', cfg, weights, prompt, length, temperature, seed, ...(edit ? { edit } : {}) }, priority);
  }

  attention(cfg: ModelConfig, weights: WeightsRef, prompt: readonly number[], sample: readonly number[], index: number): Ticket<HeadAttention> {
    return this.submit<HeadAttention>({ t: 'attention', cfg, weights, prompt, sample, index }, PRIORITY.interactive);
  }

  dispose(): void {
    for (const lane of this.lanes) lane.worker.terminate();
    this.fail(new Error('the sampler was closed'));
  }
}

export const PRIORITY = { background: 1, recent: 2, interactive: 10 } as const;
