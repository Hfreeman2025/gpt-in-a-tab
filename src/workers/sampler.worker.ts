/// <reference lib="webworker" />
/**
 * Draws text from checkpoints with the CPU reference model and a key/value cache, and replays the
 * window it wrote with to show where each head looked. Jobs are self-contained (they carry their
 * weights), run one at a time, and check for cancellation between chunks.
 */
import { attentionWhenWriting, generate } from '../model/generate';
import { fromHalf } from '../model/half';
import { type SamplerRequest, type SamplerResponse, TOP_K, type WeightsRef } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

type Job = Exclude<SamplerRequest, { t: 'cancel' }>;
const queue: Job[] = [];
const cancelled = new Set<number>();
let busy = false;

/** The last few checkpoints unpacked to full precision, by key. */
const unpacked = new Map<string, Float32Array>();
function weights(w: WeightsRef): Float32Array {
  let p = unpacked.get(w.key);
  if (!p) {
    p = fromHalf(w.half);
    unpacked.set(w.key, p);
    while (unpacked.size > 3) unpacked.delete(unpacked.keys().next().value as string);
  }
  return p;
}

const post = (m: SamplerResponse, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const breathe = () => new Promise<void>((r) => setTimeout(r, 0));

self.onmessage = (e: MessageEvent<SamplerRequest>) => {
  const m = e.data;
  if (m.t === 'cancel') {
    cancelled.add(m.job);
    return;
  }
  queue.push(m);
  void pump();
};

async function pump(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    while (queue.length) {
      const job = queue.shift() as Job;
      if (cancelled.delete(job.job)) continue;
      try {
        if (job.t === 'sample') await sample(job);
        else attention(job);
      } catch (err) {
        post({ t: 'error', job: job.job, message: String(err) });
      }
      await breathe();
    }
  } finally {
    busy = false;
  }
}

async function sample(job: Extract<Job, { t: 'sample' }>): Promise<void> {
  const t0 = performance.now();
  const it = generate(job.cfg, weights(job.weights), job.prompt, job.length, job.temperature, job.seed, TOP_K, job.edit);
  const tokens = new Uint32Array(job.length);
  const probs = new Float32Array(job.length);
  const tops = new Float32Array(job.length * TOP_K * 2).fill(-1);
  let n = 0;
  for (const s of it) {
    tokens[n] = s.token;
    probs[n] = s.p;
    for (let k = 0; k < s.top.length; k++) {
      tops[(n * TOP_K + k) * 2] = s.top[k][0];
      tops[(n * TOP_K + k) * 2 + 1] = s.top[k][1];
    }
    n++;
    if (n % 48 === 0) {
      await breathe();
      if (cancelled.delete(job.job)) return;
    }
  }
  post({ t: 'sample', job: job.job, tokens, probs, tops, ms: performance.now() - t0 }, [tokens.buffer, probs.buffer, tops.buffer]);
}

function attention(job: Extract<Job, { t: 'attention' }>): void {
  const a = attentionWhenWriting(job.cfg, weights(job.weights), job.prompt, job.sample, job.index);
  const w = a.weights;
  post({ t: 'attention', job: job.job, start: a.start, length: a.length, promptOffset: a.promptOffset, weights: w }, [w.buffer]);
}
