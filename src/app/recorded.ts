import type { ModelConfig, TrainConfig } from '../model/config';
import { fromHalf } from '../model/half';
import type { Moment, ProbeReading } from '../model/probe';
import type { Knockouts, Patching } from '../gpu/microscope';
import type { SaeFeature, SaeOptions, SaeStep } from '../gpu/sae';
import { type SampleId, sampleById } from '../samples';
import { type Tokenizer, buildTokenizer, normaliseText } from '../model/tokenizer';
import type { Reference } from '../ui/scope';

/**
 * A run of this page's own code, recorded in headless Chromium by scripts/record.mjs and shipped
 * with the page so the instrument has something real to show the moment it opens, and on devices
 * that cannot train the full model. Everything in it is labelled as recorded wherever it is shown.
 */
export interface RecordedRun {
  readonly version: 1;
  /** When and where it was recorded. */
  readonly recorded: string;
  readonly device: string;
  readonly text: SampleId;
  readonly cfg: ModelConfig;
  readonly train: TrainConfig;
  readonly totalSteps: number;
  readonly seconds: number;
  readonly trace: {
    readonly steps: readonly (readonly [number, number])[];
    readonly evals: readonly (readonly [number, number])[];
    readonly lr: readonly (readonly [number, number])[];
  };
  readonly probes: readonly ProbeReading[];
  readonly moment: Moment | null;
  readonly sample: string;
  readonly dissection: {
    readonly step: number;
    readonly knockouts: Knockouts;
    readonly patching: Patching;
    /** Attention on the induction test for a few heads, T×T, quantised to bytes, base64. */
    readonly patterns: Readonly<Record<string, string>>;
    readonly sae: {
      readonly opts: SaeOptions;
      readonly curve: readonly SaeStep[];
      readonly fvu: number;
      readonly l0: number;
      readonly dead: number;
      readonly scale: number;
      readonly positions: number;
      readonly seconds: number;
    };
  };
  /** The recorded model and autoencoder weights, half precision, in a separate file. */
  readonly weights: { readonly file: string; readonly bytes: number; readonly model: number; readonly sae: number };
}

export const RECORDED_BASE = 'recorded/';

export async function loadRecorded(signal?: AbortSignal): Promise<RecordedRun> {
  const r = await fetch(`${RECORDED_BASE}run.json`, { signal: signal ?? null });
  if (!r.ok) throw new Error(`the recorded run did not load (HTTP ${r.status})`);
  return (await r.json()) as RecordedRun;
}

export async function loadRecordedFeatures(signal?: AbortSignal): Promise<SaeFeature[]> {
  const r = await fetch(`${RECORDED_BASE}features.json`, { signal: signal ?? null });
  if (!r.ok) throw new Error(`the recorded features did not load (HTTP ${r.status})`);
  const raw = (await r.json()) as { id: number; density: number; mean: number; top: [number, number][] }[];
  return raw.map((f) => ({ id: f.id, density: f.density, mean: f.mean, top: f.top.map(([pos, value]) => ({ pos, value })) }));
}

/** The recording, ready to use: its text's tokenizer, and the run as the scope's reference waveform. */
export interface OpenedRecording {
  readonly run: RecordedRun;
  readonly tok: Tokenizer;
  readonly tokens: Uint32Array;
  readonly reference: Reference;
}

/** Loads the recorded run and the text it was trained on. Throws if either is unavailable. */
export async function openRecording(): Promise<OpenedRecording> {
  const run = await loadRecorded();
  const sample = sampleById(run.text);
  if (!sample) throw new Error(`the recording names an unknown text: ${run.text}`);
  const text = normaliseText(await sample.load());
  const tok = buildTokenizer(text);
  const reference: Reference = {
    train: smoothTrace(run.trace.steps),
    copy: run.probes.map((p) => ({ step: p.step, value: p.copySecond })),
    probes: run.probes,
    moment: run.moment,
    total: run.totalSteps,
    heads: { L: run.cfg.nLayer, H: run.cfg.nHead },
  };
  return { run, tok, tokens: tok.encode(text), reference };
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

export function decodePattern(b64: string): Float32Array {
  const bin = atob(b64);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255;
  return out;
}

export function encodePattern(p: ArrayLike<number>): string {
  let s = '';
  for (let i = 0; i < p.length; i++) s += String.fromCharCode(Math.max(0, Math.min(255, Math.round(p[i] * 255))));
  return btoa(s);
}

export interface RecordedWeights {
  readonly model: Float32Array;
  readonly sae: Float32Array;
}

export type DownloadState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'loading'; readonly received: number; readonly total: number; readonly cached: boolean }
  | { readonly kind: 'ready'; readonly weights: RecordedWeights; readonly fromCache: boolean }
  | { readonly kind: 'failed'; readonly message: string }
  | { readonly kind: 'cancelled' };

const CACHE = 'gpt-in-a-tab-recorded-v3';

/**
 * Fetches the recorded weights: progress as it goes, from the Cache API when it has them,
 * cancellable, and stored in the cache once complete. The page stays useful without them.
 */
export class WeightsDownload {
  state: DownloadState = { kind: 'idle' };
  private abort: AbortController | null = null;

  constructor(
    private readonly run: RecordedRun,
    private readonly onChange: (s: DownloadState) => void,
  ) {}

  private set(s: DownloadState): void {
    this.state = s;
    this.onChange(s);
  }

  get url(): string {
    return `${RECORDED_BASE}${this.run.weights.file}`;
  }

  /** Whether the weights are already in this browser's cache (so no download is needed). */
  async cached(): Promise<boolean> {
    try {
      const c = await caches.open(CACHE);
      return (await c.match(this.url)) !== undefined;
    } catch {
      return false;
    }
  }

  async start(): Promise<void> {
    if (this.state.kind === 'loading' || this.state.kind === 'ready') return;
    const abort = new AbortController();
    this.abort = abort;
    const total = this.run.weights.bytes;
    try {
      let cache: Cache | null = null;
      try {
        cache = await caches.open(CACHE);
      } catch {
        cache = null;
      }
      const hit = cache ? await cache.match(this.url) : undefined;
      this.set({ kind: 'loading', received: 0, total, cached: hit !== undefined });
      const response = hit ?? (await fetch(this.url, { signal: abort.signal }));
      if (!response.ok) throw new Error(`the server answered HTTP ${response.status}`);
      const reader = response.body?.getReader();
      const parts: Uint8Array[] = [];
      let received = 0;
      if (reader) {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (abort.signal.aborted) throw new DOMException('cancelled', 'AbortError');
          parts.push(value);
          received += value.length;
          this.set({ kind: 'loading', received, total, cached: hit !== undefined });
        }
      } else {
        const b = new Uint8Array(await response.arrayBuffer());
        parts.push(b);
        received = b.length;
      }
      if (received !== total) throw new Error(`expected ${total.toLocaleString('en-GB')} bytes, received ${received.toLocaleString('en-GB')}`);
      const bytes = new Uint8Array(received);
      let o = 0;
      for (const p of parts) {
        bytes.set(p, o);
        o += p.length;
      }
      if (!hit && cache) {
        try {
          await cache.put(this.url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }));
        } catch {
          // A full or unavailable cache only costs a download next time.
        }
      }
      const halves = new Uint16Array(bytes.buffer, 0, received / 2);
      const { model, sae } = this.run.weights;
      this.set({
        kind: 'ready',
        weights: { model: fromHalf(halves.subarray(0, model)), sae: fromHalf(halves.subarray(model, model + sae)) },
        fromCache: hit !== undefined,
      });
    } catch (e) {
      if (abort.signal.aborted || (e instanceof DOMException && e.name === 'AbortError')) this.set({ kind: 'cancelled' });
      else this.set({ kind: 'failed', message: e instanceof Error ? e.message : String(e) });
    } finally {
      this.abort = null;
    }
  }

  cancel(): void {
    this.abort?.abort();
  }
}
