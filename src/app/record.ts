import type { EngineInfo, Dissection } from './engine';
import { type RecordedRun, encodePattern } from './recorded';
import { toHalf } from '../model/half';
import type { Moment, ProbeReading } from '../model/probe';
import type { SampleId } from '../samples';

/** What the controller hands over from a finished, dissected run. */
export interface RecordInput {
  readonly text: SampleId;
  readonly info: EngineInfo;
  readonly totalSteps: number;
  readonly seconds: number;
  readonly trace: { readonly steps: readonly [number, number][]; readonly lr: readonly [number, number][]; readonly evals: readonly [number, number][] };
  readonly probes: readonly ProbeReading[];
  readonly moment: Moment | null;
  readonly sample: string;
  /** The final weights, half precision. */
  readonly weights: Uint16Array;
}

declare global {
  interface Window {
    __record?: { run: RecordedRun; features: unknown[]; weights: string };
  }
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;
const r4 = (x: number) => Math.round(x * 10000) / 10000;

/** Every n-th point, plus the last: the trace at a resolution the screen can show. */
function thin<T>(xs: readonly T[], max: number): T[] {
  if (xs.length <= max) return [...xs];
  const step = xs.length / max;
  const out: T[] = [];
  for (let i = 0; i < max; i++) out.push(xs[Math.floor(i * step)]);
  out.push(xs[xs.length - 1]);
  return out;
}

/**
 * The recording scripts/record.mjs writes to public/recorded/: run.json (trace, induction test,
 * moment, causal results, autoencoder summary), features.json (every feature's top contexts) and
 * weights.bin (model and autoencoder, half precision).
 */
export function record(input: RecordInput, d: Dissection): void {
  const { cfg, train, device } = input.info;
  const keep = [input.moment?.head, input.moment?.partner].filter((k): k is number => k !== undefined && k !== null);
  const patterns: Record<string, string> = {};
  for (const k of keep) patterns[String(k)] = encodePattern(d.patterns[k]);
  const sae = toHalf(d.sae.params);
  const all = new Uint16Array(input.weights.length + sae.length);
  all.set(input.weights, 0);
  all.set(sae, input.weights.length);
  const bytes = new Uint8Array(all.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  const r = d.sae.result;
  const run: RecordedRun = {
    version: 1,
    recorded: new Date().toISOString(),
    device,
    text: input.text,
    cfg,
    train,
    totalSteps: input.totalSteps,
    seconds: r3(input.seconds),
    trace: {
      steps: thin(input.trace.steps, 600).map(([s, v]) => [s, r4(v)] as const),
      evals: input.trace.evals.map(([s, v]) => [s, r4(v)] as const),
      lr: thin(input.trace.lr, 200).map(([s, v]) => [s, v] as const),
    },
    probes: input.probes.map((p) => ({
      step: p.step,
      copyFirst: r4(p.copyFirst),
      copySecond: r4(p.copySecond),
      prefix: p.prefix.map(r4),
      prev: p.prev.map(r4),
    })),
    moment: input.moment,
    sample: input.sample,
    dissection: {
      step: d.step,
      knockouts: { baseText: r4(d.knockouts.baseText), baseCopy: r4(d.knockouts.baseCopy), text: d.knockouts.text.map(r4), copy: d.knockouts.copy.map(r4) },
      patching: { clean: r4(d.patching.clean), corrupt: r4(d.patching.corrupt), patched: d.patching.patched.map(r4), recovered: d.patching.recovered.map(r4) },
      patterns,
      sae: {
        opts: d.sae.opts,
        curve: d.sae.curve.map((c) => ({ step: c.step, mse: r4(c.mse), l0: r3(c.l0) })),
        fvu: r4(r.fvu),
        l0: r3(r.l0),
        dead: r4(r.dead),
        scale: r.scale,
        positions: r.positions,
        seconds: r3(d.sae.seconds),
      },
    },
    weights: { file: 'weights.bin', bytes: bytes.length, model: input.weights.length, sae: sae.length },
  };
  const features = r.features.map((f) => ({ id: f.id, density: r4(f.density * 100) / 100, mean: r3(f.mean), top: f.top.map((t) => [t.pos, r3(t.value)]) }));
  window.__record = { run, features, weights: btoa(bin) };
}
