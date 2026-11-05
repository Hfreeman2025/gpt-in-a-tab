import { toHalf } from '../model/half';
import { headName } from '../model/probe';
import { normaliseText } from '../model/tokenizer';
import type { Decode } from '../ui/decode';
import { $, h, int } from '../ui/dom';
import type { Scope } from '../ui/scope';
import { TOP_K } from '../workers/protocol';
import type { Snapshot } from './engine';
import { Cancelled, type HeadAttention, PRIORITY, type Sample, type SamplerPool, type Ticket } from './sampler';
import { type Checkpoint, MAX_CHECKPOINTS, SAMPLE_LENGTH, SAMPLE_SEED, type Session, promptFor } from './session';

interface ReplayDeps {
  readonly pool: SamplerPool;
  readonly decode: Decode;
  readonly scope: Scope;
  /** The session in view; work for any other session is dropped when it lands. */
  session(): Session | null;
  /** Whether the run is training, so the newest checkpoint means "follow it". */
  running(): boolean;
}

/**
 * The checkpoints of a run and the text drawn from each: which to draw first, which is on
 * screen (following the run, or replaying the one under the cursor), and inspecting a character
 * of it (where the model looked when it wrote it).
 */
export class Replay {
  /** The head whose attention lights the text (null: every head, averaged). */
  selectedHead: number | null = null;
  /** Step 0's text, drawn from the initial weights before the run's first checkpoint arrives. */
  earlySample: { session: number; sample: Sample } | null = null;
  private inspectSummary: string | null = null;
  private shownEpoch = -1;
  private settingsTimer = 0;

  constructor(private readonly d: ReplayDeps) {}

  private current(s: Session): boolean {
    return this.d.session() === s;
  }

  // ------------------------------------------------------------------ checkpoints and sampling

  addCheckpoint(s: Session, snap: Snapshot): void {
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

  /** Keep memory bounded on very long runs: drop the checkpoint closest to its neighbours. */
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
      const ticket = this.d.pool.sample(s.cfg, { key: cp.key, half: cp.half }, promptFor(s), SAMPLE_LENGTH, s.temperature, SAMPLE_SEED, priority);
      cp.ticket = ticket;
      cp.urgent = urgent;
      ticket.result.then(
        (sample) => {
          if (!this.current(s) || cp.ticket !== ticket) return;
          cp.ticket = null;
          if (epoch !== s.epoch) return;
          cp.sample = sample;
          cp.epoch = epoch;
          this.paintMarks(s);
          this.refresh(s);
        },
        (e: unknown) => {
          if (cp.ticket === ticket) cp.ticket = null;
          if (e instanceof Cancelled || !this.current(s)) return;
          $('text-note').textContent = `Could not draw the text for step ${int(cp.step)}: ${String(e)}`;
        },
      );
    }
  }

  paintMarks(s: Session): void {
    this.d.scope.setMarks(s.checkpoints.map((c) => ({ step: c.step, ready: c.sample !== null && c.epoch === s.epoch })));
  }

  /** The prompt and temperature from the controls; prompt characters the text never uses are left out. */
  readSettings(s: Session): void {
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
  settingsChanged(): void {
    window.clearTimeout(this.settingsTimer);
    this.settingsTimer = window.setTimeout(() => {
      const s = this.d.session();
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

  scrub(step: number): void {
    const s = this.d.session();
    if (!s || s.checkpoints.length === 0) return;
    let best = 0;
    for (let i = 1; i < s.checkpoints.length; i++) if (Math.abs(s.checkpoints[i].step - step) < Math.abs(s.checkpoints[best].step - step)) best = i;
    this.moveCursor(s, best);
  }

  stepCursor(delta: number): void {
    const s = this.d.session();
    if (!s || s.checkpoints.length === 0) return;
    const from = s.live ? s.checkpoints.length - 1 : s.cursor;
    this.moveCursor(s, Math.max(0, Math.min(s.checkpoints.length - 1, from + delta)));
  }

  private moveCursor(s: Session, index: number): void {
    const last = s.checkpoints.length - 1;
    if (index === last && this.d.running()) {
      this.goLive();
      return;
    }
    s.live = false;
    s.cursor = index;
    this.schedule(s);
    this.refresh(s);
  }

  goLive(): void {
    const s = this.d.session();
    if (!s) return;
    s.live = true;
    this.schedule(s);
    this.refresh(s);
  }

  /** Show the text for the cursor's checkpoint (or, following the run, the newest one drawn). */
  refresh(s: Session): void {
    if (!this.current(s)) return;
    const { decode, scope } = this.d;
    const cps = s.checkpoints;
    let index = -1;
    if (s.live) {
      // The newest checkpoint that has any text; after a change of settings its old text stays
      // up (marked as being re-drawn) rather than jumping back to an older checkpoint.
      for (let i = cps.length - 1; i >= 0 && index < 0; i--) if (cps[i].sample) index = i;
      // While a character is being inspected the text holds still (training carries on), so the
      // heads and the glyph under the pointer stay about the same text.
      const held = cps[s.shown];
      if (decode.inspected !== null && held?.sample && held.epoch === s.epoch) index = s.shown;
    } else index = cps[s.cursor]?.sample ? s.cursor : -1;
    const target = s.live ? cps.length - 1 : s.cursor;
    const cp = cps[index];
    const note = $('text-note');
    const behind = target >= 0 && index !== target ? cps[target] : null;
    if (this.inspectSummary && decode.inspected !== null) note.textContent = this.inspectSummary;
    else if (behind && !s.live) note.textContent = `Drawing the text for step ${int(behind.step)}…`;
    else if (cp && cp.epoch !== s.epoch) note.textContent = 'Re-drawing every checkpoint with these settings…';
    else
      note.textContent = s.dropped
        ? `${s.dropped} character${s.dropped === 1 ? '' : 's'} of the prompt never appear in the text, so the model cannot read them; left out.`
        : '';
    note.title = note.textContent ?? '';

    const where = $('where');
    const total = cps.length ? cps[cps.length - 1].step : 0;
    if (!cp) {
      where.textContent = s.live ? 'live · waiting for a checkpoint' : 'replay';
    } else {
      const holding = s.live && index !== cps.length - 1 && decode.inspected !== null;
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
    scope.setCursor(cp || cps[s.cursor] ? cursorStep : null, s.live);

    if (!cp?.sample || (index === s.shown && s.attention?.key === `${cp.key}|${s.epoch}`)) {
      if (!cp) decode.message(cps.length ? 'Drawing the first text…' : 'Waiting for the first checkpoint…');
      return;
    }
    if (index !== s.shown || decode.length === 0 || cp.epoch !== this.shownEpoch) {
      s.shown = index;
      this.shownEpoch = cp.epoch;
      decode.show({
        key: cp.key,
        settings: `${cp.epoch}`,
        prompt: cp.epoch === s.epoch ? s.promptChars : [],
        chars: Array.from(cp.sample.tokens, (t) => s.tok.chars[t] ?? ''),
        probs: cp.sample.probs,
      });
      // Keep the heads showing something live: the last character, unless one is being inspected.
      this.requestAttention(s, decode.inspected ?? Math.max(0, (decode.length || cp.sample.tokens.length) - 1), 120);
    }
  }

  // ------------------------------------------------------------------ inspecting a character

  inspect(index: number | null): void {
    const s = this.d.session();
    if (!s) return;
    const cp = s.checkpoints[s.shown];
    if (!cp?.sample) return;
    this.requestAttention(s, index ?? Math.max(0, (this.d.decode.length || cp.sample.tokens.length) - 1), index === null ? 200 : 40);
    // Letting go of the text lets the live view catch up with the run.
    if (index === null && s.live) queueMicrotask(() => this.refresh(s));
  }

  private requestAttention(s: Session, index: number, delay: number): void {
    const cp = s.checkpoints[s.shown];
    if (!cp?.sample) return;
    const sample = cp.sample;
    const key = `${cp.key}|${cp.epoch}`;
    if (s.attention && s.attention.key === key && s.attention.index === index) {
      this.showAttention(s, cp, index);
      return;
    }
    if (s.inspect) {
      window.clearTimeout(s.inspect.timer);
      s.inspect.ticket?.cancel();
    }
    const job: { index: number; ticket: Ticket<HeadAttention> | null; timer: number } = { index, ticket: null, timer: 0 };
    s.inspect = job;
    job.timer = window.setTimeout(() => {
      const prompt = cp.epoch === s.epoch ? promptFor(s) : [s.start];
      job.ticket = this.d.pool.attention(s.cfg, { key: cp.key, half: cp.half }, prompt, Array.from(sample.tokens), index);
      job.ticket.result.then(
        (value) => {
          if (!this.current(s) || s.inspect !== job) return;
          s.attention = { key, index, value };
          if (s.checkpoints[s.shown] === cp) this.showAttention(s, cp, index);
        },
        (e: unknown) => {
          if (!(e instanceof Cancelled) && this.current(s)) {
            this.inspectSummary = `Could not compute attention: ${String(e)}`;
            this.refresh(s);
          }
        },
      );
    }, delay);
  }

  /** The inspected character's readout (what was drawn, the runners-up), then light the text. */
  private showAttention(s: Session, cp: Checkpoint, index: number): void {
    const sample = cp.sample;
    if (!sample) return;
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
    this.refresh(s);
    this.lightAttention();
  }

  /** Light the output with the selected head's attention, or the mean of all of them. */
  lightAttention(): void {
    const s = this.d.session();
    const a = s?.attention;
    const { decode } = this.d;
    if (!s || !a || decode.inspected === null) {
      decode.unlight();
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
    const prompt = cp && cp.epoch === s.epoch ? promptFor(s) : [s.start];
    // The window the model read: the prompt (up to a context's worth), then its own text. The
    // default start token has no glyph on screen, so only a typed prompt has anything to light.
    decode.light(Math.min(prompt.length, s.cfg.context - 1), a.value.start, w);
  }
}
