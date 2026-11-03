import { type Dispatch, type Gpu, type Ref, grid, gridThreads } from './context';

/**
 * Builders for the GPU versions of the operations in src/cpu/ops.ts. Each returns the
 * dispatches to run, in order; nothing executes until a program is encoded and submitted.
 * Shapes and semantics mirror the CPU reference exactly, including which outputs accumulate.
 */

export interface MatmulSpec {
  readonly label: string;
  readonly a: Ref;
  readonly b: Ref;
  readonly c: Ref;
  readonly bias?: Ref;
  readonly M: number;
  readonly N: number;
  readonly K: number;
  readonly lda: number;
  readonly ldb: number;
  readonly ldc: number;
  readonly transA?: boolean;
  readonly transB?: boolean;
  readonly accum?: boolean;
  readonly alpha?: number;
  readonly aOff?: number;
  readonly bOff?: number;
  readonly cOff?: number;
  /** Two-level batch: z = z1·nb2 + z2, each operand offset by z1·s1 + z2·s2 floats. */
  readonly batch?: {
    readonly nb1: number;
    readonly nb2: number;
    readonly aS1: number;
    readonly aS2: number;
    readonly bS1: number;
    readonly bS2: number;
    readonly cS1: number;
    readonly cS2: number;
  };
}

export function matmul(g: Gpu, s: MatmulSpec, dummy: Ref): Dispatch {
  const bt = s.batch ?? { nb1: 1, nb2: 1, aS1: 0, aS2: 0, bS1: 0, bS2: 0, cS1: 0, cS2: 0 };
  const nb = bt.nb1 * bt.nb2;
  if (nb > 65535) throw new Error('too many matmul batches');
  return g.dispatch(
    s.label,
    'matmul',
    [s.M, s.N, s.K, bt.nb2, s.lda, s.ldb, s.ldc, { f: s.alpha ?? 1 }, s.aOff ?? 0, s.bOff ?? 0, s.cOff ?? 0, 0, bt.aS1, bt.aS2, bt.bS1, bt.bS2, bt.cS1, bt.cS2, 0, 0],
    [s.a, s.b, s.c, s.bias ?? dummy],
    [Math.ceil(s.N / 64), Math.ceil(s.M / 64), nb],
    {
      constants: { TRANS_A: s.transA ?? false, TRANS_B: s.transB ?? false, HAS_BIAS: s.bias !== undefined, ACCUM: s.accum ?? false },
      flops: 2 * s.M * s.N * s.K * nb,
    },
  );
}

export function reduce(g: Gpu, label: string, parts: Ref, out: Ref, len: number, S: number, stride: number, off = 0, accum = false): Dispatch {
  return g.dispatch(label, 'reduce', [len, S, stride, off], [parts, out], gridThreads(len), { constants: { ACCUM: accum } });
}

/**
 * How many ways to split a long reduction of length K so that `tiles` output tiles still fill
 * the GPU. Always divides K exactly and leaves at least 64 per split.
 */
export function splitFor(K: number, tiles: number, target = 512): number {
  let S = 1;
  while (S * 2 * tiles <= target && K % (S * 2) === 0 && K / (S * 2) >= 64) S *= 2;
  return S;
}

/** out[K×N] = aᵀ·b with a stored M×K: split over M, then reduced. Scratch needs S·K·N floats. */
export function matmulTNSplit(g: Gpu, label: string, a: Ref, b: Ref, out: Ref, scratch: Ref, M: number, K: number, N: number, dummy: Ref, accum = false): Dispatch[] {
  const S = splitFor(M, Math.ceil(K / 64) * Math.ceil(N / 64));
  const chunk = M / S;
  return [
    matmul(
      g,
      {
        label: `${label} (split ${S})`,
        a,
        b,
        c: scratch,
        M: K,
        N,
        K: chunk,
        lda: K,
        ldb: N,
        ldc: N,
        transA: true,
        batch: { nb1: 1, nb2: S, aS1: 0, aS2: chunk * K, bS1: 0, bS2: chunk * N, cS1: 0, cS2: K * N },
      },
      dummy,
    ),
    reduce(g, `${label} reduce`, scratch, out, K * N, S, K * N, 0, accum),
  ];
}

export function scratchForTN(M: number, K: number, N: number): number {
  return splitFor(M, Math.ceil(K / 64) * Math.ceil(N / 64)) * K * N;
}

/** db[N] = Σᵢ dy[i, :]: partial sums over row chunks, then reduced. */
export function colSum(g: Gpu, label: string, x: Ref, out: Ref, scratch: Ref, rows: number, cols: number): Dispatch[] {
  const S = colSplit(rows, cols);
  const chunk = Math.ceil(rows / S);
  return [g.dispatch(`${label} partial`, 'colsum', [rows, cols, S, chunk], [x, scratch], gridThreads(cols * S)), reduce(g, `${label} reduce`, scratch, out, cols, S, cols)];
}

function colSplit(rows: number, cols: number): number {
  let S = 1;
  while (S * cols < 65536 && rows / (S * 2) >= 16) S *= 2;
  return S;
}

export function scratchForColSum(rows: number, cols: number): number {
  return colSplit(rows, cols) * cols;
}

// ---------------------------------------------------------------------------------------------

export interface LinearRefs {
  readonly x: Ref;
  readonly W: Ref;
  readonly b: Ref | null;
  readonly y: Ref;
}

/** y[M×N] = x[M×K] · W[K×N] + b. */
export function linear(g: Gpu, label: string, r: LinearRefs, M: number, K: number, N: number, dummy: Ref): Dispatch {
  return matmul(g, { label, a: r.x, b: r.W, c: r.y, ...(r.b ? { bias: r.b } : {}), M, N, K, lda: K, ldb: N, ldc: N }, dummy);
}

export interface LinearGradRefs {
  readonly dy: Ref;
  readonly x: Ref;
  readonly W: Ref;
  readonly dx: Ref | null;
  readonly dW: Ref;
  readonly db: Ref | null;
  readonly scratch: Ref;
}

export function linearBackward(g: Gpu, label: string, r: LinearGradRefs, M: number, K: number, N: number, dummy: Ref): Dispatch[] {
  const out: Dispatch[] = [];
  if (r.dx) out.push(matmul(g, { label: `${label} dx`, a: r.dy, b: r.W, c: r.dx, M, N: K, K: N, lda: N, ldb: N, ldc: K, transB: true }, dummy));
  out.push(...matmulTNSplit(g, `${label} dW`, r.x, r.dy, r.dW, r.scratch, M, K, N, dummy));
  if (r.db) out.push(...colSum(g, `${label} db`, r.dy, r.db, r.scratch, M, N));
  return out;
}

// ---------------------------------------------------------------------------------------------

export interface LayerNormRefs {
  readonly x: Ref;
  readonly g: Ref;
  readonly b: Ref;
  readonly y: Ref;
  readonly mean: Ref;
  readonly rstd: Ref;
}

export function layerNorm(g: Gpu, label: string, r: LayerNormRefs, N: number, C: number): Dispatch {
  return g.dispatch(label, 'layernorm', [N, C, { f: 1e-5 }, 0], [r.x, r.g, r.b, r.y, r.mean, r.rstd], grid(N));
}

export interface LayerNormGradRefs {
  readonly dy: Ref;
  readonly x: Ref;
  readonly mean: Ref;
  readonly rstd: Ref;
  readonly g: Ref;
  readonly dx: Ref;
  readonly dg: Ref;
  readonly db: Ref;
  readonly scratch: Ref;
}

/** dx accumulates unless accum is false; dg and db are overwritten. */
export function layerNormBackward(g: Gpu, label: string, r: LayerNormGradRefs, N: number, C: number, accum = true): Dispatch[] {
  const S = lnSplit(N, C);
  const chunk = Math.ceil(N / S);
  return [
    g.dispatch(`${label} dx`, 'layernorm_bwd', [N, C, 0, 0], [r.dy, r.x, r.mean, r.rstd, r.g, r.dx], grid(N), { constants: { ACCUM: accum } }),
    g.dispatch(`${label} dg db partial`, 'layernorm_bwd_params', [N, C, S, chunk], [r.dy, r.x, r.mean, r.rstd, r.scratch], gridThreads(C * S)),
    reduce(g, `${label} dg`, r.scratch, r.dg, C, S, 2 * C, 0),
    reduce(g, `${label} db`, r.scratch, r.db, C, S, 2 * C, C),
  ];
}

function lnSplit(N: number, C: number): number {
  let S = 1;
  while (S * C < 32768 && N / (S * 2) >= 16) S *= 2;
  return S;
}

export function scratchForLayerNorm(N: number, C: number): number {
  return lnSplit(N, C) * 2 * C;
}

// ---------------------------------------------------------------------------------------------

export interface AttentionShape {
  readonly B: number;
  readonly T: number;
  readonly C: number;
  readonly H: number;
}

export interface AttentionRefs {
  readonly qkv: Ref;
  readonly scores: Ref;
  readonly probs: Ref;
  readonly y: Ref;
}

/** Scores = Q·Kᵀ/√hs per (batch, head), causal softmax, then probabilities · V. */
export function attention(g: Gpu, label: string, r: AttentionRefs, s: AttentionShape, dummy: Ref): Dispatch[] {
  const { B, T, C, H } = s;
  const hs = C / H;
  const C3 = 3 * C;
  const TT = T * T;
  return [
    matmul(
      g,
      {
        label: `${label} q·kᵀ`,
        a: r.qkv,
        b: r.qkv,
        c: r.scores,
        M: T,
        N: T,
        K: hs,
        lda: C3,
        ldb: C3,
        ldc: T,
        transB: true,
        alpha: 1 / Math.sqrt(hs),
        bOff: C,
        batch: { nb1: B, nb2: H, aS1: T * C3, aS2: hs, bS1: T * C3, bS2: hs, cS1: H * TT, cS2: TT },
      },
      dummy,
    ),
    softmax(g, `${label} softmax`, r.scores, r.probs, B * H * T, T),
    matmul(
      g,
      {
        label: `${label} p·v`,
        a: r.probs,
        b: r.qkv,
        c: r.y,
        M: T,
        N: hs,
        K: T,
        lda: T,
        ldb: C3,
        ldc: C,
        bOff: 2 * C,
        batch: { nb1: B, nb2: H, aS1: H * TT, aS2: TT, bS1: T * C3, bS2: hs, cS1: T * C, cS2: hs },
      },
      dummy,
    ),
  ];
}

export interface AttentionGradRefs {
  readonly dy: Ref;
  readonly qkv: Ref;
  readonly probs: Ref;
  readonly dScores: Ref;
  readonly dqkv: Ref;
}

export function attentionBackward(g: Gpu, label: string, r: AttentionGradRefs, s: AttentionShape, dummy: Ref): Dispatch[] {
  const { B, T, C, H } = s;
  const hs = C / H;
  const C3 = 3 * C;
  const TT = T * T;
  const alpha = 1 / Math.sqrt(hs);
  const qkvBatch = { aS1: T * C3, aS2: hs, bS1: T * C3, bS2: hs };
  return [
    // dP = dY · Vᵀ
    matmul(
      g,
      {
        label: `${label} dp`,
        a: r.dy,
        b: r.qkv,
        c: r.dScores,
        M: T,
        N: T,
        K: hs,
        lda: C,
        ldb: C3,
        ldc: T,
        transB: true,
        bOff: 2 * C,
        batch: { nb1: B, nb2: H, aS1: T * C, aS2: hs, bS1: T * C3, bS2: hs, cS1: H * TT, cS2: TT },
      },
      dummy,
    ),
    // dV = Pᵀ · dY
    matmul(
      g,
      {
        label: `${label} dv`,
        a: r.probs,
        b: r.dy,
        c: r.dqkv,
        M: T,
        N: hs,
        K: T,
        lda: T,
        ldb: C,
        ldc: C3,
        transA: true,
        cOff: 2 * C,
        batch: { nb1: B, nb2: H, aS1: H * TT, aS2: TT, bS1: T * C, bS2: hs, cS1: T * C3, cS2: hs },
      },
      dummy,
    ),
    // dS = softmax backward, in place
    g.dispatch(`${label} softmax bwd`, 'softmax_bwd', [B * H * T, T, 0, 0], [r.probs, r.dScores], grid(B * H * T)),
    // dQ = α · dS · K
    matmul(
      g,
      {
        label: `${label} dq`,
        a: r.dScores,
        b: r.qkv,
        c: r.dqkv,
        M: T,
        N: hs,
        K: T,
        lda: T,
        ldb: C3,
        ldc: C3,
        alpha,
        bOff: C,
        batch: { nb1: B, nb2: H, aS1: H * TT, aS2: TT, bS1: qkvBatch.bS1, bS2: hs, cS1: T * C3, cS2: hs },
      },
      dummy,
    ),
    // dK = α · dSᵀ · Q
    matmul(
      g,
      {
        label: `${label} dk`,
        a: r.dScores,
        b: r.qkv,
        c: r.dqkv,
        M: T,
        N: hs,
        K: T,
        lda: T,
        ldb: C3,
        ldc: C3,
        transA: true,
        alpha,
        cOff: C,
        batch: { nb1: B, nb2: H, aS1: H * TT, aS2: TT, bS1: qkvBatch.bS1, bS2: hs, cS1: T * C3, cS2: hs },
      },
      dummy,
    ),
  ];
}

// ---------------------------------------------------------------------------------------------

export function softmax(g: Gpu, label: string, x: Ref, y: Ref, rows: number, T: number): Dispatch {
  return g.dispatch(label, 'softmax', [rows, T, 0, 0], [x, y], grid(rows));
}

export function softmaxBackward(g: Gpu, label: string, y: Ref, d: Ref, rows: number, T: number): Dispatch {
  return g.dispatch(label, 'softmax_bwd', [rows, T, 0, 0], [y, d], grid(rows));
}

export function gelu(g: Gpu, label: string, x: Ref, y: Ref, n: number): Dispatch {
  return g.dispatch(label, 'gelu', [n, 0, 0, 0], [x, y], gridThreads(n));
}

/** d ← d ⊙ gelu'(x), in place. */
export function geluBackward(g: Gpu, label: string, x: Ref, d: Ref, n: number): Dispatch {
  return g.dispatch(label, 'gelu_bwd', [n, 0, 0, 0], [x, d], gridThreads(n));
}

/** y = a + dropout(b): the residual connection, with dropout on the branch. */
export function residual(g: Gpu, label: string, a: Ref, b: Ref, drop: Ref, y: Ref, n: number, site: number): Dispatch {
  return g.dispatch(label, 'residual', [n, site, 0, 0], [a, b, drop, y], gridThreads(n));
}

/** out = d ⊙ mask(site): dropout's backward (and forward) multiplication. */
export function dropoutMask(g: Gpu, label: string, d: Ref, drop: Ref, out: Ref, n: number, site: number): Dispatch {
  return g.dispatch(label, 'dropout', [n, site, 0, 0], [d, drop, out], gridThreads(n));
}

/** x = dropout(wte[token] + wpe[position]). */
export function embed(g: Gpu, label: string, tokens: Ref, wte: Ref, wpe: Ref, drop: Ref, x: Ref, N: number, T: number, C: number, site = 0): Dispatch {
  return g.dispatch(label, 'embed', [N, T, C, site], [tokens, wte, wpe, drop, x], gridThreads(N * C));
}

/** dwpe is overwritten; dwte is ACCUMULATED onto (the tied head writes it first). */
export function embedBackward(g: Gpu, label: string, tokens: Ref, dx: Ref, dwte: Ref, dwpe: Ref, scratch: Ref, B: number, T: number, C: number, V: number): Dispatch[] {
  const N = B * T;
  const S = wteSplit(N, V, C);
  const chunk = Math.ceil(N / S);
  return [
    g.dispatch(`${label} wpe`, 'embed_bwd_wpe', [B, T, C, 0], [dx, dwpe], gridThreads(T * C)),
    g.dispatch(`${label} wte partial`, 'embed_bwd_wte', [N, C, V, S, chunk, 0, 0, 0], [tokens, dx, scratch], gridThreads(V * C * S)),
    reduce(g, `${label} wte`, scratch, dwte, V * C, S, V * C, 0, true),
  ];
}

function wteSplit(N: number, V: number, C: number): number {
  let S = 1;
  while (S * V * C < 262144 && N / (S * 2) >= 32) S *= 2;
  return S;
}

export function scratchForEmbed(N: number, V: number, C: number): number {
  return wteSplit(N, V, C) * V * C;
}

export function crossEntropy(g: Gpu, label: string, logits: Ref, targets: Ref, dlogits: Ref, losses: Ref, N: number, V: number): Dispatch {
  return g.dispatch(label, 'cross_entropy', [N, V, { f: 1 / N }, 0], [logits, targets, dlogits, losses], grid(N));
}
