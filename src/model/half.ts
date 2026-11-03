/**
 * IEEE 754 half precision, by hand: checkpoints are kept as 16-bit floats so that dozens of
 * snapshots of the weights fit in memory. Round to nearest, ties to even; overflow saturates to
 * infinity, tiny values become subnormals or zero.
 */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalfBits(x: number): number {
  f32[0] = x;
  const b = u32[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf or NaN
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow
  if (e <= 0) {
    if (e < -10) return sign; // underflow to zero
    mant |= 0x800000; // implicit leading one
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    const rest = mant & ((1 << shift) - 1);
    let m = mant >>> shift;
    if (rest > half || (rest === half && m & 1)) m++;
    return sign | m;
  }
  const rest = mant & 0x1fff;
  mant >>>= 13;
  if (rest > 0x1000 || (rest === 0x1000 && mant & 1)) {
    mant++;
    if (mant === 0x400) {
      mant = 0;
      e++;
      if (e >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (e << 10) | mant;
}

export function fromHalfBits(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : sign * Infinity;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

export function toHalf(x: Float32Array): Uint16Array {
  const out = new Uint16Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = toHalfBits(x[i]);
  return out;
}

const table = new Float32Array(65536);
for (let h = 0; h < 65536; h++) table[h] = fromHalfBits(h);

export function fromHalf(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = table[h[i]];
  return out;
}
