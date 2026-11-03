// Counter-based dropout, identical to src/model/dropout.ts: element i at a site survives when
// hash(i ^ siteSeed) ≥ p·2³², so backward recomputes the forward mask instead of storing it.

struct Dropout { seed: u32, threshold: u32, scale: f32, _pad: u32 };

fn hash32(x: u32) -> u32 {
  var v = x;
  v ^= v >> 16u;
  v *= 0x7feb352du;
  v ^= v >> 15u;
  v *= 0x846ca68bu;
  v ^= v >> 16u;
  return v;
}

fn site_seed(seed: u32, site: u32) -> u32 {
  return hash32(seed ^ hash32(site + 1u));
}

fn drop_mask(d: Dropout, s: u32, i: u32) -> f32 {
  if (d.threshold == 0u) { return 1.0; }
  if (hash32(i ^ s) >= d.threshold) { return d.scale; }
  return 0.0;
}

