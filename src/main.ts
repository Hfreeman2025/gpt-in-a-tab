import { Controller } from './app/controller';
import { acquireDevice } from './gpu/context';
import { SAMPLES } from './samples';

async function boot(): Promise<void> {
  const q = new URLSearchParams(location.search);
  const forceCpu = q.has('cpu');
  const acq = forceCpu ? { error: 'CPU mode was asked for in the address.' } : await acquireDevice();
  const gpu = 'error' in acq ? null : acq;
  const app = new Controller({ gpu, gpuError: 'error' in acq ? acq.error : null, forceCpu });
  const wanted = q.get('text');
  await app.choose(SAMPLES.find((s) => s.id === wanted)?.id ?? 'sonnets');
  // The first screen is the run itself. ?hold keeps it armed until RUN is pressed.
  if (!q.has('hold')) app.autostart();
  (window as unknown as { __app: Controller }).__app = app;
}

void boot();
