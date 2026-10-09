import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const root = resolve('.');
const server = http.createServer(async (req, res) => {
  try { const p = resolve(root, '.' + req.url); if (!p.startsWith(root + '\\') && !p.startsWith(root + '/')) throw Error('path');
    res.writeHead(200, { 'Content-Type': extname(p) === '.html' ? 'text/html' : 'text/javascript' }); res.end(await readFile(p));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}/tests/meanders.html`);
  await page.waitForFunction(() => window.riverReady);
  const variants = JSON.parse(process.env.RIVER_VARIANTS || '[{}]');
  for (let v = 0; v < variants.length; v++) {
    const initial = await page.evaluate(options => window.setupRiver(options), variants[v]);
    console.log(JSON.stringify({ variant: v, initial: { ...initial, path: undefined }, options: variants[v] }));
    await page.locator('canvas').screenshot({ path: resolve(root, `tests/river-${v}-before.png`) });
    const stages = [];
    for (const seconds of [180, 180, 360]) {
      const result = await page.evaluate(seconds => window.evolveRiver(seconds), seconds);
      console.log(JSON.stringify({ variant: v, ...result, path: undefined, sinceLast: { ...result.sinceLast, path: undefined } }));
      await page.locator('canvas').screenshot({ path: resolve(root, `tests/river-${v}-${Math.round(result.time)}.png`) });
      stages.push(result);
    }
    assert(initial.sinuosity < 1.02, 'river starts nearly straight');
    assert(stages.some(s => s.sinuosity > 1.12), 'bends develop over time');
    assert(stages.some(s => s.carved > 100 && s.abandoned > 50), 'new channels carve and abandoned initial reaches fill');
    assert(stages[1].sinceLast.carved > 20 && stages[1].sinceLast.abandoned > 20, 'channel migration continues between evolved states');
    assert(stages.every(s => s.outletDischarge > 0.005), 'the supplied river keeps flowing through the downstream outlet');
    await page.evaluate(() => window.riverView(9));
    await page.locator('canvas').screenshot({ path: resolve(root, `tests/river-${v}-flow.png`) });
  }
} finally { await browser.close(); server.close(); }
