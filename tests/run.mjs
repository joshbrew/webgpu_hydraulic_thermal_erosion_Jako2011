import http from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import './pacing.mjs';
import { renderGallery } from '../tools/gallery-template.mjs';
import { checkGallery } from './gallery.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const library = await readFile(resolve(root, 'WebGPUTerrainErosion.js'), 'utf8');
for (const [constant, file] of [['COMPUTE_WGSL_SOURCE', 'erosionCompute.wgsl'], ['RENDER_WGSL_SOURCE', 'erosionRender.wgsl'], ['COMPOSITE_WGSL_SOURCE', 'erosionComposite.wgsl']]) {
  const marker = `const ${constant} = /* wgsl */ \``;
  const start = library.indexOf(marker) + marker.length;
  const end = library.indexOf('`;', start);
  const shader = await readFile(resolve(root, file), 'utf8');
  if (library.slice(start, end).replace(/\r\n/g, '\n').trim() !== shader.replace(/\r\n/g, '\n').trim()) throw new Error(`${file} differs from the embedded shader; run npm run build.`);
}
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.wgsl': 'text/plain', '.css': 'text/css' };
const server = http.createServer(async (req, res) => {
  try {
    if (req.url === '/tests/baseline.js') {
      res.writeHead(200, { 'Content-Type': 'text/javascript' });
      res.end(execFileSync('git', ['show', 'HEAD:WebGPUTerrainErosion.js'], { cwd: root }));
      return;
    }
    const path = resolve(root, '.' + decodeURIComponent(req.url.split('?')[0] === '/' ? '/index.html' : req.url.split('?')[0]));
    if (!path.startsWith(root + '\\') && !path.startsWith(root + '/')) throw new Error('Outside root');
    const body = await readFile(path);
    res.writeHead(200, { 'Content-Type': mime[extname(path)] || 'application/octet-stream' }); res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/tests/physics.html`);
  await page.waitForFunction(() => window.testResults, undefined, { timeout: 120000 });
  const result = await page.evaluate(() => window.testResults);
  console.log(JSON.stringify(result, null, 2));
  if (!result.passed) process.exitCode = 1;
  if (result.passed) {
    const pacing = await page.evaluate(async () => (await import('/tests/worker.js')).checkWorkerPacing());
    console.log('Real GPU worker pacing and scene resets:', JSON.stringify(pacing));
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const run = page.getByRole('button', { name: 'Run', exact: true });
    await page.locator('label').filter({ hasText: /^Size/ }).locator('select').selectOption('256');
    for (const name of ['alpine', 'fan', 'badlands', 'canyon', 'talus', 'basin', 'river_delta', 'mountain_canyon', 'volcano', 'lowland_river', 'crater_lake']) {
      await page.locator('label').filter({ hasText: /^Example/ }).locator('select').selectOption(name);
      await page.getByRole('button', { name: 'Load example', exact: true }).click();
      await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent === 'Load example' && !b.disabled)
        && [...document.querySelectorAll('button')].some(b => b.textContent === 'Run' && !b.disabled), undefined, { timeout: 60000 });
      await page.getByRole('button', { name: 'Step', exact: true }).click();
      await page.waitForTimeout(500);
    }
    await page.getByRole('button', { name: 'Apply preset', exact: true }).click();
    if (!(await page.getByRole('checkbox', { name: 'Closed boundaries', exact: true }).isChecked())) throw new Error('Crater preset must retain closed boundaries.');
    await run.click(); await page.waitForTimeout(600);
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect mass', exact: true }).click();
    await page.waitForFunction(() => document.body.textContent.includes('Closed boundaries retain solid material.'), undefined, { timeout: 30000 });
    await run.click(); await page.waitForTimeout(600);
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    const rain = page.getByRole('checkbox', { name: 'Enable precipitation', exact: true });
    const springs = page.getByRole('checkbox', { name: 'Enable springs', exact: true });
    const duration = page.locator('label').filter({ hasText: /^Rain \/ spring duration s/ }).locator('input');
    await rain.uncheck(); await springs.check();
    if (await duration.isDisabled()) throw new Error('Spring-only duration control must stay available.');
    await duration.fill('0.071');
    await page.getByRole('button', { name: 'Reset rain timer', exact: true }).click();
    await page.getByRole('button', { name: 'Step', exact: true }).click();
    await page.waitForTimeout(500);
    await springs.uncheck();
    if (!(await duration.isDisabled())) throw new Error('Duration control must disable with all sources off.');
    await rain.check(); await duration.fill('6');
    const automatic = page.getByRole('checkbox', { name: 'Auto batch for speed', exact: true });
    await automatic.check();
    await run.click(); await page.waitForTimeout(1200);
    await page.waitForFunction(() => document.body.textContent.includes('batch mode: auto'), undefined, { timeout: 10000 });
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await automatic.uncheck();
    if (errors.length) throw new Error(errors.join('\n'));
    console.log('Built app: eleven layered examples, single steps, run/pause/restart, mass inspection and spring-only timing passed.');
    await page.screenshot({ path: resolve(root, 'tests/preview.png'), fullPage: true });
  }
  if (process.argv.includes('--examples') && result.passed) {
    const output = resolve(root, 'examples');
    await mkdir(output, { recursive: true });
    await page.goto(`http://127.0.0.1:${server.address().port}/tests/examples.html`);
    await page.waitForFunction(() => window.examplesReady, undefined, { timeout: 60000 });
    const cases = await page.evaluate(async () => (await import('/tests/examples.js')).cases);
    const gallery = [];
    for (const [terrain, preset, seconds, size = 256] of cases) {
      const id = terrain + '-' + preset;
      await page.evaluate(([terrain, preset, size]) => window.loadExample(terrain, preset, size), [terrain, preset, size]);
      await page.locator('canvas').screenshot({ path: resolve(output, id + '-before.png') });
      const report = await page.evaluate(seconds => window.advanceExample(seconds), seconds);
      await page.locator('canvas').screenshot({ path: resolve(output, id + '-after.png') });
      gallery.push({ terrain, preset, id, ...report });
      console.log('Long-run example:', id, JSON.stringify(report));
    }
    await writeFile(resolve(output, 'report.json'), JSON.stringify(gallery, null, 2) + '\n');
    await writeFile(resolve(output, 'index.html'), renderGallery(gallery));
  }
  if (process.argv.includes('--benchmark') && result.passed) {
    const report = await page.evaluate(async () => (await import('/tests/benchmark.js')).benchmark());
    console.log(JSON.stringify({ benchmark: report }, null, 2));
  }
  if (result.passed) {
    const count = JSON.parse(await readFile(resolve(root, 'examples/report.json'), 'utf8')).length;
    await checkGallery(page, `http://127.0.0.1:${server.address().port}/examples/index.html`, count);
    await checkGallery(page, pathToFileURL(resolve(root, 'examples/index.html')).href, count, { full: false });
  }
} finally { await browser.close(); server.close(); }
