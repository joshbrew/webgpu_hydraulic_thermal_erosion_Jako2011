import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const root = resolve('.'), server = http.createServer(async (req, res) => {
  try {
    if (req.url === '/') { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<!doctype html><title>River performance</title>'); return; }
    const p = resolve(root, '.' + req.url); if (!p.startsWith(root + '\\') && !p.startsWith(root + '/')) throw Error('path');
    res.writeHead(200, { 'Content-Type': extname(p) === '.html' ? 'text/html' : 'text/javascript' }); res.end(await readFile(p));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async compare => {
    const Baseline = compare ? (await import('/tests/river-performance-baseline.js')).default : null;
    return (await import('/tests/river-performance.js')).benchmark(Baseline);
  }, process.argv.includes('--compare-before'));
  console.log(JSON.stringify(result, null, 2));
} finally { await browser.close(); server.close(); }
