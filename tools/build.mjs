import { pathToFileURL } from 'node:url';
import config from '../tinybuild.config.js';
import { readFile, writeFile } from 'node:fs/promises';
import { renderGallery } from './gallery-template.mjs';
const module = process.env.TINYBUILD_MODULE ? pathToFileURL(process.env.TINYBUILD_MODULE).href : 'tinybuild';
const { bundle } = await import(module);
await bundle(config.bundler);
await bundle({ ...config.bundler, entryPoints: ['examples/gallery.js'], outfile: 'dist/examples' });
await bundle({ ...config.bundler, entryPoints: ['examples/erode-mesh.js'], outfile: 'dist/erode-mesh' });
const report = await readFile(new URL('../examples/report.json', import.meta.url), 'utf8').catch(error => {
  if (error.code !== 'ENOENT') throw error;
});
if (report) await writeFile(new URL('../examples/index.html', import.meta.url), renderGallery(JSON.parse(report)));
