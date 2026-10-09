import { readFile, writeFile } from 'node:fs/promises';
const path = new URL('../WebGPUTerrainErosion.js', import.meta.url);
let library = await readFile(path, 'utf8');
for (const [constant, file] of [
  ['COMPUTE_WGSL_SOURCE', 'erosionCompute.wgsl'],
  ['RENDER_WGSL_SOURCE', 'erosionRender.wgsl'],
  ['COMPOSITE_WGSL_SOURCE', 'erosionComposite.wgsl'],
]) {
  const marker = `const ${constant} = /* wgsl */ \``;
  const start = library.indexOf(marker) + marker.length;
  if (start < marker.length) throw new Error(`Missing embedded shader: ${constant}`);
  const end = library.indexOf('`;', start);
  const source = '\n' + (await readFile(new URL('../' + file, import.meta.url), 'utf8')).replace(/\r\n/g, '\n').trim() + '\n';
  library = library.slice(0, start) + source + library.slice(end);
}
await writeFile(path, library);
