import { EROSION_PRESETS } from '../erosionPresets.js';
import { TERRAIN_EXAMPLES } from '../terrainExamples.js';
const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export function renderGallery(gallery) {
  const cards = gallery.map(g => {
    const title = EROSION_PRESETS[g.preset].label, terrain = TERRAIN_EXAMPLES[g.terrain].label;
    return `<article data-terrain="${escape(g.terrain)}" data-preset="${escape(g.preset)}" data-seconds="${g.seconds}" data-size="${g.size}" data-seed="${g.seed}">
<h2>${escape(title)} · ${escape(terrain)}</h2><p>Seed ${g.seed} · ${g.size} × ${g.size} · ${g.seconds} simulated seconds</p>
<button type="button" class="run-sim">Run sim</button>
<a class="open-studio" href="../index.html?map=${escape(g.terrain)}&amp;preset=${escape(g.preset)}&amp;size=${g.size}">Open in studio ↗</a>
<div class="pair"><figure><img src="${escape(g.id)}-before.png" alt="${escape(title)} before erosion" loading="lazy"><figcaption>Starting terrain</figcaption></figure><figure><img src="${escape(g.id)}-after.png" alt="${escape(title)} after erosion" loading="lazy"><figcaption>After ${g.seconds} seconds</figcaption></figure></div>
<p>${g.closed ? 'Closed basin solid drift: ' + (g.solidChangeFraction * 100).toFixed(5) + '%' : 'Open boundaries allow sediment export.'}</p></article>`;
  }).join('\n');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Erosion preset examples</title>
<style>
.demo-nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:28px}.demo-nav a,.open-studio{color:#92d4d2;border:1px solid #344f5b;border-radius:8px;padding:9px 13px;text-decoration:none;font-size:13px}.demo-nav a:hover,.demo-nav [aria-current]{background:#20383f}.open-studio{margin-left:10px;display:inline-block}
body{background:#10151c;color:#e9edf3;font:16px/1.6 system-ui;max-width:1400px;margin:auto;padding:32px}h1,h2{line-height:1.2}article{margin:48px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;display:block;border-radius:10px}figcaption,p{color:#abb9ca}button,select{font:inherit;color:#e9edf3;background:#263b4a;border:1px solid #55728a;border-radius:7px;padding:8px 14px;cursor:pointer}button:hover{background:#365267}button:disabled,select:disabled{opacity:.5;cursor:wait}.run-sim{margin:0 0 16px}.live-panel{margin-top:20px;padding:16px;border:1px solid #55728a;border-radius:12px;background:#141e28}.live-panel[hidden]{display:none}.live-panel canvas{display:block;width:100%;max-width:1000px;aspect-ratio:4/3;height:auto;margin:auto;border-radius:8px;cursor:grab}.live-panel canvas:active{cursor:grabbing}.live-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:14px}.live-status{color:#b4d7c7;margin-left:auto}.layer-legend{display:flex;flex-wrap:wrap;gap:12px}.layer-legend span{font-size:14px;border-left:12px solid;padding-left:8px;color:#c3cdd8}@media(max-width:700px){body{padding:16px}.pair{grid-template-columns:1fr}.live-status{width:100%;margin:0}.live-panel{padding:10px}}
</style><script src="../dist/examples.js" defer></script></head><body>
<nav class="demo-nav" aria-label="Demo navigation"><a href="../index.html">Erosion studio</a><a href="index.html" aria-current="page">Visual examples</a><a href="erode-mesh.html">Final mesh demo</a><a href="../../hydraulic_triangular_irregular_networks/index.html">Triangle mesh studio</a></nav>
<h1>Erosion preset examples</h1><p>Seeded terrain with distinct material layers, shown in slope relief. Run sim continues until paused. Drag to orbit, scroll to zoom, adjust river sediment load, or use View result to reach the pictured time. Only the active preview runs.</p>
<p><a href="erode-mesh.html" style="color:#92d4d2">Erode → final adaptive mesh</a> · Pause an evolved landscape and compare its Delaunay surface with the full grid.</p>
${cards}</body></html>`;
}
