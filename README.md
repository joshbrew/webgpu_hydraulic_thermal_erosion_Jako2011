# WebGPU real-time Hydraulic + Thermal erosion
<img width="800" alt="Screenshot 2026-04-09 145612" src="https://github.com/user-attachments/assets/2f41f99e-cb94-48fc-8844-99460129a27f" />

## Try it: [https://webgpuerosion.netlify.app/](https://webgpuerosion.netlify.app/)

Browser WebGPU implementation of "[Fast Hydraulic and Thermal Erosion on the GPU](https://old.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf)" by Balazs Jako (2011) with some modern tweaks for WebGPU and more control knobs.

Import images (heightmaps, DEM, anything) and run erosion simulations. Upload up to 4 bands for different sediment/bedrock layer topologies with different erosion rates. Works on up to ~2048x2048 images in browser in the current version (we have a laptop RTX 4070). We tried higher res texture arrays but it looked bad using a chunking technique to overcome buffer size limits (2GB in browser) on our last attempt. We might revisit it. I'm sure there's more performance optimization to do, and the SWE render is jank.

You can also paint terrain and spring sources onto your base map image. Drag to
orbit, scroll to zoom, or choose **Reset view**. **Fly camera** in Camera and
navigation enables pointer look and WASD movement; Esc returns to orbit.

### Build and run

Open this folder in the terminal.

`npm i -g tinybuild` then `tinybuild`

Or install the local development dependencies with `npm install`, then use
`npm start`. `npm run build` refreshes the embedded shaders and rebuilds the
checked-in browser bundle.

Chromium-based browsers recommended for the latest webgpu support.

## Built-in terrain examples and profiles

The studio starts with a paused alpine map. Selecting an **Example map preset**
loads it immediately with its recommended erosion settings. Change resolution or
**Terrain seed**, then click **Load example** to rebuild that map.
Generation uses GPU simplex noise, six-octave FBM and domain warping inspired by
the noiseCompute shaders. Each example loads a matching erosion profile. Click
**Run** or **Step** to begin; the simulation starts paused. **Reset DEM** restores
the same generated terrain and seed. Selecting an erosion **Preset** applies it
immediately, changing the process and base material while preserving terrain and
elapsed steps, and restarts the rain timer. **Apply preset** reapplies the selected
settings after manual adjustments. The header and image cards link to the visual
example gallery and mesh demos; gallery cards can also open their map and profile
in the studio. Advanced settings use collapsible panels.

`npm run test:studio-ui` checks automatic preset application, actual rendered
orbit/zoom/reset, preserved simulation time, image links and mobile layout over
HTTP and `file://`.

| Example | Starting profile | What to watch |
| --- | --- | --- |
| Alpine catchment | Balanced drainage | Connected channels under steady rain; compare at 20–60 simulated seconds |
| Mountain outlet and fan | Alluvial fan | Sediment spreading onto the plain; rain ends after 25 simulated seconds |
| Warped badlands | Soft badlands | Rills and bank collapse in loose terrain; compare at 30–90 seconds |
| Layered canyon | Layered canyon | Upper sediment, resistant shelves, weaker lower beds and bedrock |
| Cliffs and talus | Dry talus relaxation | Dry cliffs relax into aprons; compare at 10–30 seconds |
| Closed basin | Flash flood and dry-down | Six-second storm, internal redistribution and settling without edge export |
| River delta | River delta | Sediment arriving with spring water builds bars on a shallow coastal shelf; compare at 180–360 seconds |
| Mountain river canyon | Mountain river canyon | A wet river cuts a descending canyon with a thin soil layer over resistant rock; compare at 60–120 seconds |
| Longonot-inspired volcano | Volcanic radial canyons | Broad crater, radial gullies through ash and tephra above tuff/lava; a 35-second storm followed by dry-down, compare at 90–150 seconds |
| Volcanic crater lake | Crater lake and flank erosion | A twenty-second storm cuts the flanks while the crater retains its lake; compare at 60–90 seconds |
| Lowland river and floodplain | Lowland meandering river | A nearly straight river develops bends, cuts new channels and fills abandoned reaches; compare at 180–720 seconds or run continuously |

Every generated example now has three or four actual material surfaces. Their
thicknesses vary with terrain noise, position or depositional setting. Material
rows show names, erodability and individual thermal switches. Changing an erosion
profile updates the surface material while retaining the distinct deeper units.
The badlands alternate weak beds with resistant ledges; fan and delta sediments
overlie older alluvium; alpine regolith overlies weathered rock and bedrock.

The three river scenes and crater-lake scene start with water. The rivers also seed
a painted spring at their headwaters; resetting restores both water and springs.
The delta has young sediment above older alluvium and resistant rock. Its coastal water is
an initial condition, with a retained edge depth rather than a replenished ocean
or tides. The mountain canyon has a 0.045-unit soil cover over weathered rock.
Both volcano scenes use closed boundaries to check their inventories.
The Longonot-inspired scene starts dry with a wide, irregular crater and shallow
radial drainage seeds. Storm runoff then cuts through thin ash and thicker tephra
to expose firmer tuff, while the lava core supports the cone. It is a procedural
interpretation of the reference landscape, not a surveyed reconstruction.
For the API, an optional `raster.water` Float32Array supplies initial water depth
per cell; imported heightmaps otherwise start dry.

Additional profiles provide spring-fed channels and gentle weathering. The
flash-flood profile also works well on the mountain outlet. These are artistic
profiles for normalized heightmaps, rather than calibrated geological models.
The lowland profile adds optional empirical bank retreat: wet outgoing flow and
changes in flow direction preferentially remove outer banks. Removed material
enters the adjacent river as sediment, respecting material interfaces and masks.
This is a simplified height-field approach, not a calibrated meander model such
as [Zhao et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL093516).
**River bank retreat** controls its rate; **Outer bend bias** controls curvature
sensitivity. **Inner bend sediment drift** moves suspended material toward the
inside of bends using conservative transfers, helping bars deflect the current.
The lowland example starts nearly straight, with small seeded bank irregularities.
Its bend guide is recalculated from the evolving discharge; no curve is animated
or imposed as a function of time. This optional guide assumes a river running
roughly from the top toward the bottom of the map. Side and headwater boundaries
are walls, with an open downstream outlet, and inflow is matched to channel capacity.
`openBoundaryEdges` is a bit mask: left 1, right 2, top 4, bottom 8; default 15.
Other profiles leave bank retreat and transverse drift disabled.
Tuning is based on 100 meters/pixel; changing map scale or resolution changes the
slopes and source footprint, so imported DEMs may need adjustment.

Browse the [before-and-after preset gallery](examples/index.html) for seeded
examples after 20–360 simulated seconds. The gallery includes every profile and
a closed-basin conservation check; [run measurements](examples/report.json)
record terrain changes and remaining water and sediment.
Each result has **Run sim**, opening its own seeded setup in a reusable live
viewport. **Pause**, **Reset**, and **View result** control playback; View result
advances to the exact step count behind that card's after image. Drag to orbit
and scroll to zoom. Switch between natural, slope relief, river flow, erodability and history
views. Slope relief is the default in the app and gallery. **Run** continues until
paused or closed, including beyond the pictured time. Finite storms still stop
their rainfall at the configured time while simulation and settling continue.
**River sediment load** adjusts the solid supplied per unit of spring water;
zero supplies clear water. The delta starts at 0.12 and the lowland river at 0.1.
**River flow** shows arrows from the current computed velocity. Resting water
has no arrows; the overlay makes active channels easier to distinguish from scars.
The gallery uses a standalone classic bundle so these controls work when
`examples/index.html` is opened directly from disk, as well as over HTTP.
Volcano gallery images use 512 × 512; the other cards use 256 × 256.

## Conservation and stability

- Sediment is a solid thickness per cell. Its transport follows the same outgoing
  water fractions as the water solver, so internal face transfers conserve it.
  Open map boundaries carry sediment away with runoff; masked cells are walls.
- Springs can supply sediment along with water using `sourceSedimentConcentration`.
  Rain supplies water only. Solid input uses the exact active source duration,
  including partial steps, and is never reduced by evaporation. With sediment-fed
  springs, closed-basin solid inventories grow by the external input; open scenes
  also lose solid through their boundaries.
- Erosion exchanges terrain and suspended solid without creating water.
  Entrainment requires outgoing water and a wet cell. Still water cannot keep
  cutting a basin floor. A closed basin can still erode internally while water
  moves, but its total terrain plus sediment is conserved.
  Cuts are limited by the downstream terrain drop to avoid digging isolated pits.
  Hydraulic cuts and talus transfer also stop at each material interface so a
  large step cannot bypass a resistant layer before its rates are evaluated.
  Coincident layer surfaces expose the later, deeper layer; bottom-layer
  erodability varies gradually toward the zero-height base rather than jumping
  from its minimum to maximum at the first cut.
- Evaporation removes water only. Suspended solid settles back onto terrain when
  a cell dries. Below `1e-6` water-height units, the remaining sediment settles;
  the thin water film is retained and continues to evaporate normally.
- Flow damping reduces inherited pipe momentum exponentially. **Max drainage /
  step** limits the outgoing fraction of available water (default 0.5), keeping
  thin channels from emptying and refilling abruptly. Lower values slow drainage;
  1 permits the full available volume to leave during a step.
- Deposition and talus transfer preserve solids. Upper height/sediment clipping
  has been removed; diagonal talus transfers stay inside the grid. Thermal
  erosion can continue on dry slopes independently of hydraulic erosion.
- The legacy API names `hardnessBase`, `hardnessVariation`, and layer
  `hardnessMin/Max` represent **erodability**: 0 is resistant, 1 is loose. The UI
  uses the correct terminology. Loose-soil re-erosion uses recent deposition
  history as an empirical softening proxy.
- Talus threshold is `cellSize * (bias + scale * (1 - erodability))`, with
  diagonal distance accounted for. Evaporation and exchange use exponential
  rates, and changing the timestep preserves elapsed simulation time.
- Rain, springs and evaporation are integrated analytically for half a step on
  each side of transport. Finite storms include their final partial step and
  the evaporation of that partial input, so timestep changes preserve input
  and the remaining water. Pulse 2 extends the same event at its current
  intensity. Gentle slope capacity uses the horizontal normal components to
  avoid rounding small but nonzero grades to zero.
- “Cut / depth limit” controls both maximum cut per step and the water-depth
  scale used by sediment capacity. It is not a total lifetime erosion limit.

**Inspect mass** pauses the run and reads total terrain, suspended sediment,
their combined solid inventory, and water from the GPU. Compare successive
inspections in a closed basin to check conservation; open boundaries intentionally
export solid material. Hierarchical GPU reductions return just 64 bytes per
inspection. Totals use float32 reductions, so small roundoff differences are
expected. The inspection records its simulation time; ordinary status updates
reuse that snapshot without triggering another reduction or readback.

## Real-time performance

The simulation uses three compute dispatches without thermal erosion and four
with thermal erosion or bank retreat, versus six previously. Bank transfers share
the thermal preparation pass. The lowland meander guide adds two small dispatches
per step when enabled: parallel discharge reductions and cached row bends.
Bank and sediment transfers reuse those bends, avoiding repeated neighborhood
reads for each receiver. Both passes and their intermediate values stay on the GPU.
Fixed springs add one preparation dispatch only
when source settings or the grid change. Their cached rates reuse the source
buffer; painted springs are restored when switching layouts. Fixed spring
locations use the same full 32-bit integer seed and hash on the CPU and GPU, so
markers match their source cells even for large seeds.
The worker overlaps up to two submitted frames, avoiding an unlimited GPU queue
and the old per-second synchronous benchmark stalls.
Enable **Auto batch for speed** to adjust the number of steps per frame from
GPU completion feedback, targeting an eight-millisecond batch within 1–64 steps.
It uses the existing completion signal, adds no timing queries, and preserves
the chosen timestep. Leave it off for a fixed simulation speed; faster hardware
and lighter scenes advance more simulated time in auto mode. The status line
shows the current batch size.

**Mesh detail: auto** caps the preview mesh at roughly 512 cells along its longest
axis. A 1024-square DEM uses approximately a quarter of the previous mesh cells;
2048 uses approximately a sixteenth. Simulation and exports retain full
resolution. Choose **full resolution** for fine channels or narrow mask holes.
Terrain and water height scales affect only the preview. GPU pass timing fields
are populated by explicit single-step benchmarks. Hardware supporting
`timestamp-query` measures time inside the compute and render passes; otherwise
the timing mode is marked `completion-time`, including submission and waiting.
Short timestamps may be rounded to zero by the browser. The real-time loop
reports CPU submission timing without querying or waiting for individual passes.

Water reflection normals come from the displayed mesh, with subtle surface
ripples. Water geometry reads the actual terrain-plus-water surface, retaining
flat resting lakes and dry shore cells while avoiding neighborhood reads in
every vertex. Reflection normals avoid those reads in every water pixel. The
natural view uses muted soil, rock and wet alluvium colors; low terrain is not
automatically colored as ocean, and diagnostic history bands stay in their
dedicated view.
Slope coloration distinguishes gentle surfaces, inclined soil and steep rock
walls. Subtle curvature shading improves valley/ridge separation, and natural
colors follow the currently exposed material surface. **Slope relief** provides
a stronger slope palette with the same lighting. The water pass uses surface
derivatives and skips terrain-normal neighborhood sampling in its vertices.
Water opacity increases with actual depth, keeping shallow rivers and coastal
water visible while leaving thin films and dry shores transparent.

## Regression checks

`npm test` runs real WebGPU tests in installed Chrome, then exercises all eleven
examples, stepping and run/pause/restart in the built app. Run `npm run build`
first after editing source files. `BROWSER_CHANNEL=msedge` selects Edge instead.
Tests cover still water, evaporation, dry settling, deposition, closed-basin and
masked-grid conservation, boundary export, talus edges, resistant materials,
timestep changes, large solid inventories, active material changes, and seeded
generation, thin-film water retention, hierarchical mass reduction, GPU
benchmarks, partial storms, cached source invalidation, large seeds, planar
lakes, rotation consistency and closed inventories at three resolutions.
Tests also cover sediment-fed springs through evaporating partial storms,
conservative bank transfers, still-water banks, and shifting lowland channels.
`npm run test:meanders` checks the 256-square river at 180, 360 and 720 seconds:
a nearly straight start, increasing channel sinuosity, new cuts, filled abandoned
reaches, continuing migration between snapshots and sustained outlet discharge.
The delta check requires upstream incision and downstream deposition; the
mountain canyon checks river activity and resistant rock. Volcano checks cover a
resting lake, storm water balance, deepening radial gullies, exposure of deeper
layers, retained crater relief and conserved solids. A four-layer exposure check
verifies each unit's erodability and relief rendering. Worker tests
cover adaptive pacing, queue bounds, pause/restart and exact reset of the five
scenes' water, terrain and springs.
Gallery checks exercise continuous playback beyond the pictured duration,
default slope relief, editable sediment load, pause, exact result replay,
reset, switching examples and reusing one viewport. Direct-file playback is
also checked in Chrome.
`node tests/run.mjs --benchmark` additionally compares compute and
render completion times with the current Git HEAD and reports current GPU
timestamp timings separately. Completion times include submission and waiting;
they should not be compared directly with timestamp measurements.
`npm run benchmark:river` measures the lowland river at 256, 512 and 1024 square
grids with GPU timestamps and completion timings. It warms each scene and reports
the median of five 100-step batches. The recorded local comparison in
`tests/river-performance-report.json` measured 17–56% lower GPU step time after
parallel row analysis and bend caching. This measures simulation work, not frame
rate; results depend on the GPU, browser and active terrain.
`node tests/run.mjs --examples` regenerates the gallery and checks every profile
over its documented duration, including finite states and closed-basin solids.

## Results

### Optional final adaptive mesh

The [original demo](index.html) and each live viewport in the
[preset gallery](examples/index.html) now include the same **Final hydraulic
mesh** controls. Select a map preset, run erosion, pause, then **Build hydraulic
mesh**. Preview/Balanced/High presets and custom minimum/maximum spacing control
fidelity. The triangle limit is enforced by coarsening and rebuilding from the
cached snapshot, never by dropping triangles. The display reports sampled RMS
and maximum height errors; these samples are not a guaranteed global error bound.

In the original demo, the existing erosion worker builds Delaunay topology while
paused. Gallery builds run in a separate topology worker. Both use one cached
snapshot and upload the finished mesh once. Surface, wireframe, opacity, camera
and color changes reuse the topology. Resuming erosion discards the final mesh.

**Run mesh water** solves the shallow-water equations on fixed triangle control
volumes, leaving the raster erosion result unchanged. It uses shared conservative
Rusanov edge fluxes and [hydrostatic reconstruction](https://epubs.siam.org/doi/10.1137/S1064827503431090),
GPU CFL timestep selection, wet/dry handling, semi-implicit Manning friction,
and analytic rain/evaporation balance. Triangle geometry and edge adjacency are
built once; every water step and its rendering stay on the GPU, with four compute
dispatches per step and adaptive frame batching. There is no CPU readback in a
water step. **Inspect mesh water** is an explicit paused readback for water volume,
depth, invalid states and the actual selected timestep.

Bed and water depth use the same triangle quadrature when initialized from the
paused raster on the GPU, preserving the level surface of fully wet lakes.
Momentum starts from rest, then water continues with mesh rain and the captured
springs. Mesh rain is a constant independent water input; it
does not replay the erosion storm schedule. **Reset mesh water** returns to the
paused erosion water. This is a first-order fixed-bed solver: coarse triangles
smooth fine channels and it does not transport sediment, change terrain or remesh.
Smaller spacing helps narrow channels but may require shorter stable timesteps.

`npm run test:mesh-water` checks curved raster lake projection, stepped lakes and wet/dry shores, closed dam-break
mass, positive dry-bed flooding and analytic weather balance.
`npm run test:mesh-pages` exercises both integrated pages over HTTP and `file://`,
including pause-only builds, hard triangle limits, wireframes, water run/pause,
inspection/reset and returning to grid erosion.

`npm run benchmark:mesh-water` compares the same water solver on adaptive and
full-grid triangles. On the tested NVIDIA adapter, the 512 lowland example used
6,566 adaptive triangles at 0.0170 ms per GPU compute step, versus 522,242 full-grid
triangles at 1.061 ms. These are compute-only medians, not frame rates or an
equal-accuracy comparison; the meshes retain different spatial detail.

Open [Erode → final mesh](examples/erode-mesh.html), also linked from the preset
gallery and main erosion controls. Run continuously or advance to an example
time, then choose **Build final mesh**. This pauses erosion, takes one terrain
snapshot and runs the existing hydraulic TIN CPU Delaunay builder in a background
worker. The vendored MIT sources keep this demo independent of a sibling checkout.
The [ready-made river preview](examples/erode-mesh.html?scene=lowland_river&seconds=180&mesh=1)
advances a 512 river and builds its mesh automatically, then remains paused for comparison.

The resulting triangles render in WebGPU with the same slope relief, layer colors,
water and diagnostic shading as the grid. Toggle the wireframe or switch between
the adaptive surface and full grid. Spacing controls trade detail for triangle
count; this is a visual approximation, not an error-bounded simplification.
River guides follow terrain drainage inferred by the TIN builder, rather than
replacing the erosion solver's water fluxes. Smaller spacing retains narrow gullies.

Rebuilding mesh settings reuses the paused snapshot. **Run** discards that final
mesh and returns to continuous grid erosion. No mesh building, state readbacks or
topology changes occur in a simulation step. This integration therefore adds no
meshing cost to continuous erosion. Mesh display samples paused GPU heights and
materials; it never writes simulation state. Existing erosion behavior is unchanged.

`npm run test:mesh` checks bit-identical simulation state/time before and after
building, cache reuse, fresh snapshots after resuming, actual grid/triangle and
wireframe rendering, fixed-mesh water on five landscapes, an evolved 512 river, responsive controls
and direct `file://` loading.

The results are pretty good on default settings, which you can completely override in the demo UI:

Video 1: https://www.youtube.com/watch?v=qz4CPlcCj7E

Video 2: https://www.youtube.com/watch?v=o19LsY70Aco

Test image made in our [webgpu noise generator](https://github.com/joshbrew/webgpu_noise_compute_textures):

<img width="300" height="300" alt="noise_test" src="https://github.com/user-attachments/assets/94550ae7-bb3a-4f3c-9366-2e524d928909" />
<img width="300" height="300" alt="Screenshot 2026-04-08 172111" src="https://github.com/user-attachments/assets/b39114c2-5b55-462a-9c5c-51b71cb38adb" />
<img width="300" height="300" alt="Screenshot 2026-04-08 172106" src="https://github.com/user-attachments/assets/c19658c0-d506-437f-a500-8d833090eb1b" />

Another result on the same map:

<img width="300" height="300" alt="Screenshot 2026-04-09 145612" src="https://github.com/user-attachments/assets/aa8d260b-7793-4988-8378-3fa0d5cb169b" />
<img width="300" height="300" alt="Screenshot 2026-04-09 145435" src="https://github.com/user-attachments/assets/640892ac-2ebe-4f51-8b3a-5a5f41b1f72b" />
<img width="300" height="300" alt="Screenshot 2026-04-09 145456" src="https://github.com/user-attachments/assets/c854e42f-90d9-412d-96cb-7813437f6a0c" />

Another erosion result (same map, exported in-app):

<img width="300" height="300" alt="noise-main_export" src="https://github.com/user-attachments/assets/929fe38e-c4ea-4bef-9e1d-0616a8591706" />

More results:


<img width="300" height="300" alt="noise-main (10)" src="https://github.com/user-attachments/assets/5496fa7b-5ca1-44ef-b5bb-55f8bae2ca4a" />
<img width="300" height="300" alt="noise-main (10)_dem_export" src="https://github.com/user-attachments/assets/9032e907-3cbd-4dc7-b81a-a8b8fcb3ba93" />
<img width="300" height="300" alt="noise-main (10)_dem_export (2)" src="https://github.com/user-attachments/assets/635cf9d6-00d5-4d81-bcec-443cb0753002" />


<img width="300" height="300" alt="Screenshot 2026-04-08 165014" src="https://github.com/user-attachments/assets/8e22699c-df47-4ae1-b438-5380c780931e" />
<img width="300" height="300" alt="Screenshot 2026-04-08 163138" src="https://github.com/user-attachments/assets/82f4aa14-f01e-4645-b7ec-0cd70c8a95b2" />

And for fun using an image from our [webgpu fractal explorer](https://github.com/joshbrew/webgpu_fractals):

<img width="300" height="300" alt="fractal-1536-yfbvgz" src="https://github.com/user-attachments/assets/d26aac72-706d-4988-9bb7-2e2c5487edd8" />
<img width="300" height="300" alt="fractal-1536-yfbvgz_dem_export" src="https://github.com/user-attachments/assets/4de40ea7-a221-4a2a-9cf4-929cca641cc8" />
<img width="300" height="300" alt="fractal-1536-yfbvgz_dem_export (1)" src="https://github.com/user-attachments/assets/aed1d993-f820-4a92-8592-36fdfdecc278" />

This was heavily vibe coded, just FYI.

### License

MIT

Go crazy! Feel free to make pull requests for feature improvements/additions.



