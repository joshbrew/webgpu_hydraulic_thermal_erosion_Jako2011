import { TERRAIN_EXAMPLES } from './terrainExamples.js';

// Arrange existing controls without changing their handlers or simulation state.
export default function mountStudio(ui) {
  const node=(tag,className,text)=>{const el=document.createElement(tag);el.className=className;if(text)el.textContent=text;return el;};
  const link=(href,title)=>{const a=node('a','nav-link',title);a.href=href;return a;};
  const {root,heading,controls,layout,viewerPanel,viewerTitle,canvasWrap,sidePanel,settingsStack}=ui;
  root.className='erosion-studio';root.style.cssText='';layout.className='studio-workspace';layout.style.cssText='';
  viewerPanel.className='terrain-panel';viewerPanel.style.cssText='';canvasWrap.classList.add('terrain-canvas');
  sidePanel.className='studio-settings';sidePanel.style.cssText='';settingsStack.className='settings-stack';settingsStack.style.cssText='';
  const header=node('header','studio-header'),brand=node('div','studio-brand');
  heading.textContent='Erosion studio';heading.style.cssText='';
  brand.append(heading,node('p','','Shape rivers, canyons and layered terrain.'));
  const nav=node('nav','studio-nav');nav.setAttribute('aria-label','Demo navigation');
  const home=link('index.html','Studio');home.setAttribute('aria-current','page');
  nav.append(home,link('examples/index.html','Visual examples'),link('examples/erode-mesh.html','Final mesh demo'),link('../hydraulic_triangular_irregular_networks/index.html','Triangle mesh studio'));
  header.append(brand,nav);
  const demos=node('div','demo-shortcuts');
  for(const [href,title,description,image] of [
    ['examples/index.html','Visual examples','Before & after images · run any result','examples/volcano-volcanic_cone-after.png'],
    ['examples/erode-mesh.html?scene=lowland_river&seconds=180&mesh=1','Erosion → final mesh','Evolved river · compare grid and triangles','examples/lowland_river-lowland_meanders-after.png'],
    ['../hydraulic_triangular_irregular_networks/index.html','Triangle mesh studio','Terrain, river guides and mesh overlays','examples/mountain_canyon-mountain_canyon-after.png']
  ]){
    const a=node('a','demo-shortcut');a.href=href;const img=node('img','');img.src=image;img.alt='';img.width=120;img.height=80;
    const text=node('div','');text.append(node('strong','',title),node('span','',description));a.append(img,text,node('span','shortcut-arrow','↗'));demos.append(a);
  }
  const map=node('section','map-card'),mapHeading=node('h3','','Starting terrain'),mapImage=node('img','map-thumbnail');mapImage.alt='Starting terrain preview';
  const mapInfo=node('p','map-info','Selecting a map loads its terrain and recommended erosion settings.');
  ui.exampleLabel.input.setAttribute('aria-label','Map preset');ui.presetLabel.input.setAttribute('aria-label','Erosion preset');
  const mapFields=node('div','map-fields'),mapSizing=node('div','field-row');mapSizing.append(ui.exampleSizeLabel,ui.exampleSeedLabel);
  mapFields.append(ui.exampleLabel,mapSizing,ui.generateExampleButton);
  const importPanel=node('details','import-panel');importPanel.append(node('summary','','Import a DEM image'),ui.fileLabel);
  map.append(mapHeading,mapImage,mapFields,mapInfo,importPanel);
  const profile=node('section','profile-card');profile.append(node('h3','','Erosion settings'),ui.presetLabel);
  const applied=node('p','applied-profile');applied.setAttribute('role','status');
  ui.applyPresetButton.classList.add('subtle-button');ui.applyPresetButton.title='Reapply the selected settings after manual adjustments';
  profile.append(applied,ui.presetNote,ui.applyPresetButton);
  const toolbar=node('div','simulation-toolbar'),playback=node('div','playback-controls');
  ui.runButton.classList.add('primary-button');playback.append(ui.runButton,ui.stepButton,ui.resetButton);
  const utilities=node('div','terrain-utilities');utilities.append(ui.cameraResetButton,ui.inspectMassButton,ui.exportDemButton,ui.clearButton);
  toolbar.append(playback,utilities);
  const titlebar=node('div','terrain-titlebar');viewerTitle.className='terrain-title';viewerTitle.style.cssText='';
  ui.statusLine.className='terrain-status';ui.statusLine.style.cssText='';ui.statusLine.setAttribute('role','status');titlebar.append(viewerTitle,ui.statusLine);
  const interaction=node('p','interaction-hint','Drag to orbit · scroll to zoom · Run continues until paused');
  const meshPanel=node('details','mesh-panel');meshPanel.open=true;
  meshPanel.append(node('summary','','Final mesh & water'),ui.meshControls.root,ui.meshDemoLink);
  ui.massNote.classList.add('mass-note');viewerPanel.replaceChildren(titlebar,toolbar,canvasWrap,interaction,ui.massNote,meshPanel);
  const statistics=node('details','setting-panel statistics-panel');ui.statsTitle.remove();statistics.append(node('summary','','Detailed statistics'),ui.statsBlock);
  sidePanel.replaceChildren(map,profile,settingsStack,statistics);
  controls.remove();root.replaceChildren(header,demos,layout);
  return {
    showExample(name,loading=false){const map=TERRAIN_EXAMPLES[name];viewerTitle.textContent=map.label;mapImage.src=`examples/${name}-${map.preset}-before.png`;mapImage.alt=`${map.label} starting terrain`;mapInfo.textContent=loading?'Loading terrain…':'Map selection loads immediately. Size and seed apply when you choose Load example.';},
    showProfile(label){applied.textContent=`Applied · ${label}`;}
  };
}
