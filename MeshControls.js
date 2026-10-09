import { MESH_FIDELITY } from './MeshSettings.js';

// Shared controls for the worker-owned demo and the reusable gallery viewport.
export default function createMeshControls({ onbuild, onview, onwater }={}) {
  const root=document.createElement('fieldset');root.className='mesh-controls';root.style.cssText='border:1px solid #46636f;border-radius:10px;padding:12px;margin:12px 0;color:#d9e8ed';
  const legend=document.createElement('legend');legend.textContent='Final hydraulic mesh · build while paused';root.append(legend);
  const body=document.createElement('div');body.style.cssText='display:flex;flex-wrap:wrap;align-items:center;gap:10px';root.append(body);
  let row=body;
  function nextRow(title){row=document.createElement('div');row.style.cssText=body.style.cssText+';border-top:1px solid #46636f;padding-top:10px;margin-top:12px';
    const heading=document.createElement('strong');heading.textContent=title;heading.style.cssText='font-size:12px;flex-basis:100%';row.append(heading);root.append(row);}
  const controls={};
  function field(id,title,type,value,items){const label=document.createElement('label');label.style.cssText='display:flex;align-items:center;gap:6px';label.textContent=title+' ';
    const input=document.createElement(type==='select'?'select':'input');input.setAttribute('aria-label',title);input.dataset.mesh=id;
    if(items)for(const[v,text]of items){const o=document.createElement('option');o.value=v;o.textContent=text;input.append(o);}
    else input.type=type;
    input.value=value;if(type==='checkbox')input.checked=value;
    input.style.cssText='background:#20333f;color:#e6f0f2;border:1px solid #46636f;border-radius:6px;padding:6px;max-width:170px';
    if(type==='number'){input.min='1';input.max=id==='budget'?'250000':'512';input.style.width='90px';input.step='1';}
    label.append(input);row.append(label);controls[id]=input;return input;}
  const fidelity=field('fidelity','Mesh fidelity','select','balanced',[['preview','Preview'],['balanced','Balanced'],['high','High'],['custom','Custom']]);
  field('mode','Mesh priority','select','hybrid',[['hybrid','Terrain + rivers'],['traditional','Relief'],['hydrographic','Channels'],['hydrological','Catchments']]);
  field('min','Min spacing','number',3);field('max','Max spacing','number',20);field('budget','Triangle limit','number',16000);field('guides','River guides','checkbox',true);
  controls.budget.min=128;
  const button=(text)=>{const b=document.createElement('button');b.type='button';b.textContent=text;b.style.cssText='background:#24585c;color:#e6f0f2;border:1px solid #46636f;border-radius:6px;padding:8px;cursor:pointer';row.append(b);return b;};
  const build=button('Build hydraulic mesh');
  nextRow('Surface comparison');
  field('surface','Mesh surface','select','adaptive',[['adaptive','Adaptive triangles'],['grid','Full grid']]);
  field('wire','Mesh wireframe','checkbox',true);
  const opacity=field('opacity','Wire opacity','range',.55);opacity.min=0;opacity.max=1;opacity.step=.05;opacity.value=.55;
  nextRow('Water over the frozen mesh');
  const water=button('Run mesh water'),resetWater=button('Reset mesh water'),inspect=button('Inspect mesh water');
  const roughness=field('roughness','Water roughness','number',.025);roughness.min=0;roughness.max=.2;roughness.step=.005;
  const dt=field('dt','Water max step','number',.02);dt.min=.0001;dt.max=1;dt.step=.005;
  const rain=field('rain','Mesh rain','number',.001);rain.min=0;rain.max=1;rain.step=.001;
  const note=document.createElement('div');note.style.cssText='font-size:12px;color:#afc5cf;margin-top:10px;white-space:pre-line';root.append(note);
  let info=null,busy=false,running=false,waterRunning=false,waterSteps=0,ready=false;
  const options=()=>({mode:controls.mode.value,minSpacing:Number(controls.min.value),maxSpacing:Number(controls.max.value),maxTriangles:Number(controls.budget.value),guideEnabled:controls.guides.checked});
  const view=()=>({adaptive:controls.surface.value==='adaptive',wireframe:controls.wire.checked,wireOpacity:Number(controls.opacity.value)});
  const waterOptions=()=>({roughness:Number(roughness.value),maxTimeStep:Number(dt.value),rainRate:Number(rain.value)});
  fidelity.onchange=()=>{const p=MESH_FIDELITY[fidelity.value];if(p){controls.min.value=p.minSpacing;controls.max.value=p.maxSpacing;controls.budget.value=p.maxTriangles;}};
  for(const id of ['min','max','budget'])controls[id].oninput=()=>{fidelity.value='custom';};
  for(const id of ['surface','wire','opacity'])controls[id].oninput=()=>onview?.(view());
  build.onclick=()=>onbuild?.(options());water.onclick=()=>onwater?.(waterRunning?'pause':'run',waterOptions());resetWater.onclick=()=>onwater?.('reset',waterOptions());inspect.onclick=()=>onwater?.('inspect',waterOptions());
  function update(next={}){if('info'in next)info=next.info;if('busy'in next)busy=next.busy;if('running'in next)running=next.running;if('waterRunning'in next)waterRunning=next.waterRunning;
    if('ready'in next)ready=next.ready;if('waterSteps'in next)waterSteps=next.waterSteps;
    for(const[id,c]of Object.entries(controls))c.disabled=busy||running||(waterRunning&&!['wire','opacity'].includes(id))||(!info&&['surface','wire','opacity'].includes(id));
    build.disabled=busy||running||waterRunning||!ready;water.disabled=resetWater.disabled=busy||running||!info;inspect.disabled=busy||running||waterRunning||!info;
    water.textContent=waterRunning?'Pause mesh water':'Run mesh water';
    if(waterRunning)controls.surface.value='adaptive';
    note.textContent=next.message||(info?`${info.triangles.toLocaleString()} triangles / ${info.gridTriangles.toLocaleString()} grid · ${info.vertices.toLocaleString()} vertices · ${(info.buildMs/1000).toFixed(2)} s build\nBuilt spacing ${info.minSpacing.toFixed(1)}–${info.maxSpacing.toFixed(1)} cells · limit ${info.maxTriangles.toLocaleString()} triangles\nSampled height error: RMS ${info.sampledRmsError.toFixed(5)} · max ${info.sampledMaxError.toFixed(5)} · frozen at ${info.time.toFixed(2)} s${waterSteps?`\nMesh water: ${waterSteps.toLocaleString()} steps · ${waterRunning?'running':'paused'}`:''}\nMesh water keeps the terrain fixed; resuming erosion returns to the grid.`:'Pause erosion to build. Fidelity and the triangle limit control detail and size. Water over the fixed mesh runs separately from erosion.');}
  update({ready:false});return {root,update,options,view,waterOptions};
}
