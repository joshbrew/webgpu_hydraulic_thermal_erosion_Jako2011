import assert from 'node:assert/strict';
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import {pathToFileURL} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE?pathToFileURL(process.env.PLAYWRIGHT_MODULE).href:'playwright');
const root=resolve('.'),server=http.createServer(async(req,res)=>{try{const p=resolve(root,'.'+req.url.split('?')[0]);if(!p.startsWith(root+'\\')&&!p.startsWith(root+'/'))throw Error('path');res.writeHead(200,{'Content-Type':({'.html':'text/html','.css':'text/css','.png':'image/png'})[extname(p)]||'text/javascript'});res.end(await readFile(p));}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-unsafe-webgpu']});
try{const page=await browser.newPage({viewport:{width:1550,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'&&!m.text().includes('404'))errors.push(m.text());});
  for(const protocol of ['http','file']){
    const base=protocol==='http'?`http://127.0.0.1:${server.address().port}/`:pathToFileURL(root+'/').href;
    await page.goto(base+'index.html');
    await page.locator('label').filter({hasText:/^Example map preset/}).locator('select').selectOption('lowland_river');
    await page.locator('label').filter({hasText:/^Size/}).locator('select').selectOption('256');
    await page.getByRole('button',{name:'Load example',exact:true}).click();
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Run'&&!b.disabled),undefined,{timeout:90000});
    console.log(`${protocol}: original preset loaded`);
    const build=page.getByRole('button',{name:'Build hydraulic mesh',exact:true});
    await page.getByRole('button',{name:'Run',exact:true}).click();assert(await build.isDisabled(),'Mesh build must disable during erosion');
    await page.waitForTimeout(500);await page.getByRole('button',{name:'Pause',exact:true}).click();
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Build hydraulic mesh'&&!b.disabled));
    await page.getByLabel('Mesh fidelity',{exact:true}).selectOption('high');await page.getByLabel('Triangle limit',{exact:true}).fill('1500');
    await build.click();console.log(`${protocol}: requested mesh build`);await page.getByRole('button',{name:'Run mesh water',exact:true}).waitFor();
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Run mesh water'&&!b.disabled),undefined,{timeout:30000}).catch(async error=>{console.log(await page.locator('.mesh-controls').innerText(),errors);throw error;});
    const text=await page.locator('.mesh-controls').innerText(),count=Number(text.match(/([\d,]+) triangles \/ /)[1].replaceAll(',',''));
    assert(count<=1500&&count>10,'Requested triangle size limit must be enforced');
    // Worker rendering and canvas presentation happen after the control reply.
    await page.waitForTimeout(200);
    const before=await page.locator('canvas').nth(1).screenshot();
    await page.getByLabel('Mesh wireframe',{exact:true}).uncheck();
    let changed=false;
    for(let attempt=0;attempt<30&&!changed;attempt++){
      await page.waitForTimeout(100);
      changed=!before.equals(await page.locator('canvas').nth(1).screenshot());
    }
    assert(changed,'Wireframe must affect the original demo');
    await page.getByRole('button',{name:'Run mesh water',exact:true}).click();await page.getByRole('button',{name:'Pause mesh water',exact:true}).waitFor();
    assert(await page.getByLabel('Erosion preset',{exact:true}).isDisabled(),'Erosion presets must not pretend to modify the separate running mesh-water solver');
    await page.getByText('Camera and navigation',{exact:true}).click();
    await page.locator('label').filter({hasText:/^Azimuth/}).locator('input').fill('90');await page.waitForTimeout(100);
    assert(await page.getByRole('button',{name:'Pause mesh water',exact:true}).isEnabled(),'Camera changes must preserve the running water mesh');
    await page.getByRole('button',{name:'Reset view',exact:true}).click();
    await page.waitForTimeout(600);await page.getByRole('button',{name:'Pause mesh water',exact:true}).click();
    await page.getByRole('button',{name:'Inspect mesh water',exact:true}).click();
    await page.waitForFunction(()=>document.body.textContent.includes('invalid cells'),undefined,{timeout:20000});
    assert((await page.locator('.mesh-controls').innerText()).includes('0 invalid cells'),'Mesh water must stay finite');
    await page.getByRole('button',{name:'Reset mesh water',exact:true}).click();
    await page.getByRole('button',{name:'Run',exact:true}).click();await page.waitForTimeout(300);await page.getByRole('button',{name:'Pause',exact:true}).click();
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Run mesh water'&&b.disabled));
    console.log(`Original demo ${protocol}: presets, pause-only size-limited mesh, wireframe, GPU water, inspection/reset and erosion resume passed (${count} triangles).`);
    await page.goto(base+'examples/index.html');
    const card=page.locator('article').filter({has:page.locator('[data-terrain]')});
    const first=page.locator('article').filter({has:page.getByRole('button',{name:'Run sim',exact:true})}).first();
    await first.getByRole('button',{name:'Run sim',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.live-panel')?.dataset.running==='true');
    await page.getByRole('button',{name:'Pause',exact:true}).click();await page.getByLabel('Triangle limit',{exact:true}).fill('2000');
    await page.getByRole('button',{name:'Build hydraulic mesh',exact:true}).click();
    await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='Run mesh water'&&!b.disabled),undefined,{timeout:90000});
    await page.getByRole('button',{name:'Run mesh water',exact:true}).click();await page.waitForTimeout(500);await page.getByRole('button',{name:'Pause mesh water',exact:true}).click();
    await page.getByRole('button',{name:'Inspect mesh water',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.mesh-controls')?.textContent.includes('0 invalid cells'),undefined,{timeout:20000});
    await page.getByRole('button',{name:'Run',exact:true}).click();await page.waitForTimeout(300);await page.getByRole('button',{name:'Pause',exact:true}).click();
    assert(await page.getByRole('button',{name:'Run mesh water',exact:true}).isDisabled());
    if(protocol==='http')await page.screenshot({path:resolve(root,'tests/mesh-gallery.png'),fullPage:false});
    console.log(`Gallery ${protocol}: controlled hydraulic mesh and GPU water controls passed.`);
  }
  assert.deepEqual(errors,[]);
}finally{await browser.close();server.close();}
