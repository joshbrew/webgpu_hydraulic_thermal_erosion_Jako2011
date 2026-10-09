import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import {pathToFileURL} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE?pathToFileURL(process.env.PLAYWRIGHT_MODULE).href:'playwright');
const root=resolve('.'),server=http.createServer(async(req,res)=>{try{const p=resolve(root,'.'+req.url);if(!p.startsWith(root+'\\')&&!p.startsWith(root+'/'))throw Error('path');res.writeHead(200,{'Content-Type':extname(p)==='.html'?'text/html':'text/javascript'});res.end(await readFile(p));}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-unsafe-webgpu']});
try{const page=await browser.newPage();page.on('pageerror',e=>console.log(e.message));await page.goto(`http://127.0.0.1:${server.address().port}/tests/mesh-water.html`);await page.waitForFunction(()=>window.meshWaterResults,undefined,{timeout:120000});const result=await page.evaluate(()=>window.meshWaterResults);console.log(JSON.stringify(result,null,2));if(!result.passed)process.exitCode=1;}finally{await browser.close();server.close();}
