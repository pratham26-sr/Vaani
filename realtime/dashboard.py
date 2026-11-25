"""Interactive dashboard: live stats, speaker timeline, transcript and controls."""
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn




def start_dashboard(convo, stats, ctrl, port=8000):
    """ctrl is a dict the main loop reads: bypass, reply_all, tts, listening, quit."""
    app = FastAPI()


    @app.get("/state")
    def state():
        turns, names, windows = convo.snapshot()
        return {"stats": stats, "names": names, "ctrl": ctrl, "turns": turns,
                "windows": [(a, b, l, convo.display(l)) for a, b, l in windows]}


    @app.post("/rename/{label}/{name}")
    def rename(label: str, name: str):
        convo.rename(label, name)
        return {"ok": True}


    @app.post("/toggle/{what}")
    def toggle(what: str):
        if what in ctrl:
            ctrl[what] = not ctrl[what]
        return {"ok": True, "value": ctrl.get(what)}


    @app.post("/say/{text}")
    def say(text: str):
        convo.add_typed(text)
        return {"ok": True}


    @app.post("/stop")
    def stop():
        ctrl["quit"] = True
        return {"ok": True}


    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE


    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()




PAGE = """
<!doctype html><meta charset=utf-8><title>Live Speech System</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;margin:0;padding:18px;background:#0d0d11;color:#e8e8ea}
 h1{font-size:19px;margin:0 0 2px} .sub{color:#8a8a94;font-size:12px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
 @media(max-width:820px){.grid{grid-template-columns:1fr}}
 .card{background:#16161c;border:1px solid #26262f;border-radius:10px;padding:14px}
 .card h3{margin:0 0 10px;font-size:12px;letter-spacing:.08em;color:#8a8a94;text-transform:uppercase}
 .nums{display:flex;gap:20px;flex-wrap:wrap}
 .n{font-size:26px;font-weight:600} .nl{font-size:11px;color:#8a8a94}
 #spark{display:flex;align-items:flex-end;gap:2px;height:56px;margin-top:10px}
 #spark div{flex:1;background:#3d7fd6;border-radius:2px 2px 0 0;min-height:2px}
 #meter{height:12px;background:#24242c;border-radius:6px;overflow:hidden;margin-top:10px}
 #bar{height:100%;width:0;background:linear-gradient(90deg,#39b36a,#d6c23d,#d65050);transition:width .1s}
 #tl{display:flex;gap:2px;height:34px}
 .b{flex:1;border-radius:3px;min-width:2px}
 .leg{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:12px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
 #log{max-height:430px;overflow-y:auto}
 .t{border-left:3px solid #555;padding:8px 11px;margin:8px 0;background:#1b1b22;border-radius:0 7px 7px 0}
 .who{font-weight:600;font-size:14px}
 .emo{font-size:10px;padding:2px 8px;border-radius:10px;margin-left:8px;vertical-align:middle}
 .txt{margin-top:3px;color:#d2d2d8;font-size:14px}
 .rep{color:#7fd4a0;margin-top:6px;font-size:13px;border-top:1px dashed #333;padding-top:6px}
 .wait{opacity:.45}
 button{font:inherit;font-size:13px;padding:7px 13px;margin:3px 4px 3px 0;background:#24242c;
        color:#e8e8ea;border:1px solid #35353f;border-radius:6px;cursor:pointer}
 button:hover{background:#2e2e38}
 .on{background:#2a6b45;border-color:#38885a} .off{background:#6b2a2a;border-color:#8a3a3a}
 input{font:inherit;font-size:13px;padding:7px;background:#1b1b22;color:#e8e8ea;
       border:1px solid #35353f;border-radius:6px}
 .row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
</style>


<h1>Real-time Speech Enhancement, Diarization &amp; Response</h1>
<div class=sub>U-Net enhancement &bull; CAM++ diarization &bull; streaming ASR &bull; emotion &bull; spoken reply</div>


<div class=grid>
 <div class=card>
  <h3>Performance</h3>
  <div class=nums>
   <div><div class=n id=lat>-</div><div class=nl>latency ms</div></div>
   <div><div class=n id=cmp>-</div><div class=nl>compute ms / 250 ms hop</div></div>
   <div><div class=n id=und>0</div><div class=nl>underruns</div></div>
   <div><div class=n id=spk>0</div><div class=nl>speakers</div></div>
  </div>
  <div id=spark></div>
  <div class=nl style="margin-top:8px">input level</div>
  <div id=meter><div id=bar></div></div>
 </div>


 <div class=card>
  <h3>Controls</h3>
  <button id=bListen onclick="tog('listening')">listening</button>
  <button id=bBypass onclick="tog('bypass')">audio</button>
  <button id=bReply onclick="tog('reply_all')">reply to all</button>
  <button id=bTts onclick="tog('tts')">voice output</button>
  <button onclick="stopIt()">stop session</button>
  <div class=row>
   <input id=lab placeholder="S1" style="width:70px">
   <input id=nm placeholder="Priyanshu" style="width:140px">
   <button onclick="ren()">rename speaker</button>
  </div>
  <div class=row>
   <input id=msg placeholder="type a message to the assistant" style="flex:1;min-width:180px">
   <button onclick="say()">send</button>
  </div>
  <h3 style="margin-top:16px">Who is speaking</h3>
  <div id=tl></div>
  <div class=leg id=leg></div>
 </div>
</div>


<div class=card style="margin-top:14px">
 <h3>Live transcript</h3>
 <div id=log></div>
</div>


<script>
var col={}, pal=['#4fc3f7','#ffb74d','#81c784','#ba68c8','#e57373'];
var emoCol={happy:'#2e7d32',angry:'#c62828',sad:'#1565c0',
            surprised:'#ef6c00',confused:'#6a1b9a',neutral:'#3a3a44'};
var hist=[];


function c(l){ if(!(l in col)) col[l]=pal[Object.keys(col).length%5]; return col[l]; }
function esc(s){ return String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function btn(id,on,onTxt,offTxt){
  var b=document.getElementById(id);
  b.className = on ? 'on' : 'off';
  b.textContent = on ? onTxt : offTxt;
}


async function tick(){
 try{
  var r = await fetch('/state');
  var s = await r.json();
  var t = s.stats || {};


  document.getElementById('lat').textContent = (t.latency_ms==null?'-':t.latency_ms);
  document.getElementById('cmp').textContent = (t.compute_ms==null?'-':t.compute_ms);
  document.getElementById('und').textContent = (t.underruns||0);
  document.getElementById('spk').textContent = Object.keys(col).length;


  if(t.latency_ms!=null){ hist.push(t.latency_ms); if(hist.length>60) hist.shift(); }
  var mx = Math.max.apply(null, hist.concat([700]));
  document.getElementById('spark').innerHTML =
    hist.map(function(v){ return '<div style="height:'+(v/mx*100)+'%"></div>'; }).join('');


  document.getElementById('bar').style.width = Math.min(100,(t.rms||0)*400) + '%';


  btn('bListen', s.ctrl.listening, 'listening: ON', 'listening: PAUSED');
  btn('bBypass', !s.ctrl.bypass, 'audio: ENHANCED', 'audio: RAW');
  btn('bReply',   s.ctrl.reply_all, 'reply to all: ON', 'reply to all: OFF');
  btn('bTts',     s.ctrl.tts, 'voice output: ON', 'voice output: OFF');


  document.getElementById('tl').innerHTML = s.windows.map(function(x){
    return '<div class=b title="'+esc(x[3])+' '+x[0].toFixed(1)+'s" style="background:'+c(x[2])+'"></div>';
  }).join('');


  document.getElementById('leg').innerHTML = Object.keys(col).map(function(l){
    return '<span><span class=dot style="background:'+col[l]+'"></span>'+esc(s.names[l]||l)+'</span>';
  }).join('');


  document.getElementById('log').innerHTML = s.turns.slice().reverse().map(function(e){
    var lc  = e.label ? c(e.label) : '#555';
    var emo = e.emotion ? '<span class=emo style="background:'+(emoCol[e.emotion]||'#3a3a44')+'">'
                          + esc(e.emotion) + '</span>' : '';
    var rep = e.reply ? '<div class=rep>&#8594; '+esc(e.reply)+'</div>'
            : (e.status==='thinking' ? '<div class=rep>thinking...</div>' : '');
    var cls = (e.status==='waiting') ? 't wait' : 't';
    return '<div class="'+cls+'" style="border-color:'+lc+'">'
         + '<span class=who>'+esc(e.speaker||'...')+'</span>'+emo
         + '<div class=txt>'+esc(e.text)+'</div>'+rep+'</div>';
  }).join('');
 }catch(err){}
}


async function tog(w){ await fetch('/toggle/'+w,{method:'POST'}); tick(); }
async function stopIt(){
  if(confirm('Stop the session and save audio?')) await fetch('/stop',{method:'POST'});
}
async function ren(){
  var l=document.getElementById('lab').value, n=document.getElementById('nm').value;
  if(!l||!n) return;
  await fetch('/rename/'+encodeURIComponent(l)+'/'+encodeURIComponent(n),{method:'POST'});
  document.getElementById('nm').value=''; tick();
}
async function say(){
  var m=document.getElementById('msg');
  if(!m.value) return;
  await fetch('/say/'+encodeURIComponent(m.value),{method:'POST'});
  m.value=''; tick();
}
document.getElementById('msg').addEventListener('keydown',function(e){ if(e.key==='Enter') say(); });
setInterval(tick,350); tick();
</script>
"""