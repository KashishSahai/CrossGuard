"""Bloom City renderer: self-contained HTML (inline SVG + small JS player).

Designed for `streamlit.components.v1.html`, which renders in a real iframe - unlike
`st.markdown(unsafe_allow_html=True)`, which sanitises SVG/JS and caused the blank canvas.
Python is authoritative: this module only draws frames recorded by BloomCityEngine.
"""
from __future__ import annotations

import json
from typing import List, Sequence

from core.models import PHASE_LABELS, PHASE_ORDER
from . import geometry as G
from .simulation_agents import KIND_ORDER, KIND_SPECS

# ----------------------------------------------------------------------------- static scene
PALETTE = dict(grass="#dff1e2", block="#e8f6e9", road="#cbc6de", box="#c4bfd9", walk="#f7f3ff",
               curb="#ddd5f1", line="#fffdfb", center="#f5d98e")


def _building(x, y, w, h, fill, roof, cols=4, rows=2):
    wins = []
    pad = 10
    gw = (w - 2 * pad) / cols
    gh = (h - 2 * pad - 14) / rows
    for r in range(rows):
        for c in range(cols):
            wins.append(f'<rect x="{x + pad + c * gw + 3:.1f}" y="{y + 22 + r * gh + 3:.1f}" '
                        f'width="{gw - 8:.1f}" height="{gh - 8:.1f}" rx="3" fill="#ffffff" opacity=".75"/>')
    return (f'<rect x="{x + 6}" y="{y + 6}" width="{w}" height="{h}" rx="10" fill="#6a4fb3" opacity=".12"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="16" rx="8" fill="{roof}"/>' + "".join(wins))


def _tree(x, y, r=20):
    return (f'<circle cx="{x + 4}" cy="{y + 5}" r="{r}" fill="#6a4fb3" opacity=".12"/>'
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="#9fd8a8"/>'
            f'<circle cx="{x - r * .3:.1f}" cy="{y - r * .3:.1f}" r="{r * .55:.1f}" fill="#b6e6bd"/>'
            f'<circle cx="{x + r * .35:.1f}" cy="{y + r * .25:.1f}" r="{r * .35:.1f}" fill="#8ccf99"/>')


def _flowers(x, y, n=7, spread=22):
    cols = ["#ff9ec7", "#ffd27a", "#c9a7ff", "#ffffff", "#ffb3a7"]
    out = []
    for i in range(n):
        dx = ((i * 37) % (2 * spread)) - spread
        dy = ((i * 53) % (2 * spread)) - spread
        out.append(f'<circle cx="{x + dx}" cy="{y + dy * .6:.1f}" r="3.6" fill="{cols[i % 5]}"/>'
                   f'<circle cx="{x + dx}" cy="{y + dy * .6:.1f}" r="1.2" fill="#ffe8a3"/>')
    return f'<ellipse cx="{x}" cy="{y}" rx="{spread + 8}" ry="{spread * .6 + 8:.0f}" fill="#c9ecc9"/>' + "".join(out)


def _bench(x, y):
    return (f'<rect x="{x - 14}" y="{y - 4}" width="28" height="8" rx="3" fill="#d6a77a"/>'
            f'<rect x="{x - 14}" y="{y - 7}" width="28" height="3" rx="1.5" fill="#b98a5e"/>')


def _cafe(x, y):
    stripes = "".join(f'<rect x="{x + i * 16}" y="{y + 72}" width="16" height="22" '
                      f'fill="{"#ffb3c7" if i % 2 == 0 else "#fff4f7"}"/>' for i in range(8))
    tables = "".join(f'<circle cx="{x + 24 + i * 40}" cy="{y + 112}" r="10" fill="#fff"/>'
                     f'<circle cx="{x + 24 + i * 40}" cy="{y + 112}" r="10" fill="none" stroke="#ffb3c7" stroke-width="2"/>'
                     f'<circle cx="{x + 12 + i * 40}" cy="{y + 112}" r="4" fill="#e4d6ff"/>'
                     f'<circle cx="{x + 36 + i * 40}" cy="{y + 112}" r="4" fill="#e4d6ff"/>' for i in range(3))
    return (_building(x, y, 128, 74, "#fff0e0", "#ffc9a8", 4, 1) + stripes + tables +
            f'<text x="{x + 64}" y="{y + 12}" font-size="12" font-weight="800" text-anchor="middle" '
            f'fill="#7a4a2c">CAFÉ BLOOM</text>')


def _static_scene() -> str:
    P, c = PALETTE, G
    s: List[str] = []
    s.append(f'<rect width="1000" height="700" fill="{P["grass"]}"/>')
    # quadrant lawns
    for (x, y, w, h) in ((0, 0, 402, 252), (598, 0, 402, 252), (0, 448, 402, 252), (598, 448, 402, 252)):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{P["block"]}"/>')
    # NW houses
    s += [_building(22, 22, 118, 92, "#ffd9d2", "#ffb9ad", 3, 1), _building(156, 22, 118, 92, "#fff0c2", "#ffd98a", 3, 1),
          _building(290, 22, 96, 92, "#e0d4ff", "#c4aeff", 2, 1)]
    s += [_tree(60, 170, 22), _tree(150, 190, 18), _tree(248, 168, 24), _tree(340, 192, 18),
          _flowers(104, 206, 8), _flowers(300, 218, 7)]
    # NE cafe + shops
    s += [_cafe(628, 18), _building(776, 22, 100, 92, "#d6f0ff", "#aedcf7", 2, 1),
          _building(890, 22, 90, 92, "#ffe1f0", "#ffc0de", 2, 1)]
    s += [_tree(660, 210, 20), _tree(800, 200, 22), _tree(930, 196, 18), _flowers(730, 222, 7),
          _bench(860, 226)]
    # SW apartments
    s += [_building(24, 586, 150, 90, "#e7e1ff", "#cbbdff", 4, 2), _building(190, 596, 120, 80, "#dff5e9", "#b9e8d0", 3, 1)]
    s += [_tree(60, 500, 22), _tree(150, 520, 18), _tree(250, 498, 22), _tree(340, 526, 18),
          _flowers(106, 548, 8), _flowers(310, 566, 6), _bench(210, 470)]
    # SE park with pond
    s.append('<ellipse cx="800" cy="568" rx="92" ry="52" fill="#bfe6f7"/>'
             '<ellipse cx="800" cy="568" rx="92" ry="52" fill="none" stroke="#e9f8ff" stroke-width="4"/>'
             '<ellipse cx="780" cy="560" rx="28" ry="9" fill="#ffffff" opacity=".35"/>')
    s += [_tree(660, 496, 24), _tree(930, 490, 22), _tree(670, 640, 22), _tree(930, 640, 24),
          _tree(740, 666, 16), _flowers(880, 520, 8), _flowers(716, 520, 7), _flowers(900, 604, 6),
          _bench(800, 650)]
    # roads
    s.append(f'<rect x="0" y="{c.CY - 70}" width="1000" height="140" fill="{P["road"]}"/>')
    s.append(f'<rect x="{c.CX - 70}" y="0" width="140" height="700" fill="{P["road"]}"/>')
    # sidewalks (never drawn across a road)
    for (x, y, w, h) in ((0, 252, 430, 28), (570, 252, 430, 28), (0, 420, 430, 28), (570, 420, 430, 28),
                         (402, 0, 28, 280), (570, 0, 28, 280), (402, 420, 28, 280), (570, 420, 28, 280)):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{P["walk"]}"/>')
    s.append(f'<rect x="430" y="280" width="140" height="140" fill="{P["box"]}"/>')
    cu = f'stroke="{P["curb"]}" stroke-width="3"'
    for x in (430, 570):
        s.append(f'<line x1="{x}" y1="0" x2="{x}" y2="280" {cu}/><line x1="{x}" y1="420" x2="{x}" y2="700" {cu}/>')
    for y in (280, 420):
        s.append(f'<line x1="0" y1="{y}" x2="430" y2="{y}" {cu}/><line x1="570" y1="{y}" x2="1000" y2="{y}" {cu}/>')
    # centre lines
    cl = f'stroke="{P["center"]}" stroke-width="3" stroke-dasharray="16 12"'
    s.append(f'<line x1="500" y1="0" x2="500" y2="196" {cl}/><line x1="500" y1="504" x2="500" y2="700" {cl}/>')
    s.append(f'<line x1="0" y1="350" x2="346" y2="350" {cl}/><line x1="654" y1="350" x2="1000" y2="350" {cl}/>')
    # zebra crossings
    z = P["line"]
    for x in range(432, 568, 14):
        s.append(f'<rect x="{x}" y="214" width="8" height="32" fill="{z}"/><rect x="{x}" y="454" width="8" height="32" fill="{z}"/>')
    for y in range(282, 418, 14):
        s.append(f'<rect x="604" y="{y}" width="32" height="8" fill="{z}"/><rect x="364" y="{y}" width="32" height="8" fill="{z}"/>')
    # stop lines
    s.append(f'<rect x="430" y="206" width="70" height="6" fill="{z}"/><rect x="500" y="488" width="70" height="6" fill="{z}"/>'
             f'<rect x="638" y="280" width="6" height="70" fill="{z}"/><rect x="356" y="350" width="6" height="70" fill="{z}"/>')
    # waiting zones (curb ends of every crossing)
    for cw in range(4):
        for end in G.CROSSWALK_ENDS[cw]:
            w, h = (26, 40) if cw % 2 == 0 else (40, 26)
            s.append(f'<rect x="{end[0] - w / 2}" y="{end[1] - h / 2}" width="{w}" height="{h}" rx="8" '
                     f'fill="#e4d6ff" opacity=".55" stroke="#b79bff" stroke-width="1.5" stroke-dasharray="4 3"/>')
    # lamps, bins
    for (x, y) in ((408, 246), (592, 246), (592, 454), (408, 454)):
        s.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#8f86b5"/>')
    # compass labels
    lab = 'font-size="15" font-weight="800" fill="#ffffff" opacity=".8" text-anchor="middle"'
    s.append(f'<text x="500" y="30" {lab}>NORTH</text><text x="500" y="684" {lab}>SOUTH</text>'
             f'<text x="46" y="355" {lab}>WEST</text><text x="954" y="355" {lab}>EAST</text>')
    return "".join(s)


HEADS = {0: (416, 174), 1: (664, 266), 2: (584, 526), 3: (336, 434)}
PED_SIGNALS = ((416, 266), (584, 266), (584, 434), (416, 434))


def _signal_layer() -> str:
    out = []
    for i, (x, y) in HEADS.items():
        out.append(f'<g id="vh{i}"><rect x="{x - 12}" y="{y - 34}" width="24" height="68" rx="9" fill="#3d3566"/>'
                   f'<rect x="{x - 12}" y="{y - 34}" width="24" height="68" rx="9" fill="none" stroke="#ffffff" stroke-width="2"/>')
        for j, (k, col) in enumerate((("r", "#ff5d6c"), ("y", "#ffd23f"), ("g", "#38d889"))):
            out.append(f'<circle id="vl{i}{k}" data-on="{col}" cx="{x}" cy="{y - 22 + j * 22}" r="8.5" fill="#575079" opacity=".5"/>')
        out.append('</g>')
    for i, (x, y) in enumerate(PED_SIGNALS):
        out.append(f'<g id="pg{i}"><rect x="{x - 14}" y="{y - 14}" width="28" height="28" rx="8" fill="#3d3566" stroke="#fff" stroke-width="2"/>'
                   f'<circle id="pl{i}" cx="{x}" cy="{y}" r="10" fill="#ff5d6c"/>'
                   f'<text id="pt{i}" x="{x}" y="{y + 5}" font-size="14" font-weight="900" text-anchor="middle" fill="#fff">✋</text></g>')
    return "".join(out)


def _lamp_glow() -> str:
    return "".join(f'<circle cx="{x}" cy="{y}" r="58" fill="url(#glow)"/><circle cx="{x}" cy="{y}" r="5" fill="#fff4b8"/>'
                   for (x, y) in ((408, 246), (592, 246), (592, 454), (408, 454)))


# ----------------------------------------------------------------------------- HTML shell
_CSS = """
:root{--ink:#2b1f4f;--purple:#5b34b8;--lav:#f3efff;--card:#ffffff;--line:#e3dbf7;--soft:#7b6ca8;
--go:#1f9d63;--warn:#c98a00;--stop:#d6384a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:transparent;
font-family:"Nunito","Segoe UI Rounded","Segoe UI",system-ui,sans-serif;color:var(--ink)}
#bc{display:flex;flex-direction:column;gap:10px;height:100%;padding:2px}
.main{display:flex;gap:12px;flex:1;min-height:0}
.stage{position:relative;flex:1 1 0;min-width:0;background:var(--card);border-radius:22px;
box-shadow:0 8px 28px rgba(91,52,184,.16);overflow:hidden;display:flex;align-items:center;justify-content:center}
.stage svg{width:100%;height:100%;display:block}
.tag{position:absolute;left:12px;top:12px;background:rgba(255,255,255,.94);border-radius:999px;padding:6px 14px;
font-weight:800;font-size:13px;color:var(--purple);box-shadow:0 2px 8px rgba(91,52,184,.18)}
.note{position:absolute;left:12px;bottom:12px;background:rgba(255,255,255,.94);border-radius:12px;padding:5px 12px;
font-size:12.5px;font-weight:700;color:var(--soft)}
.ctl{position:absolute;right:12px;bottom:12px;display:flex;gap:6px}
.ctl button{border:0;border-radius:999px;background:var(--purple);color:#fff;font-weight:800;font-size:13px;
padding:7px 14px;cursor:pointer;font-family:inherit}.ctl button:focus-visible{outline:3px solid #b79bff}
.panel{width:clamp(250px,27%,290px);flex:0 0 clamp(250px,27%,290px);background:var(--card);border-radius:22px;padding:16px 14px 12px;
box-shadow:0 8px 28px rgba(91,52,184,.16);display:flex;flex-direction:column;gap:10px;overflow:auto}
.lbl{font-size:12.5px;font-weight:800;color:var(--soft)}
#dec{font-size:24px;line-height:1.15;font-weight:900;color:var(--purple)}
#why{font-size:14px;line-height:1.35;font-weight:600;margin:0}
.sigs{display:flex;flex-direction:column;gap:8px}
.sig{border-radius:16px;padding:8px 12px;background:var(--lav);border:2px solid var(--line);
display:grid;grid-template-columns:1fr auto;align-items:center;column-gap:8px}
.sig .k{grid-column:1/3;font-size:12px;font-weight:800;color:var(--soft)}
.sig b{font-size:22px;line-height:1.1;font-weight:900;white-space:nowrap}.sig .a{font-size:14px;font-weight:900;text-align:right}
.dot{display:inline-block;width:16px;height:16px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.cnt{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.cnt div{background:var(--lav);border-radius:14px;padding:7px 10px}.cnt b{font-size:22px;font-weight:900;display:block;line-height:1.05}
.cnt span{font-size:11.5px;font-weight:800;color:var(--soft)}
.phases{display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap}
.chip{padding:7px 14px;border-radius:999px;background:#fff;border:2px solid var(--line);font-weight:800;font-size:13.5px;color:var(--soft)}
.chip.on{background:var(--purple);border-color:var(--purple);color:#fff;box-shadow:0 4px 14px rgba(91,52,184,.35)}
.arr{color:var(--soft);font-weight:900}
#tm{font-size:12.5px;font-weight:800;color:var(--soft)}
.lw{font-size:13px;font-weight:800;color:var(--ink)}
@media (max-width:640px){
  #bc{height:auto;min-height:100%}
  .main{flex-direction:column;flex:none}
  .stage{width:100%;aspect-ratio:10/7;min-height:280px;flex:none}
  .panel{width:100%;flex:none}
}
@media (prefers-reduced-motion:reduce){.chip{transition:none}}
"""

_JS = r"""
(function(){
const D=JSON.parse(document.getElementById('bcdata').textContent);
const F=D.frames, K=D.kinds, R=D.reasons, SVGNS='http://www.w3.org/2000/svg';
const isLive = Boolean(D.mode && D.mode.indexOf('LIVE') !== -1);
const $=id=>document.getElementById(id);
const VS=['GREEN','YELLOW','RED'], VL=['GO','CAUTION','STOP'], VC=['#38d889','#ffd23f','#ff5d6c'], VT=['#1f9d63','#c98a00','#d6384a'];
const PS=["DON'T WALK",'WALK','CLEARING'], PL=['WAIT','CROSS','FINISH CROSSING'], PC=['#ff5d6c','#38d889','#ffb020'], PG=['✋','🚶','⏳'];
const DEC=['VEHICLE PRIORITY','PEDESTRIAN PRIORITY','FIXED-TIME FALLBACK'];
const PH=['Vehicle Green','Yellow','All Red','Pedestrian Walk','Clearance'];
const COL=['#ffb3c7','#aedcf7','#ffe08a','#c4aeff','#b9e8d0','#ffc9a8','#d6c9ff','#9fe0d6'];
const PCOL=['#ff8fb8','#8fc7ff','#ffcf6e','#b69bff','#7fdcb2','#ffa98f'];
const E=(n,a,p)=>{const e=document.createElementNS(SVGNS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;};
const vLayer=$('vehicles'), pLayer=$('peds'), vEls={}, pEls={};
// phase chips
const strip=$('phases');PH.forEach((n,i)=>{if(i){const a=document.createElement('span');a.className='arr';a.textContent='\u2192';strip.appendChild(a);}
 const c=document.createElement('span');c.className='chip';c.id='chip'+i;c.textContent=n;strip.appendChild(c);});

function buildVehicle(id,kind){
 const g=E('g',{}),k=K[kind],L=k.len,W=k.w,col=COL[id%COL.length];
 const dark='rgba(60,40,110,.45)';
 if(k.name==='truck'){const cab=L*.32;
  E('rect',{x:-L/2,y:-W/2,width:cab,height:W,rx:5,fill:col,stroke:dark,'stroke-width':1.2},g);
  E('rect',{x:-L/2+cab+2,y:-W/2,width:L-cab-2,height:W,rx:4,fill:'#f4f1fb',stroke:dark,'stroke-width':1.2},g);
  E('rect',{x:-L/2+cab*.45,y:-W/2+4,width:cab*.4,height:W-8,rx:2,fill:'#5a6a8c',opacity:.75},g);
 }else if(k.name==='motorcycle'){
  E('rect',{x:-L/2,y:-2.5,width:L,height:5,rx:2.5,fill:'#6d5aa8'},g);
  E('rect',{x:L*.18,y:-W/2,width:3,height:W,rx:1.5,fill:'#3d3566'},g);
  E('circle',{cx:-1,cy:0,r:4.4,fill:col,stroke:dark,'stroke-width':1},g);
  E('circle',{cx:3,cy:0,r:3,fill:'#ffe0c2'},g);
 }else{
  E('rect',{x:-L/2,y:-W/2,width:L,height:W,rx:Math.min(9,W*.38),fill:col,stroke:dark,'stroke-width':1.2},g);
  if(k.name==='bus'){for(let i=0;i<6;i++)E('rect',{x:-L/2+8+i*(L-30)/6,y:-W/2+4,width:(L-30)/6-4,height:W-8,rx:2,fill:'#5a6a8c',opacity:.6},g);
   E('rect',{x:L/2-16,y:-W/2+3,width:10,height:W-6,rx:3,fill:'#5a6a8c',opacity:.8},g);}
  else{E('rect',{x:L*.1,y:-W/2+3,width:L*.24,height:W-6,rx:3,fill:'#5a6a8c',opacity:.8},g);
   E('rect',{x:-L*.38,y:-W/2+4,width:L*.16,height:W-8,rx:3,fill:'#5a6a8c',opacity:.6},g);
   E('rect',{x:-L*.14,y:-W/2+2,width:L*.22,height:W-4,rx:3,fill:'#ffffff',opacity:.32},g);}
 }
 E('rect',{x:L/2-3,y:-W/2+2,width:3,height:4,fill:'#fff6b8'},g);E('rect',{x:L/2-3,y:W/2-6,width:3,height:4,fill:'#fff6b8'},g);
 const t1=E('rect',{x:-L/2,y:-W/2+2,width:3,height:4,fill:'#ff3b4e',opacity:.35},g),t2=E('rect',{x:-L/2,y:W/2-6,width:3,height:4,fill:'#ff3b4e',opacity:.35},g);
 g._t=[t1,t2];return g;}
function buildPed(id){const g=E('g',{}),c=PCOL[id%PCOL.length];
 E('ellipse',{cx:0,cy:0,rx:5,ry:11,fill:c,stroke:'rgba(60,40,110,.45)','stroke-width':1},g);
 E('circle',{cx:1,cy:0,r:5.6,fill:'#ffe0c2',stroke:'rgba(60,40,110,.4)','stroke-width':.8},g);return g;}

function setSignals(f){
 for(let i=0;i<4;i++){[['r',2],['y',1],['g',0]].forEach(([k,si])=>{const el=$('vl'+i+k),on=(si===f.vs);
   el.setAttribute('fill',on?el.dataset.on:'#575079');el.setAttribute('opacity',on?1:.5);
   el.setAttribute('filter',on?'url(#lens)':'');});
  $('pl'+i).setAttribute('fill',PC[f.ps]);$('pt'+i).textContent=PG[f.ps];}
}
function setPanel(f){
 // Everything shown here comes from the frame object itself (authoritative controller state of
 // that frame). Nothing is extrapolated from the wall clock.
 const num=(v,d)=>(v===undefined||v===null?d:v);
 $('dec').textContent=DEC[f.dc];$('why').textContent=R[f.rs]||'';
 $('vsb').textContent=VS[f.vs];$('vsl').textContent=VL[f.vs];$('vdot').style.background=VC[f.vs];$('vsb').style.color=VT[f.vs];
 $('psb').textContent=PS[f.ps];$('psl').textContent=PL[f.ps];$('pdot').style.background=PC[f.ps];
 $('np').textContent=num(f.np,0);$('nc').textContent=num(f.nc,0);
 $('na').textContent=num(f.na,num(f.nv,0));$('nq').textContent=num(f.nq,0);
 $('lw').textContent=(f.lw===undefined?'':'Longest pedestrian wait '+f.lw.toFixed(1)+' s');
 for(let i=0;i<5;i++)$('chip'+i).classList.toggle('on',i===f.ph);
 let t=PH[f.ph]+' \u00b7 '+(f.ph===0?'min. green ':'')+f.rm.toFixed(1)+'s left';
 if(isLive&&f.frame_index!==undefined&&f.frame_index>=0)t+=' \u00b7 video frame '+f.frame_index+(f.vt!==undefined?' ('+f.vt.toFixed(2)+'s)':'');
 $('tm').textContent=t+(f.fs?' \u00b7 fail-safe':'');
}
function draw(tau){
 let i=0;while(i<F.length-2&&F[i+1].t<=tau)i++;
 const a=F[i],b=F[Math.min(i+1,F.length-1)],al=b.t>a.t?Math.min(1,Math.max(0,(tau-a.t)/(b.t-a.t))):0;
 const bv={},bp={};b.v.forEach(v=>bv[v[0]]=v);b.p.forEach(p=>bp[p[0]]=p);
 const seenV={},seenP={};
 a.v.forEach(v=>{let g=vEls[v[0]];if(!g){g=buildVehicle(v[0],v[4]);vLayer.appendChild(g);vEls[v[0]]=g;}
  const n=bv[v[0]]||v,x=v[1]+(n[1]-v[1])*al,y=v[2]+(n[2]-v[2])*al;
  g.setAttribute('transform','translate('+x+','+y+') rotate('+v[3]+')');
  g._t.forEach(t=>t.setAttribute('opacity',v[5]?1:.35));seenV[v[0]]=1;});
 for(const id in vEls)if(!seenV[id]){vEls[id].remove();delete vEls[id];}
 a.p.forEach(p=>{let g=pEls[p[0]];if(!g){g=buildPed(p[0]);pLayer.appendChild(g);pEls[p[0]]=g;}
  const n=bp[p[0]]||p,x=p[1]+(n[1]-p[1])*al,y=p[2]+(n[2]-p[2])*al;
  g.setAttribute('transform','translate('+x+','+y+') rotate('+p[4]+')');seenP[p[0]]=1;});
 for(const id in pEls)if(!seenP[id]){pEls[id].remove();delete pEls[id];}
 // signals + panel follow the frame the playhead is at (at the end of the clip: the last frame)
 const ci=(al>=1?Math.min(i+1,F.length-1):i);
 if(draw.last!==ci){draw.last=ci;setSignals(F[ci]);setPanel(F[ci]);}
}
const T=F[F.length-1].t;
// Live mode opens on the LATEST processed frame and holds it; "Replay history" plays the recorded
// frames once in video time. Scenario mode loops as before.
let playing=true,t0=performance.now(),off=isLive?T:0;
function tauAt(now){const el=(now-t0)/1000*D.speed+off;return isLive?Math.min(T,el):(T>0?el%T:0);}
function loop(now){if(playing)draw(tauAt(now));requestAnimationFrame(loop);}
if(isLive)$('rs').textContent='Replay history';
$('pp').onclick=()=>{const now=performance.now();
 if(playing){off=(isLive?Math.min(T,(now-t0)/1000*D.speed+off):((now-t0)/1000*D.speed+off)%(T||1));playing=false;}
 else{t0=now;playing=true;}
 $('pp').textContent=playing?'Pause':'Play';};
$('rs').onclick=()=>{off=0;t0=performance.now();playing=true;$('pp').textContent='Pause';draw(0);};
window.__bloom={draw:t=>{playing=false;draw(t);},frames:F.length,duration:T,live:isLive};
draw(isLive?T:0);requestAnimationFrame(loop);
})();
"""


def render_bloom_city_html(frames: Sequence[dict], reasons: Sequence[str], *, theme: str = "day",
                           mode: str = "SCENARIO DEMO", scenario_title: str = "",
                           source_note: str = "Representative agents derived from scenario telemetry",
                           speed: float = 1.0) -> str:
    """Return one self-contained HTML document (scene + decision panel + phase strip)."""
    if not frames:
        raise ValueError("render_bloom_city_html needs at least one frame")
    night = theme == "night"
    data = {"frames": list(frames), "reasons": list(reasons), "speed": speed, "mode": mode,
            "kinds": [{"name": k.value, "len": KIND_SPECS[k][0], "w": KIND_SPECS[k][1]} for k in KIND_ORDER]}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    overlay = ('<rect width="1000" height="700" fill="#1d1450" opacity=".52"/>' if night else "")
    glow = _lamp_glow() if night else ""
    title = f"{mode}" + (f" \u2014 {scenario_title}" if scenario_title else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bloom City</title>
<style>{_CSS}</style></head><body>
<div id="bc" class="{theme}" data-theme="{theme}">
 <div class="main">
  <div class="stage">
   <svg id="scene" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid meet" role="img"
        aria-label="Bloom City intersection with traffic lights, vehicles and pedestrians">
    <defs>
     <radialGradient id="glow"><stop offset="0" stop-color="#fff4b8" stop-opacity=".9"/><stop offset="1" stop-color="#fff4b8" stop-opacity="0"/></radialGradient>
     <filter id="lens" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <g id="static">{_static_scene()}</g>
    <g id="vehicles"></g><g id="peds"></g>
    {overlay}
    <g id="signals">{_signal_layer()}</g>
    {glow}
   </svg>
   <div class="tag">\U0001F338 Bloom City \u00b7 {title}</div>
   <div class="note">{source_note}</div>
   <div class="ctl"><button id="pp" type="button">Pause</button><button id="rs" type="button">Restart</button></div>
  </div>
  <aside class="panel" aria-live="polite">
   <div class="lbl">Current decision</div>
   <div id="dec">&nbsp;</div><p id="why"></p>
   <div class="sigs">
    <div class="sig"><div class="k">Vehicle signal</div><b><i class="dot" id="vdot"></i><span id="vsb"></span></b><span class="a" id="vsl"></span></div>
    <div class="sig"><div class="k">Pedestrian signal</div><b><i class="dot" id="pdot"></i><span id="psb"></span></b><span class="a" id="psl"></span></div>
   </div>
   <div class="cnt"><div><b id="np">0</b><span>pedestrians waiting</span></div><div><b id="nc">0</b><span>in crosswalk</span></div>
    <div><b id="na">0</b><span>vehicles approaching</span></div><div><b id="nq">0</b><span>vehicles queued</span></div></div>
   <div id="lw" class="lw"></div>
   <div id="tm"></div>
  </aside>
 </div>
 <div class="phases" id="phases" aria-label="Controller phase"></div>
</div>
<script id="bcdata" type="application/json">{payload}</script>
<script>{_JS}</script>
</body></html>"""
