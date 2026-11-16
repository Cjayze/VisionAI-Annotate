import { useMemo, useRef, useState } from "react";
import { Upload, ScanSearch, Download, Image as ImageIcon, Loader2, RefreshCw, CheckCircle2, AlertCircle, Boxes, Square, Pentagon, Trash2, Check, BookOpen } from "lucide-react";

const API = "http://localhost:8000";
const palette = ["#55d6be", "#ffbd69", "#8da7ff", "#f78fb3", "#c3a6ff", "#a3e635"];
const uid = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

export default function App() {
  const inputRef = useRef(null), svgRef = useRef(null), drawingRef = useRef(null);
  const [file, setFile] = useState(null), [preview, setPreview] = useState("");
  const [result, setResult] = useState(null), [annotations, setAnnotations] = useState([]);
  const [confidence, setConfidence] = useState(25), [loading, setLoading] = useState(false), [error, setError] = useState("");
  const [showBoxes, setShowBoxes] = useState(true), [tool, setTool] = useState("select"), [selected, setSelected] = useState(null);
  const [drawPreview, setDrawPreview] = useState(null);
  const [label, setLabel] = useState("person"), [format, setFormat] = useState("YOLO");
  const image = result?.image;
  const detections = annotations;
  const setItem = (id, patch) => setAnnotations((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item));

  const chooseFile = (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith("image/")) { setError("Please select an image file."); return; }
    if (selectedFile.size > 10 * 1024 * 1024) { setError("Image must be 10 MB or smaller."); return; }
    if (preview) URL.revokeObjectURL(preview);
    setFile(selectedFile); setPreview(URL.createObjectURL(selectedFile)); setResult(null); setAnnotations([]); setError("");
  };

  const runDetection = async () => {
    if (!file) return;
    setLoading(true); setError("");
    const body = new FormData(); body.append("file", file); body.append("conf", String(confidence / 100));
    try {
      const response = await fetch(`${API}/api/detect`, { method: "POST", body });
      const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Detection request failed.");
      setResult(data); setAnnotations(data.detections.map((d) => ({ ...d, box: { ...d.box }, polygon: null, source: "AI", status: "needs_review" })));
      setSelected(null);
    } catch (e) { setError(`${e.message} Make sure the FastAPI backend is running at ${API}.`); }
    finally { setLoading(false); }
  };

  const coords = (event) => {
    const point = svgRef.current.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    const p = point.matrixTransform(svgRef.current.getScreenCTM().inverse());
    return { x: Math.max(0, Math.min(image.width, p.x)), y: Math.max(0, Math.min(image.height, p.y)) };
  };
  const startDraw = (event) => {
    if (tool === "select" || !image) return;
    event.preventDefault(); const p = coords(event);
    if (tool === "box") drawingRef.current = { kind: "box", start: p };
    if (tool === "polygon") drawingRef.current = { kind: "polygon", points: [...(drawingRef.current?.kind === "polygon" ? drawingRef.current.points : []), p] };
    setDrawPreview({ ...drawingRef.current });
    setSelected(null);
  };
  const moveDraw = (event) => { if (drawingRef.current?.kind === "box") { drawingRef.current = { ...drawingRef.current, current: coords(event) }; setDrawPreview({ ...drawingRef.current }); } };
  const finishDraw = (event) => {
    if (drawingRef.current?.kind !== "box") return;
    const { start } = drawingRef.current, end = coords(event); drawingRef.current = null; setDrawPreview(null);
    if (Math.abs(start.x-end.x) < 3 || Math.abs(start.y-end.y) < 3) return;
    const item = { id: uid(), label, confidence: null, source: "human", status: "pending", polygon: null, box: { x1: Math.min(start.x,end.x), y1: Math.min(start.y,end.y), x2: Math.max(start.x,end.x), y2: Math.max(start.y,end.y) } };
    setAnnotations((items) => [...items, item]); setSelected(item.id);
  };
  const finishPolygon = () => {
    const drawing = drawingRef.current; if (drawing?.kind !== "polygon" || drawing.points.length < 3) return;
    drawingRef.current = null; setDrawPreview(null);
    const xs = drawing.points.map((p) => p.x), ys = drawing.points.map((p) => p.y);
    const item = { id: uid(), label, confidence: null, source: "human", status: "pending", polygon: drawing.points, box: { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) } };
    setAnnotations((items) => [...items, item]); setSelected(item.id);
  };
  const confirmAll = () => setAnnotations((items) => items.map((item) => ({ ...item, status: "confirmed" })));

  const exportData = () => {
    if (!image || !file) return;
    const labels = [...new Set(detections.map((d) => d.label))].sort(), ids = Object.fromEntries(labels.map((name, i) => [name, i]));
    let content, type, extension;
    if (format === "YOLO") {
      content = detections.map((d) => { const b=d.box; return `${ids[d.label]} ${((b.x1+b.x2)/2/image.width).toFixed(6)} ${((b.y1+b.y2)/2/image.height).toFixed(6)} ${((b.x2-b.x1)/image.width).toFixed(6)} ${((b.y2-b.y1)/image.height).toFixed(6)}`; }).join("\n"); type="text/plain"; extension="txt";
    } else if (format === "COCO") {
      content = JSON.stringify({ info:{description:"VisionAI annotated dataset"}, images:[{id:1,file_name:file.name,width:image.width,height:image.height}], categories:labels.map((name,i)=>({id:i+1,name,supercategory:"object"})), annotations:detections.map((d,i)=>({id:i+1,image_id:1,category_id:ids[d.label]+1,bbox:[d.box.x1,d.box.y1,d.box.x2-d.box.x1,d.box.y2-d.box.y1],area:(d.box.x2-d.box.x1)*(d.box.y2-d.box.y1),iscrowd:0, ...(d.polygon ? {segmentation:[d.polygon.flatMap(p=>[p.x,p.y])]} : {})})) },null,2); type="application/json"; extension="json";
    } else {
      const escapeXml = (s) => s.replace(/[<>&"']/g, (c) => ({"<":"&lt;",">":"&gt;","&":"&amp;",'"':"&quot;","'":"&apos;"}[c]));
      content = `<?xml version="1.0"?><annotation><filename>${escapeXml(file.name)}</filename><size><width>${image.width}</width><height>${image.height}</height><depth>3</depth></size>${detections.map(d=>`<object><name>${escapeXml(d.label)}</name><bndbox><xmin>${Math.round(d.box.x1)}</xmin><ymin>${Math.round(d.box.y1)}</ymin><xmax>${Math.round(d.box.x2)}</xmax><ymax>${Math.round(d.box.y2)}</ymax></bndbox></object>`).join("")}</annotation>`; type="application/xml"; extension="xml";
    }
    const blob = new Blob([content], { type }), url = URL.createObjectURL(blob), a = document.createElement("a"); a.href=url; a.download=`${file.name.replace(/\.[^.]+$/, "")}.${extension}`; a.click(); URL.revokeObjectURL(url);
  };

  const summary = useMemo(() => { const counts = {}; detections.forEach((d) => counts[d.label] = (counts[d.label] || 0) + 1); return Object.entries(counts).sort((a,b)=>b[1]-a[1]); }, [detections]);
  const selectedItem = detections.find((d) => d.id === selected);
  const draftBox = drawPreview?.kind === "box" && drawPreview.current ? { x1:Math.min(drawPreview.start.x,drawPreview.current.x), y1:Math.min(drawPreview.start.y,drawPreview.current.y), x2:Math.max(drawPreview.start.x,drawPreview.current.x), y2:Math.max(drawPreview.start.y,drawPreview.current.y) } : null;

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-mark"><ScanSearch size={20}/></span><span>Vision<span className="brand-accent">AI</span></span></a><div className="top-meta"><span className="status-dot"/> ANNOTATION WORKSPACE <span className="version">v2.0</span></div></header>
    <main className="main">
      <section className="hero"><div className="eyebrow"><span/> COMPUTER VISION PLATFORM</div><h1>See objects.<br/><span>Build better datasets.</span></h1><p className="hero-copy">Run AI detection, refine annotations by hand, review quality, and export training data in standard formats.</p></section>
      <div className="workspace">
        <section className="panel image-panel">
          <div className="panel-heading"><div><div className="section-kicker">01 / ANNOTATE</div><h2>Image workspace</h2></div>{preview && <button className="ghost-btn" onClick={()=>{if(preview)URL.revokeObjectURL(preview);setFile(null);setPreview("");setResult(null);setAnnotations([]);setError("");}}><RefreshCw size={14}/> Reset</button>}</div>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e)=>chooseFile(e.target.files?.[0])}/>
          {!preview ? <button className="dropzone" onClick={()=>inputRef.current?.click()} onDragOver={(e)=>e.preventDefault()} onDrop={(e)=>{e.preventDefault();chooseFile(e.dataTransfer.files?.[0]);}}><span className="upload-icon"><Upload size={23}/></span><strong>Drop an image here</strong><span>or <u>browse files</u></span><small>JPG, PNG, WEBP · Maximum 10 MB</small></button> : <div className="preview-wrap">
            <img src={preview} alt="Uploaded preview" onLoad={(e)=>{if(!result)setResult({image:{width:e.currentTarget.naturalWidth,height:e.currentTarget.naturalHeight},detections:[],count:0});}}/>
            {image && <svg ref={svgRef} className={`annotation-layer tool-${tool}`} viewBox={`0 0 ${image.width} ${image.height}`} preserveAspectRatio="xMidYMid meet" onPointerDown={startDraw} onPointerMove={moveDraw} onPointerUp={finishDraw} onDoubleClick={finishPolygon}>
              {showBoxes && detections.map((d,i)=>{const b=d.box,c=palette[i%palette.length],polygon=d.polygon?.map(p=>`${p.x},${p.y}`).join(" ");return <g key={d.id} onClick={(e)=>{e.stopPropagation();setSelected(d.id);}} style={{cursor:"pointer"}}>
                {polygon ? <polygon points={polygon} fill={`${c}20`} stroke={c} strokeWidth={Math.max(2,image.width/500)} vectorEffect="non-scaling-stroke"/> : <rect x={b.x1} y={b.y1} width={b.x2-b.x1} height={b.y2-b.y1} fill={selected===d.id?`${c}20`:"transparent"} stroke={c} strokeWidth={selected===d.id?3:2} vectorEffect="non-scaling-stroke"/>}
                <text x={b.x1} y={Math.max(15,b.y1-6)} fill={c} fontSize={Math.max(14,image.width/65)} fontWeight="700" paintOrder="stroke" stroke="#07111b" strokeWidth="3">{d.label}{d.confidence!=null?` ${Math.round(d.confidence*100)}%`:""} · {d.source === "AI" && d.status === "needs_review" ? "REVIEW" : d.status === "confirmed" ? "✓" : "EDIT"}</text>
              </g>;})}
              {draftBox && <rect x={draftBox.x1} y={draftBox.y1} width={draftBox.x2-draftBox.x1} height={draftBox.y2-draftBox.y1} fill="#55d6be20" stroke="#55d6be" strokeWidth="2" vectorEffect="non-scaling-stroke"/>}
              {drawPreview?.kind === "polygon" && <><polyline points={drawPreview.points.map(p=>`${p.x},${p.y}`).join(" ")} fill="none" stroke="#55d6be" strokeWidth="2" vectorEffect="non-scaling-stroke"/>{drawPreview.points.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r={4} fill="#55d6be" vectorEffect="non-scaling-stroke"/>)}</>}
            </svg>}
          </div>}
          <div className="annotation-tools"><button className={tool==="select"?"active":""} onClick={()=>{setTool("select");drawingRef.current=null;setDrawPreview(null);}}><ScanSearch size={14}/> Select</button><button className={tool==="box"?"active":""} onClick={()=>{setTool("box");drawingRef.current=null;setDrawPreview(null);}} disabled={!image}><Square size={14}/> Box</button><button className={tool==="polygon"?"active":""} onClick={()=>{setTool("polygon");drawingRef.current=null;setDrawPreview(null);}} disabled={!image}><Pentagon size={14}/> Polygon</button>{tool==="polygon"&&drawPreview?.kind==="polygon"&&<button onClick={finishPolygon}><Check size={14}/> Finish</button>}{selectedItem&&<button onClick={()=>setAnnotations(items=>items.filter(i=>i.id!==selected))}><Trash2 size={14}/> Delete</button>}</div>
          {tool!=="select"&&<div className="draw-hint">{tool==="box"?"Drag on the image to draw a box.":"Click to add polygon points, then double-click or finish."}</div>}
          <div className="image-footer"><div className="file-info"><ImageIcon size={15}/><span>{file?file.name:"No image selected"}</span></div>{preview&&<label className="toggle"><input type="checkbox" checked={showBoxes} onChange={e=>setShowBoxes(e.target.checked)}/> Show annotations</label>}</div>
        </section>
        <aside className="panel control-panel"><div className="panel-heading"><div><div className="section-kicker">02 / REVIEW</div><h2>Labels & quality</h2></div></div>
          <div className="setting-block"><div className="setting-label"><span>New annotation label</span></div><input className="label-input" list="label-options" value={label} onChange={e=>setLabel(e.target.value)} placeholder="Enter class name"/><datalist id="label-options">{summary.map(([name])=><option key={name} value={name}/>)}</datalist></div>
          <div className="setting-block"><div className="setting-label"><span>AI confidence threshold</span><strong>{confidence}%</strong></div><input className="range" type="range" min="5" max="95" step="5" value={confidence} onChange={e=>setConfidence(Number(e.target.value))}/><div className="range-labels"><span>More detections</span><span>Higher certainty</span></div></div>
          <div className="model-card"><div className="model-icon"><Boxes size={19}/></div><div><strong>YOLO11 Nano</strong><span>Pretrained object detection</span></div><span className="model-tag">READY</span></div>
          <button className="primary-btn" disabled={!file||loading} onClick={runDetection}>{loading?<><Loader2 className="spin" size={17}/> Detecting objects...</>:<><ScanSearch size={17}/> Run AI detection</>}</button>
          {result&&<button className="confirm-btn" disabled={!detections.length} onClick={confirmAll}><CheckCircle2 size={16}/> Confirm all {detections.length} labels</button>}
          <div className="qc-card"><div className="qc-title"><BookOpen size={15}/> QC checklist</div><div className="qc-row"><span>Labels need review</span><strong className={detections.some(d=>d.status!=="confirmed")?"qc-pending":"qc-done"}>{detections.filter(d=>d.status!=="confirmed").length}</strong></div><div className="qc-row"><span>Confirmed annotations</span><strong>{detections.filter(d=>d.status==="confirmed").length}</strong></div><div className="qc-note">Check class names, object boundaries, missing objects, and duplicates before export.</div></div>
          {error&&<div className="error"><AlertCircle size={16}/><span>{error}</span></div>}{result&&<div className="success"><CheckCircle2 size={16}/> Image ready · {detections.length} annotations</div>}
        </aside>
      </div>
      {selectedItem&&<section className="panel edit-panel"><div><div className="section-kicker">SELECTED ANNOTATION</div><strong>{selectedItem.source} · {selectedItem.status.replace("_"," ")}</strong></div><label>Class name<input className="label-input" value={selectedItem.label} onChange={e=>setItem(selectedItem.id,{label:e.target.value,status:"pending"})}/></label>{selectedItem.confidence!=null&&<span className="edit-confidence">AI confidence {(selectedItem.confidence*100).toFixed(1)}%</span>}<button className="confirm-one" onClick={()=>setItem(selectedItem.id,{status:"confirmed"})}><Check size={14}/> Confirm label</button></section>}
      <section className="results-section"><div className="results-title"><div><div className="section-kicker">03 / DATASET</div><h2>Annotations & export</h2></div>{result&&<div className="export-controls"><select value={format} onChange={e=>setFormat(e.target.value)}><option>YOLO</option><option>COCO</option><option>Pascal VOC</option></select><button className="export-btn" disabled={!detections.length} onClick={exportData}><Download size={15}/> Export {format}</button></div>}</div>
        {!preview?<div className="empty-results"><div className="empty-icon"><Boxes size={22}/></div><strong>Waiting for an image</strong><span>Upload an image to begin detection and annotation.</span></div>:<><div className="stats-grid"><div className="stat-card"><span>ANNOTATIONS</span><strong>{detections.length}</strong></div><div className="stat-card"><span>CLASSES</span><strong>{summary.length}</strong></div><div className="stat-card"><span>IMAGE DIMENSIONS</span><strong>{image?`${image.width} × ${image.height}`:"—"}</strong></div></div><div className="result-grid"><div className="panel result-card"><h3>Class counts</h3>{summary.length?summary.map(([name,count],i)=><div className="class-row" key={name}><span className="class-dot" style={{background:palette[i%palette.length]}}/><span>{name}</span><strong>{count}</strong></div>):<p className="muted">No annotations yet. Run detection or draw a box.</p>}</div><div className="panel result-card"><h3>Review annotations</h3><div className="table-wrap"><table><thead><tr><th>CLASS</th><th>SOURCE</th><th>REVIEW</th><th>BOX (PX)</th></tr></thead><tbody>{detections.map((d,i)=><tr key={d.id} onClick={()=>setSelected(d.id)} className="annotation-row"><td><span className="object-name" style={{color:palette[i%palette.length]}}>{d.label}</span></td><td>{d.source}</td><td><span className={d.status==="confirmed"?"review-ok":"review-pending"}>{d.status==="confirmed"?"Confirmed":"Needs review"}</span></td><td className="coords">{Math.round(d.box.x1)}, {Math.round(d.box.y1)} → {Math.round(d.box.x2)}, {Math.round(d.box.y2)}</td></tr>)}</tbody></table>{!detections.length&&<p className="muted">No annotations to display.</p>}</div></div></div><p className="disclaimer">AI predictions can be incorrect. Review every annotation before including it in a training dataset. Exports contain the selected image's annotations.</p></>}
      </section><footer>VISION<span>AI</span> <i>·</i> DETECTION & DATA ANNOTATION WORKSPACE</footer>
    </main>
  </div>;
}
