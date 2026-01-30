let session,video,canvas,ctx,running=false,mode='video';
let capturedCanvas,capturedCtx,capturedImage,lastDetections=[];
const W=640,H=640,CONF=0.5,IOU=0.45;
const COLORS={Small:'#ffeb3b',Medium:'#4caf50',Large:'#2196f3','X-Large':'#f44336'};

async function start(){
    document.getElementById('startBtn').style.display='none';
    document.getElementById('uploadBtn').style.display='none';
    document.getElementById('loading').style.display='block';
    try{
        session=await ort.InferenceSession.create('./model/best.onnx');
        video=document.getElementById('video');
        canvas=document.getElementById('canvas');
        ctx=canvas.getContext('2d');
        capturedImage=document.getElementById('capturedImage');
        capturedCanvas=document.getElementById('capturedCanvas');
        capturedCtx=capturedCanvas.getContext('2d');
        const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:640},height:{ideal:480}}});
        video.srcObject=stream;
        video.onloadedmetadata=()=>{
            canvas.width=video.videoWidth;
            canvas.height=video.videoHeight;
            document.getElementById('loading').style.display='none';
            document.getElementById('videoContainer').style.display='block';
            document.getElementById('stats').style.display='block';
            document.getElementById('controls').style.display='flex';
            document.getElementById('modeIndicator').style.display='block';
            running=true;
            mode='video';
            updateModeUI();
            detect();
        };
    }catch(e){
        alert('Error: '+e.message);
        document.getElementById('startBtn').style.display='block';
        document.getElementById('uploadBtn').style.display='block';
        document.getElementById('loading').style.display='none';
    }
}

async function detect(){
    if(!running||mode!=='video')return;
    const t0=performance.now();
    try{
        const input=preprocess(video);
        const out=await session.run({images:input});
        const dets=postprocess(out,video.videoWidth,video.videoHeight);
        draw(ctx,dets,video.videoWidth,video.videoHeight);
        updateUI(dets,performance.now()-t0);
    }catch(e){console.error(e)}
    requestAnimationFrame(detect);
}

function preprocess(source){
    const c=document.createElement('canvas');
    c.width=W;c.height=H;
    const cx=c.getContext('2d');
    cx.drawImage(source,0,0,W,H);
    const d=cx.getImageData(0,0,W,H).data;
    const f=new Float32Array(3*W*H);
    for(let i=0;i<W*H;i++){
        f[i]=d[i*4]/255;
        f[i+W*H]=d[i*4+1]/255;
        f[i+2*W*H]=d[i*4+2]/255;
    }
    return new ort.Tensor('float32',f,[1,3,H,W]);
}

function postprocess(out,srcW,srcH){
    const o0=out['output0'];
    const data=o0.data;
    const [,ch,nb]=o0.dims;
    let dets=[];
    for(let i=0;i<nb;i++){
        const conf=data[4*nb+i];
        if(conf>CONF){
            const x=data[i],y=data[nb+i],w=data[2*nb+i],h=data[3*nb+i];
            const sx=srcW/W,sy=srcH/H;
            dets.push({x1:Math.max(0,(x-w/2)*sx),y1:Math.max(0,(y-h/2)*sy),x2:Math.min(srcW,(x+w/2)*sx),y2:Math.min(srcH,(y+h/2)*sy),conf});
        }
    }
    dets=nms(dets);
    dets=dets.map((d,i)=>{
        const bw=d.x2-d.x1,bh=d.y2-d.y1,ar=bh/bw;
        let mm;
        if(ar<1.25)mm=45+(ar-1.1)*30;
        else if(ar<1.35)mm=50+(ar-1.25)*50;
        else if(ar<1.45)mm=55+(ar-1.35)*70;
        else mm=62+(ar-1.45)*80;
        mm=Math.max(40,Math.min(75,mm));
        mm=Math.round(mm*10)/10;
        let sz;
        if(mm<48)sz='Small';
        else if(mm<55)sz='Medium';
        else if(mm<62)sz='Large';
        else sz='X-Large';
        return {...d,mm,size:sz,area:Math.round(Math.PI*(bw/2)*(bh/2))};
    });
    dets.sort((a,b)=>b.mm-a.mm);
    dets.forEach((d,i)=>{d.rank=i+1});
    if(dets.length>0){
        const ref=dets[0].mm;
        dets.forEach((d,i)=>{d.diff=i===0?0:Math.round((d.mm-ref)*10)/10});
    }
    return dets;
}

function nms(dets){
    dets.sort((a,b)=>b.conf-a.conf);
    const sel=[];
    while(dets.length>0){
        const best=dets.shift();
        sel.push(best);
        dets=dets.filter(d=>iou(best,d)<IOU);
    }
    return sel;
}

function iou(a,b){
    const x1=Math.max(a.x1,b.x1),y1=Math.max(a.y1,b.y1),x2=Math.min(a.x2,b.x2),y2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
    const areaA=(a.x2-a.x1)*(a.y2-a.y1),areaB=(b.x2-b.x1)*(b.y2-b.y1);
    return inter/(areaA+areaB-inter);
}

function draw(context,dets,w,h){
    context.clearRect(0,0,w,h);
    dets.forEach(d=>{
        const c=COLORS[d.size],bw=d.x2-d.x1,bh=d.y2-d.y1;
        context.strokeStyle=c;
        context.lineWidth=3;
        context.beginPath();
        context.ellipse(d.x1+bw/2,d.y1+bh/2,bw/2,bh/2,0,0,Math.PI*2);
        context.stroke();
        context.fillStyle=c+'40';
        context.fill();
        context.fillStyle=c;
        context.fillRect(d.x1,d.y1-38,85,35);
        context.fillStyle='#000';
        context.font='bold 13px sans-serif';
        context.fillText(`${d.mm}mm`,d.x1+5,d.y1-22);
        context.font='11px sans-serif';
        context.fillText(d.size,d.x1+5,d.y1-8);
        context.beginPath();
        context.arc(d.x1+bw/2,d.y1+bh/2,16,0,Math.PI*2);
        context.fillStyle=c;
        context.fill();
        context.fillStyle='#000';
        context.font='bold 13px sans-serif';
        context.textAlign='center';
        context.fillText('#'+d.rank,d.x1+bw/2,d.y1+bh/2+4);
        context.textAlign='left';
    });
}

function updateUI(dets,ms){
    lastDetections=dets;
    document.getElementById('total').textContent=dets.length;
    document.getElementById('ms').textContent=Math.round(ms);
    const cnt={Small:0,Medium:0,Large:0,'X-Large':0};
    dets.forEach(d=>cnt[d.size]++);
    document.getElementById('s').textContent=cnt.Small;
    document.getElementById('m').textContent=cnt.Medium;
    document.getElementById('l').textContent=cnt.Large;
    document.getElementById('xl').textContent=cnt['X-Large'];
    if(dets.length>0){
        const sizes=dets.map(d=>d.mm);
        const min=Math.min(...sizes),max=Math.max(...sizes);
        const avg=Math.round(sizes.reduce((a,b)=>a+b,0)/sizes.length*10)/10;
        document.getElementById('min').textContent=min+'mm';
        document.getElementById('max').textContent=max+'mm';
        document.getElementById('avg').textContent=avg+'mm';
        document.getElementById('rng').textContent=(max-min).toFixed(1)+'mm';
    }else{
        ['min','max','avg','rng'].forEach(id=>document.getElementById(id).textContent='--');
    }
    let html='';
    dets.forEach(d=>{
        const cls=d.size.toLowerCase().replace('-','');
        const diff=d.diff===0?'Largest':(d.diff>0?'+':'')+d.diff+'mm';
        html+=`<div class="item ${cls}"><span><b>#${d.rank}</b> ${d.mm}mm (${d.size})</span><span>${diff}</span></div>`;
    });
    document.getElementById('list').innerHTML=html||'<p style="text-align:center;color:#888;padding:20px">Point camera at eggs</p>';
}

async function capturePhoto(){
    mode='photo';
    running=false;
    updateModeUI();
    const tempCanvas=document.createElement('canvas');
    tempCanvas.width=video.videoWidth;
    tempCanvas.height=video.videoHeight;
    tempCanvas.getContext('2d').drawImage(video,0,0);
    capturedImage.src=tempCanvas.toDataURL('image/jpeg',0.95);
    capturedImage.onload=async()=>{
        capturedCanvas.width=capturedImage.naturalWidth;
        capturedCanvas.height=capturedImage.naturalHeight;
        document.getElementById('videoContainer').style.display='none';
        document.getElementById('capturedContainer').style.display='block';
        const t0=performance.now();
        const input=preprocess(capturedImage);
        const out=await session.run({images:input});
        const dets=postprocess(out,capturedImage.naturalWidth,capturedImage.naturalHeight);
        draw(capturedCtx,dets,capturedImage.naturalWidth,capturedImage.naturalHeight);
        updateUI(dets,performance.now()-t0);
    };
}

function backToVideo(){
    mode='video';
    running=true;
    updateModeUI();
    document.getElementById('capturedContainer').style.display='none';
    document.getElementById('videoContainer').style.display='block';
    detect();
}

function switchMode(){
    if(mode==='video')capturePhoto();
    else backToVideo();
}

function updateModeUI(){
    const indicator=document.getElementById('modeIndicator');
    const captureBtn=document.getElementById('captureBtn');
    const saveBtn=document.getElementById('saveBtn');
    const backBtn=document.getElementById('backBtn');
    const switchBtn=document.getElementById('switchBtn');
    if(mode==='video'){
        indicator.textContent='🎥 Live Video Mode';
        indicator.className='mode-indicator mode-video';
        captureBtn.style.display='flex';
        saveBtn.style.display='none';
        backBtn.style.display='none';
        switchBtn.textContent='📸 Photo Mode';
    }else{
        indicator.textContent='📷 Photo Mode';
        indicator.className='mode-indicator mode-photo';
        captureBtn.style.display='none';
        saveBtn.style.display='flex';
        backBtn.style.display='flex';
        switchBtn.textContent='🎥 Video Mode';
    }
}

function saveImage(){
    const saveCanvas=document.createElement('canvas');
    saveCanvas.width=capturedImage.naturalWidth;
    saveCanvas.height=capturedImage.naturalHeight;
    const saveCtx=saveCanvas.getContext('2d');
    saveCtx.drawImage(capturedImage,0,0);
    draw(saveCtx,lastDetections,capturedImage.naturalWidth,capturedImage.naturalHeight);
    const link=document.createElement('a');
    link.download='egg_detection_'+Date.now()+'.jpg';
    link.href=saveCanvas.toDataURL('image/jpeg',0.95);
    link.click();
}

async function handleUpload(event){
    const file=event.target.files[0];
    if(!file)return;
    document.getElementById('startBtn').style.display='none';
    document.getElementById('uploadBtn').style.display='none';
    document.getElementById('loading').style.display='block';
    try{
        if(!session)session=await ort.InferenceSession.create('./model/best.onnx');
        capturedImage=document.getElementById('capturedImage');
        capturedCanvas=document.getElementById('capturedCanvas');
        capturedCtx=capturedCanvas.getContext('2d');
        const reader=new FileReader();
        reader.onload=async(e)=>{
            capturedImage.src=e.target.result;
            capturedImage.onload=async()=>{
                capturedCanvas.width=capturedImage.naturalWidth;
                capturedCanvas.height=capturedImage.naturalHeight;
                document.getElementById('loading').style.display='none';
                document.getElementById('capturedContainer').style.display='block';
                document.getElementById('stats').style.display='block';
                document.getElementById('controls').style.display='flex';
                document.getElementById('modeIndicator').style.display='block';
                mode='photo';
                updateModeUI();
                document.getElementById('backBtn').style.display='none';
                document.getElementById('switchBtn').style.display='none';
                const t0=performance.now();
                const input=preprocess(capturedImage);
                const out=await session.run({images:input});
                const dets=postprocess(out,capturedImage.naturalWidth,capturedImage.naturalHeight);
                draw(capturedCtx,dets,capturedImage.naturalWidth,capturedImage.naturalHeight);
                updateUI(dets,performance.now()-t0);
            };
        };
        reader.readAsDataURL(file);
    }catch(e){
        alert('Error: '+e.message);
        document.getElementById('startBtn').style.display='block';
        document.getElementById('uploadBtn').style.display='block';
        document.getElementById('loading').style.display='none';
    }
}

window.onload=()=>{document.getElementById('uploadBtn').style.display='block';};