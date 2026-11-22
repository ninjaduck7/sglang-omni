"""Exercise a local ASR service; keep quality, failures and deliberate faults separate."""
import argparse
import asyncio
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import signal
import time

import aiohttp
import numpy as np
import psutil
import pyarrow.parquet as pq
import soundfile as sf
from check_features import units

CLIPS = Path("../../runs/sglang-omni/cu128-5090")
EXPECTED = {"en": "The weather is sunny today.", "zh": "今天天气很好。"}
SPLITS = {"en":"5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca",
          "zh":"5d73be79cf2e84fd0bebf5d47212438a88dd264a591031d092164ba84545f1de"}


async def main(a):
    out = a.out
    url = f"http://127.0.0.1:{a.port}"
    clips = {l:(CLIPS/f"vivian-{l}.wav").read_bytes() for l in EXPECTED}
    identity = json.loads((out/"soak-ready.json").read_text())
    def verify_pid():
        proc = psutil.Process(identity["service_pid"])
        if proc.create_time() != identity["service_birth"] or "sgl-omni" not in " ".join(proc.cmdline()):
            raise RuntimeError("Service PID changed")
        return proc
    verify_pid()
    trace = (out/f"{a.mode}-events.jsonl").open("x",buffering=1)
    def event(kind, **kw):
        print(json.dumps(dict(unix=time.time(),kind=kind,**kw),ensure_ascii=False),file=trace)
    async def post(session,b,fields=None):
        form = aiohttp.FormData()
        form.add_field("file",b,filename="clip.wav",content_type="audio/wav")
        for k,v in (fields or {}).items():
            form.add_field(k,v)
        t = time.monotonic()
        try:
            async with session.post(url+"/v1/audio/transcriptions",data=form) as r:
                raw = await r.text()
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = raw[:1000]
                return dict(status=r.status,body=body,ctype=r.content_type,latency=time.monotonic()-t)
        except (aiohttp.ClientError,TimeoutError) as exc:
            return dict(status=0,error=type(exc).__name__,latency=time.monotonic()-t)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as s:
        async def health():
            async with s.get(url+"/health") as r:
                return await r.json()
        async def control(tag,l="en",fresh=False):
            if fresh:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as new:
                    x=await post(new,clips[l],{"language":l})
            else:
                x=await post(s,clips[l],{"language":l})
            ok=x["status"]==200 and units(x["body"].get("text",""),l)==units(EXPECTED[l],l)
            event("control",tag=tag,lang=l,correct=ok,**x)
            event("health",tag=tag,body=await health())
        async def isolated(action):
            before=await health()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as solo:
                task=asyncio.create_task(post(solo,clips["en"],{"language":"en"}))
                active=None
                while not task.done():
                    h=await health()
                    if before["pending_completions"]==0 and h["pending_completions"]==1:
                        active=h
                        break
                    await asyncio.sleep(.003)
                event("active",action=action,before=before,observed=active,open_task=not task.done())
                if active and not task.done():
                    if action=="disconnect":
                        task.cancel()
                        await asyncio.gather(task,return_exceptions=True)
                        await solo.close()
                        event("disconnect",after=await health())
                    else:
                        proc=verify_pid()
                        os.kill(proc.pid,signal.SIGTERM)
                        event("sigterm",pid=proc.pid,response=await task)
                else:
                    event("active_unobserved",action=action,response=await task)
            if action=="disconnect":
                await control("reconnect","zh",fresh=True)
                event("drained",state=await health())
        if a.mode == "signal":
            await isolated("sigterm")
            return
        if a.mode == "edges":
            for l in ("en","zh"):
                x=await post(s,clips[l],{"language":l,"stream":"true"})
                event("stream_true",lang=l,**x)
            # A repeated prefix prevents an all-silence window. Same exact first 30 s
            # in the above-boundary sentinel and its prefix-only control.
            from scipy.signal import resample_poly
            from math import gcd
            def pcm(raw):
                samples,rate=sf.read(io.BytesIO(raw),dtype="float32")
                d=gcd(rate,16000)
                return resample_poly(samples,16000//d,rate//d)
            prefix=pcm(clips["en"])
            tail=pcm((CLIPS/"feature-fr.wav").read_bytes())
            for name,duration,pos in [("prefix",30,None),("within",30,29.9-len(tail)/16000),
                                      ("after",35,30.3),("overflow_audio",150,145.)]:
                y=np.zeros(16000*duration,dtype="float32")
                for start in range(0,26,4):
                    y[start*16000:start*16000+len(prefix)]=prefix
                if pos is not None:
                    i=round(16000*pos)
                    y[i:i+len(tail)]=tail
                b=io.BytesIO()
                sf.write(b,y,16000,format="WAV",subtype="PCM_16")
                x=await post(s,b.getvalue(),{"response_format":"verbose_json"})
                event("boundary",name=name,duration=duration,tail_start=pos,**x)
            # The multipart prompt is accepted by the API; verify whether the
            # backend actually consumes it before interpreting this as overflow.
            x=await post(s,clips["en"],{"prompt":" context"*5000,"language":"en"})
            event("large_prompt",repetitions=5000,**x)
            await control("post_boundary",fresh=True)
            return
        bank=[]
        for l,h in SPLITS.items():
            p=out/"private"/f"{l}.parquet"
            with p.open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest()==h
            clips_batch=next(pq.ParquetFile(p).iter_batches(batch_size=32,columns=["sample_id","ref_audio"]))
            bank.append([(l,row["sample_id"],row["ref_audio"]["bytes"]) for row in clips_batch.to_pylist()])
        mixed=list(itertools.chain.from_iterable(zip(*bank)))
        idx=itertools.count()
        async def faults(c):
            await asyncio.sleep(12)
            for name,b,fields in [("empty",b"",{}),("corrupt",b"This is not an audio recording.",{}),
                    ("language",clips["en"],{"language":"invalid-language"}),
                    ("format",clips["en"],{"response_format":"unknown"})]:
                x=await post(s,b,fields)
                event("bad_input",c=c,case=name,**x)
                await control(f"after_{name}_c{c}","zh" if c%2==0 else "en",fresh=True)
        for c in (1,2,4,8):
            verify_pid()
            t0=time.monotonic()
            event("cell_start",c=c)
            with gzip.open(out/f"mixed-c{c}.jsonl.gz","xt",encoding="utf-8") as f:
                async def worker():
                    while time.monotonic()-t0 < a.seconds:
                        l,ref,b=mixed[next(idx)%len(mixed)]
                        x=await post(s,b,{"language":l})
                        print(json.dumps(dict(lang=l,id=ref,**x),ensure_ascii=False),file=f)
                        if x["status"]==0:
                            await asyncio.sleep(.1)
                await asyncio.gather(*[worker() for _ in range(c)],faults(c))
            event("cell_end",c=c,elapsed=time.monotonic()-t0,health=await health())
            if c in (1,4):
                await isolated("disconnect")
        event("soak_finished",pid=verify_pid().pid)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("mode",choices=("mixed","edges","signal"))
    p.add_argument("--port",type=int,default=18593)
    p.add_argument("--seconds",type=int,default=450)
    asyncio.run(main(p.parse_args()))
