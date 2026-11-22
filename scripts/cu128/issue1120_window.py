"""Verify what the pinned checkpoint's default preprocessor retains on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoFeatureExtractor


def main(out):
    model=out/"private/asr/qwen/huggingface/Qwen3-ASR-1.7B"
    processor=AutoFeatureExtractor.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    rate=processor.sampling_rate
    # Identical first 30 seconds. End-only and within-window changes have the
    # same waveform, so the short control checks sensitivity of this comparison.
    tone=.2*np.sin(2*np.pi*360*np.arange(rate)/rate).astype(np.float32)
    ref=np.zeros(30*rate,np.float32)
    ref[rate:rate*2]=tone
    short=ref.copy()
    short[-rate:]=tone
    rows=[]
    def extract(x):
        return processor(x,sampling_rate=rate,return_tensors="np",return_attention_mask=True)
    base=extract(ref)
    for sec in (30,35,150):
        wave=short if sec==30 else np.pad(ref,(0,(sec-30)*rate))
        wave[-rate:]=tone
        result=extract(wave)
        rows.append(dict(seconds=sec,shape=list(result.input_features.shape),
            mask_frames=int(result.attention_mask.sum()),equal_to_prefix=bool(
            np.array_equal(base.input_features,result.input_features))))
    evidence=dict(offline_cpu_only=True,rate=rate,max_frames=processor.nb_max_frames,
                  configured_context=processor.nb_max_frames//2+256+8,rows=rows)
    (out/"feature-window.json").write_text(json.dumps(evidence,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    main(p.parse_args().out)
