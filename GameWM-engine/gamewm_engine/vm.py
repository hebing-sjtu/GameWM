"""Hand a request to GameWM-VM's H3 serve workers (``proxy_loop serve``).

Encoding needs a GPU, the MiniMax-H3 snapshot, and FastVideo-oc, whose conversion service holds
the cache encoders the VM was trained with. This module only arranges the request for them, so
an engine request and an upload to the conversion UI become the same samples.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .compose import Plan, plan
from .protocol import ProtocolError, Request

JOB_SCHEMA = "h3-proxy-serve-job.v1"


def _service(fastvideo_oc: Path):
    serve = fastvideo_oc / "scripts" / "h3_proxy" / "serve"
    if not (serve / "proxy_service.py").is_file():
        raise FileNotFoundError(f"FastVideo-oc conversion service not found under {serve}")
    if str(serve) not in sys.path:
        sys.path.insert(0, str(serve))
    import proxy_service

    return proxy_service


class Encoder:
    """One resident set of cache encoders; reuse it across requests."""

    def __init__(self, fastvideo_oc: str | Path, model: str | Path = "/data/models/MiniMax-H3",
                 device: str = "cuda:0") -> None:
        self.ps = _service(Path(fastvideo_oc))
        self.ps.init_single_process_distributed()
        self.cache = self.ps.CacheEncoder(Path(model), device)

    def job(self, request: Request, job_dir: str | Path) -> tuple[Path, Plan]:
        """Encode every segment into ``job_dir`` and write its ``job.json``."""

        import numpy as np
        import torch
        from PIL import Image

        if request.model != "h3-33b":
            raise ProtocolError("the serve adapter covers h3-33b; Wan models run through solarwm infer")
        ps, job_dir = self.ps, Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        proxy = request.proxy
        reference = ps.to_reference(ps.read_proxy_video(proxy.video), proxy.modality, proxy.near_is_bright)
        layout = plan(request, int(reference.shape[0]))
        windows = ps.segment_windows(reference, len(layout.segments))
        ps.write_mp4(job_dir / "proxy_reference.mp4", reference[:layout.output_frames])

        g = request.geometry
        anchor = ps.read_anchor_image(request.first_frame, np.zeros((1, g.height, g.width, 3), np.uint8),
                                      ps.ANCHOR_SHORT_EDGE, aspect=(g.height, g.width))
        target = self.cache.still_target(request.first_frame)
        anchor_latent = self.cache.encoders.encode_keyframe(anchor)
        oc = None
        if request.character.sheet:
            with Image.open(request.character.sheet) as raw:
                sheet = ps.scale_to_short_edge(raw.convert("RGB"), ps.OC_SHORT_EDGE)
            oc = (sheet, self.cache.encoders.encode_keyframe(sheet), request.character.subject)

        rows = []
        for segment, window in zip(layout.segments, windows, strict=True):
            sample = self.cache.sample(window=window, modality=proxy.modality, role=segment.role,
                                       caption=segment.caption, anchor=anchor, target=target,
                                       anchor_latent=anchor_latent, oc=oc,
                                       info={"take": job_dir.name, "take_segment": segment.index,
                                             "take_anchor": "anchor", "engine_request": str(request.source or "")})
            encoded = sample["info"].get("oc_conditioning_prompt") or sample["info"]["conditioning_prompt"]
            if encoded != segment.document:
                raise ProtocolError(f"segment {segment.index}: the encoder's document differs from the protocol's")
            path = job_dir / f"{job_dir.name}-s{segment.index:02d}.pt"
            torch.save(sample, path)
            rows.append({"name": path.stem, "pt": str(path), "cwm_system": segment.role})
        torch.cuda.empty_cache()
        payload = {"schema": JOB_SCHEMA, "segments": rows, "fps": g.fps, "frames": layout.output_frames,
                   "steps": request.steps, "seed": request.seed}
        (job_dir / "job.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (job_dir / "plan.json").write_text(json.dumps(plan_json(layout), ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        return job_dir / "job.json", layout


def enqueue(job_json: str | Path, spool: str | Path, job_id: str | None = None) -> str:
    """Drop a job into a serve spool; the rename is what makes a worker see it."""

    job_json, spool = Path(job_json), Path(spool)
    job_id = job_id or job_json.parent.name
    queue = spool / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    scratch = queue / f".{job_id}.tmp"
    scratch.write_text(job_json.read_text(encoding="utf-8"), encoding="utf-8")
    os.replace(scratch, queue / f"{job_id}.json")
    return job_id


def wait(spool: str | Path, job_id: str, poll: float = 10.0, timeout: float | None = None) -> dict[str, Any]:
    """Block until the worker writes ``done/<id>.json`` and return it."""

    done = Path(spool) / "done" / f"{job_id}.json"
    start = time.time()
    while True:
        # A GCS-fuse spool can hide new files from a stat; listing the directory refreshes it.
        if done.parent.is_dir() and done.name in os.listdir(done.parent):
            return json.loads(done.read_text(encoding="utf-8"))
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError(f"{job_id} not done after {timeout:.0f}s")
        time.sleep(poll)


def _blocks(text: str) -> list[str]:
    """The service's caption box: one blank-line separated block per segment, ``#`` lines are labels."""
    import re

    body = "\n".join("" if line.lstrip().startswith("#") else line for line in text.splitlines())
    return [part for part in re.split(r"\n\s*\n", body.strip()) if part.strip()]


def submit_ui(url: str, request: Request, out_dir: str | Path, *, submitter: str = "GameWM-engine",
              timeout: float = 8 * 3600, gate: dict[str, Any] | None = None,
              spool: str | Path = "/workspace/serve/spool") -> dict[str, Any]:
    """Run a request through a running conversion service (its Gradio API) and wait for the take.

    The service's resident encoders do the encoding, so this needs no GPU on the caller's side. The
    captions are the protocol's, verbatim and one per segment; the service's VLM is not consulted.
    With ``gate`` the service only encodes (``/encode``), and the job is queued here, on the
    service's ``spool``, with that per-segment gate added (``anchor``/``oc`` are the job's own).
    Writes ``take.mp4``, ``compare.mp4`` and ``done.json`` into ``out_dir``.
    """
    import math
    import shutil

    from gradio_client import Client, handle_file

    if request.model != "h3-33b":
        raise ProtocolError("the conversion service runs h3-33b")
    layout = plan(request)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seconds = math.ceil(layout.output_frames / request.geometry.fps * 2) / 2
    media = {
        "proxy": handle_file(str(request.proxy.video)),
        "anchor": handle_file(str(request.first_frame)),
        "oc": handle_file(str(request.character.sheet)) if request.character.sheet else None,
        "modality": request.proxy.modality,
        "depth": "near_bright" if request.proxy.near_is_bright else "near_dark",
        "max_seconds": seconds,
    }
    client = Client(url, verbose=False)
    _, boxed, _ = client.predict(**media, text="", draft=False, captions="", api_name="/analyze")
    planned = len(_blocks(boxed))
    if planned != len(layout.segments):
        raise ProtocolError(f"the service planned {planned} segments, the protocol {len(layout.segments)}")
    captions = "\n\n".join(s.caption for s in layout.segments)
    common = {**media, "oc_subject": request.character.subject, "captions": captions, "steps": request.steps,
              "seed": request.seed, "submitter": submitter}
    if gate is None:
        job = client.submit(**common, text="", draft=False, api_name="/generate")
        summary, take, compare, info, _, _ = job.result(timeout=timeout)
        if not take:
            raise RuntimeError(f"the service did not return a take: {summary}")
        shutil.copyfile(take["video"], out / "take.mp4")
        shutil.copyfile(compare["video"], out / "compare.mp4")
    else:
        encoded = client.predict(**common, api_name="/encode")
        job_dir = Path(encoded["dir"])
        payload = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        oc = sorted(job_dir.glob("oc.*"))
        payload["gate"] = {**gate, "anchor": str(next(iter(sorted(job_dir.glob("anchor.*"))))),
                           "oc": str(oc[0]) if oc else None}
        (job_dir / "job.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        job_id = enqueue(job_dir / "job.json", spool, encoded["job"])
        info = wait(spool, job_id, timeout=timeout)
        summary = f"gated job {job_id}"
        if info.get("error"):
            raise RuntimeError(f"{job_id} failed: {info['error']}")
        shutil.copyfile(info["take"], out / "take.mp4")
        shutil.copyfile(info["compare"], out / "compare.mp4")
    (out / "done.json").write_text(json.dumps({"summary": summary, "info": info, "plan": plan_json(layout)},
                                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def plan_json(layout: Plan) -> dict[str, Any]:
    r = layout.request
    return {
        "model": r.model,
        "proxy_frames": layout.proxy_frames,
        "output_frames": layout.output_frames,
        "fps": r.geometry.fps,
        "segments": [{
            "index": s.index,
            "frames": [s.start, s.end],
            "role": s.role,
            "events": [e.__dict__ for e in s.events],
            "caption": s.caption,
            "document": s.document,
        } for s in layout.segments],
    }
