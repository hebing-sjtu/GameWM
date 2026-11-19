"""``python -m gamewm_engine {validate,plan,serve-job}`` on a request JSON."""

from __future__ import annotations

import argparse
import json
import sys

from .compose import plan
from .protocol import ProtocolError, load
from .vm import Encoder, enqueue, plan_json, submit_ui, wait


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gamewm_engine")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="check a request against the protocol")
    v.add_argument("request")
    v.add_argument("--no-files", action="store_true", help="skip checking that media paths exist")
    pl = sub.add_parser("plan", help="print the segments, captions and documents the VM will receive")
    pl.add_argument("request")
    pl.add_argument("--frames", type=int, help="proxy length at the VM's fps; defaults to take.seconds")
    pl.add_argument("--no-files", action="store_true")
    j = sub.add_parser("serve-job", help="encode the request for the H3 serve workers (GPU)")
    j.add_argument("request")
    j.add_argument("--job-dir", required=True)
    j.add_argument("--fastvideo-oc", required=True)
    j.add_argument("--model", default="/data/models/MiniMax-H3")
    j.add_argument("--spool", help="also enqueue the job into this serve spool")
    j.add_argument("--wait", action="store_true", help="block until the job is done")
    u = sub.add_parser("submit-ui", help="render requests through a running conversion service (no local GPU)")
    u.add_argument("requests", nargs="+", help="request JSONs; each renders into --out/<request's parent name>")
    u.add_argument("--url", default="http://127.0.0.1:7860/")
    u.add_argument("--out", required=True)
    u.add_argument("--parallel", type=int, default=1, help="requests in flight at once")
    u.add_argument("--submitter", default="GameWM-engine")
    u.add_argument("--gate", help="JSON of a per-segment gate (url, attempts, min_score, ...); the job is queued on "
                                  "--spool with it instead of through the UI's queue")
    u.add_argument("--spool", default="/workspace/serve/spool")
    args = p.parse_args(argv)

    if args.command == "submit-ui":
        return _submit_ui(args)
    try:
        request = load(args.request, check_files=not getattr(args, "no_files", False))
        if args.command == "validate":
            print(f"ok: {request.model}, {request.proxy.modality} proxy, {len(request.events)} events")
        elif args.command == "plan":
            print(json.dumps(plan_json(plan(request, args.frames)), ensure_ascii=False, indent=2))
        else:
            job, layout = Encoder(args.fastvideo_oc, args.model).job(request, args.job_dir)
            print(f"wrote {job}: {len(layout.segments)} segments, {layout.output_frames} frames")
            if args.spool:
                job_id = enqueue(job, args.spool)
                print(f"queued {job_id} in {args.spool}")
                if args.wait:
                    print(json.dumps(wait(args.spool, job_id), ensure_ascii=False, indent=2))
    except ProtocolError as error:
        print(f"protocol error: {error}", file=sys.stderr)
        return 2
    return 0


def _submit_ui(args: argparse.Namespace) -> int:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    def one(path: str) -> tuple[str, str]:
        name = Path(path).parent.name if Path(path).parent.name != "take" else Path(path).parent.parent.name
        began = time.time()
        info = submit_ui(args.url, load(path), Path(args.out) / name, submitter=f"{args.submitter} · {name}",
                         gate=json.loads(args.gate) if args.gate else None, spool=args.spool)
        stopped = [r["index"] for r in info.get("segments", []) if r.get("stopped")]
        tail = f", stopped at segment {stopped[0]}" if stopped else ""
        return name, f"{info.get('frames')} frames, worker {info.get('worker')}, {time.time() - began:.0f}s{tail}"

    failed = 0
    with ThreadPoolExecutor(max(1, args.parallel)) as pool:
        futures = {pool.submit(one, path): path for path in args.requests}
        for future in as_completed(futures):
            try:
                name, line = future.result()
                print(f"done {name}: {line}", flush=True)
            except Exception as error:  # keep the other requests going
                failed += 1
                print(f"failed {futures[future]}: {type(error).__name__}: {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
