# GameWM-engine

The engine half of GameWM. It keeps what must be exact, everything a designer
must be able to inspect, replay and balance: rules, scores, health,
collisions, quest state, the camera, the layout of the world, and the
positions and coarse poses of all entities. It renders a coarse **proxy
video** from the player's camera, logs every event with its frame, and hands
each take to [GameWM-VM](../GameWM-VM) together with the game's authored
assets (scene concept art, character sheet, skill text). The VM decides how
the take looks and how each interaction plays out; it never changes an
outcome. One mechanic, say `ranged_attack`, can therefore be realized as a
fireball, a blood bolt or a lightning call, while the target, the hit frame
and the damage stay the engine's.

The GameWM-engine is a contract, not a particular engine: any engine,
hand-written or produced by a coding agent, fills the role if it supplies the
proxy video and the event log in the request format below.

This directory holds the contract between the two:

| Path | What |
|---|---|
| [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md) | The request specification, v1 |
| [`protocol/STREAM.md`](protocol/STREAM.md) | The streaming output, `gamewm.engine-stream.v1`: one message per VAE chunk |
| [`protocol/request.schema.json`](protocol/request.schema.json) | JSON Schema of a request |
| [`gamewm_engine/`](gamewm_engine) | Reference implementation: parse and validate a request, cut it into VM segments, compose captions, encode and queue it for the VM; stream messages and sinks (`stream.py`); proxy codes (`proxy.py`) |
| [`gamewm_engine/games/training_ground/`](gamewm_engine/games/training_ground) | 训练场: a sample game with melee, ranged and dash skills against training dummies, rendering all four proxy modalities and streaming them; three arenas and 30 s scripted bouts played by eight authored characters |
| [`examples/`](examples) | Example requests (the seg_0162 fire-mage Fireball take) |

## A request in one picture

```
authored once                     per take (engine, run time)              GameWM-VM slot
-------------                     ---------------------------              --------------
first frame (scene concept art)                                       ->   <Picture 1>
character sheet (front/side/back)                                     ->   <Picture 2>
skill tree: mechanic -> skill     events: ranged_attack @ 3.5 s       ->   caption
                                  proxy video (capsule / low-poly /   ->   <Video 1>
                                  depth / semantic / DUV)
```

## Use

Python 3.10+, no dependencies for building and checking requests:

```bash
cd GameWM-engine
pip install -e .            # or: export PYTHONPATH=$PWD
python -m gamewm_engine validate examples/seg0162_fire_mage.json --no-files
python -m gamewm_engine plan examples/seg0162_fire_mage.json --no-files
```

`plan` prints the segments, the events in each, and the exact caption and text
document the VM will encode.

Encoding and queueing for the resident H3 workers needs a GPU, the MiniMax-H3
snapshot and FastVideo-oc (its conversion service holds the VM's cache
encoders):

```bash
python -m gamewm_engine serve-job request.json \
  --job-dir /workspace/serve/jobs/my-take \
  --fastvideo-oc /workspace/FastVideo-oc \
  --spool /workspace/serve/spool --wait
```

Or, without a GPU, through a running conversion service's web UI (needs
`gradio_client`); the service re-plans the request, checks it gives the same
segments, encodes and queues it like any user job:

```bash
python -m gamewm_engine submit-ui runs/*/take/request.json \
  --url http://127.0.0.1:7860/ --out renders/ --parallel 3
```

Each take lands in `renders/<name>/{take.mp4, compare.mp4, done.json}`.

From Python:

```python
from gamewm_engine import load, plan

request = load("examples/seg0162_fire_mage.json", check_files=False)
for segment in plan(request).segments:
    print(segment.role, segment.events, segment.document)
```

The sample game needs numpy (PyAV and Pillow for videos):

```bash
pip install -e '.[games]'
python -m gamewm_engine.games.training_ground --out runs/demo \
  --modalities style,depth,semantic,duv --take --preview --workers 8
```

## Tests

```bash
pip install -e '.[test,games]'
pytest -q tests
```
