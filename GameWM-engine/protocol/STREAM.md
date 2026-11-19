# GameWM-engine streaming output, v1

Schema id: `gamewm.engine-stream.v1`
(reference implementation: [`gamewm_engine/stream.py`](../gamewm_engine/stream.py),
producer: [the training ground](../gamewm_engine/games/training_ground)).

A request ([PROTOCOL.md](PROTOCOL.md)) describes a finished take. A stream
carries the same information while the game runs, so a causal VM can start
on a piece of video as soon as the engine has finished it. The guarantees
don't change: every event in a stream is already resolved by engine code, and
the VM only decides how it looks.

## 1. Time is cut where the VAE cuts it

The stream unit is one **chunk** = one temporal block of the VM's video VAE.
A chunk is the smallest piece the VM can encode, and therefore the smallest
piece it can usefully receive. One engine tick is one output frame at the
model's fps.

| `model` | fps | chunk 0 (prefix) | chunk *k* ≥ 1 | attention window |
|---|---|---|---|---|
| `h3-33b` | 24 | 5 frames → 2 latents | 17 frames → 5 latents | 6 chunks |
| `wan2.2-5b` | 24 | 1 frame → 1 latent | 20 frames → 5 latents | 6 chunks |
| `wan2.2-14b` | 16 | 1 frame → 1 latent | 12 frames → 3 latents | 6 chunks |

Chunk *k* ≥ 1 covers frames `[prefix + (k−1)·size, prefix + k·size)`. One H3
segment (124 frames, 37 latents) is chunks 0…7. The causal students denoise
chunk *k* from proxy chunks *k−5 … k*, which is why the text is written per
window (§3.3) and why the engine never needs to send the future.

## 2. Messages

A stream is an ordered sequence of JSON messages, all with `schema` and
`type`:

```
session            once, before any frames
chunk 0, 1, 2, …   one per finished chunk, in order
end                once
```

Frames travel beside the JSON: in process as `uint8 [n, H, W, 3]` arrays,
on disk as `.npy` files the chunk message names (§4).

### 2.1 `session`

| Field | Meaning |
|---|---|
| `session_id`, `game` | Ids |
| `model`, `fps` | The VM backbone and its frame rate (= the engine's tick rate) |
| `proxy_size`, `output_size` | `[w, h]` of streamed proxy frames and of the VM's output |
| `chunking` | `prefix_frames`, `chunk_frames`, `window_chunks`, `prefix_latents`, `chunk_latents` (§1) |
| `modalities` | Which proxy modalities each chunk carries (any of `style`, `depth`, `semantic`, `duv`) |
| `text_modality` | The modality the chunk documents name (`Reference modality: … video.`) |
| `proxy_kind` | As in v1 (`capsule`, …) |
| `assets` | `first_frame` (path or `null`), `scene`, `character` with its skill tree; the same fields as a v1 request |
| `entities` | id → noun phrase; every event participant other than `player` |
| `camera` | The closing camera sentence of every caption |
| `rules` | Optional: the numbers outcomes depend on (windups, cooldowns, damage …), for tools |

If `assets.first_frame` is `null`, the consumer paints `<Picture 1>` from
proxy frame 0 (chunk 0) before generation starts; the first frame must sit
where proxy frame 0 has its objects.

### 2.2 `chunk`

| Field | Meaning |
|---|---|
| `index` | *k* |
| `frames`, `t` | `[start, end)` in frames and in seconds since the session's frame 0 |
| `latents` | How many latent frames this chunk encodes to |
| `proxy` | On disk: `{modality: path}`; in process the arrays are `Chunk.frames` |
| `events` | Events resolved in this chunk: v1 events with `detail.frame` (§3.1) |
| `states` | One entry per frame: entity poses, health, projectiles, the camera's `c2w` (4×4, x right, y up, z forward) and field of view, and the input that produced the frame |
| `text` | `action` for this chunk alone; `window` `[k₀, k]`; `window_frames`; `window_action`; `caption` and `document` for the window (§3.3) |

A chunk is final when it is sent: later chunks never revise it.

### 2.3 `end`

`chunks`, `frames`, `reason` (`finished`, `stopped`, …). A session that
stops between chunk boundaries sends the partial last chunk first (shorter
`frames`, same fields).

## 3. Contracts

### 3.1 Events

Same as v1: `{t, mechanic, actor, target?, outcome?, detail?}`, `t` in seconds
since frame 0, `t = frame / fps` with six decimals, and `detail.frame` the
frame the engine resolved it on. An event belongs to the chunk holding that
frame. Mechanics with skills are the keys of the character's skill tree;
others (`knockdown`, `respawn`) are world events.

### 3.2 Proxy frames

Exactly the v1 modalities and codes (PROTOCOL.md §3,
[`gamewm_engine/proxy.py`](../gamewm_engine/proxy.py)): `semantic` and `duv`
carry exact palette codes and must never be lossily compressed; `depth` is
log depth between 0.3 m and 256 m, near bright, 0 where there is no surface.
All modalities of one chunk come from the same render, so they align pixel
for pixel.

### 3.3 Text

`text.action` uses v1 placeholders and follows the v1 rule: a chunk may name
a skill only if an event of that mechanic (and target, for
`{skill:<mechanic>><entity>}`) falls inside the span the text covers.

The VM conditions on the window, not on a single chunk, so each chunk also
carries the text for frames `[start of chunk k−5, end of chunk k)`:
`window_action`, and its `caption` / `document` built exactly like a v1
segment's (same `[0.00s-5.17s]` span, same identity line). A consumer that
re-encodes text only every few chunks can use the latest one it has.

**Status.** Today's causal students were trained with one caption per
124-frame segment, so the VM does not yet accept per-chunk text. Until it
does, record the stream and convert it into a v1 request (`TakeSink`, §4),
which the VM runs as is. The per-chunk text is the engine half of the live
path; text re-conditioning per window is the VM half.

## 4. Sinks (reference implementation)

| Sink | Writes |
|---|---|
| `DirectorySink(root)` | `session.json`; `chunks.jsonl`, one line per chunk and then the `end` line, flushed as each chunk lands (a consumer can tail it); `frames/<modality>/<index>.npy`. `read_directory(root)` replays it. |
| `TakeSink(root, action_for, modality)` | A v1 request: `proxy/<modality>.mp4` (lossless RGB H.264 for `semantic` / `duv`), `request.json` cut into VM segments, each narrated from the engine log, and `plan.json`; it validates with `protocol.load` + `compose.plan` before returning. |
| `PreviewSink(path)` | A side-by-side mp4 of all modalities with the chunk index and recent events burnt in. |

```python
from gamewm_engine.stream import DirectorySink, pump
from gamewm_engine.games.training_ground import DEMO, TrainingGround

game = TrainingGround("h3-33b", ("style", "depth", "semantic", "duv"))
pump(game.stream(DEMO, DEMO.seconds), DirectorySink("runs/demo/stream"), game.take_sink("runs/demo/take"))
```

A live loop drives the game itself:

```python
game = TrainingGround("h3-33b", ("style",))
send(game.session())
while playing:
    chunk = game.feed(read_gamepad())   # one frame per call
    if chunk is not None:
        send(chunk)                     # chunk.meta + chunk.frames
if (tail := game.flush()) is not None:
    send(tail)
send(game.end())
```
