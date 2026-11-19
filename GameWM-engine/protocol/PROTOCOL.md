# GameWM-engine → GameWM-VM protocol, v1

Schema id: `gamewm.engine-request.v1` ([JSON Schema](request.schema.json),
[example](../examples/seg0162_fire_mage.json)).

The engine decides *where everything is and what happens*; the VM decides *how
it looks*. A request is everything the VM needs for one **take**, a stretch of
play rendered from the player's camera. The VM never changes an outcome: by the
time a request is sent, every event in it has already been resolved by engine
code.

## 1. What the engine guarantees

| The engine owns (exact, replayable) | The VM owns (sampled) |
|---|---|
| rules, scores, health, collisions, quest state | appearance of characters, props and world |
| positions and coarse poses of all entities | limb motion, contact, deformation |
| the camera | how a skill looks (projectile, impact, residue) |
| which events happen, to whom, and when | the target's reaction, in step with the event time |

So the request must carry *event semantics* (type, participants, time,
outcome) plus coarse kinematics in the proxy, not a fully posed animation.

## 2. Assets (authored once)

| Field | VM slot | Contract |
|---|---|---|
| `assets.first_frame` | `<Picture 1>` | Scene concept art from the engine's opening camera view. Objects must sit where proxy frame 0 has them; it receives exactly the same crop as the proxy. |
| `assets.character.sheet` | `<Picture 2>` | Optional identity sheet: front, side and back of the subject. `subject` is `protagonist` or `vehicle`. |
| `assets.character.look` | text | Noun phrase for the subject's appearance (`a young dark-haired man in a dark red leather coat`). |
| `assets.character.skills` | text | The skill tree: engine mechanic → `{name, text, again?}`. `text` is a verb phrase; `{target}` inside it becomes the event target's entity text. `again` is the shorter phrase for a repeat in the same caption. |
| `assets.character.skills.*.phases` | text | Optional sentence-level text per phase (`windup`, `release`, `travel`, `impact`, `decay`, `recover`) for fine-grained captions. `windup`, `release` and `recover` are verb phrases of the character; `travel`, `impact` and `decay` are clauses with their own subject ("the fireball streaks…", "it explodes against {target}", "the flames vanish in a split second"). |
| `assets.character.skills.*.pose`, `.kind` | engine | Engine-side only, never read by the VM: the gesture the proxy body plays for the skill, and whether the proxy shows the shot flying (`projectile`) or only where it lands (`at_target`). |
| `assets.character.pronoun` | text | `he`, `she`, `it` or `they` (default); fills `{he}`, `{him}`, `{his}` and their capitalised forms in `action`. |
| `assets.character.moves` | text | Optional: how this character moves, kind → verb phrase (`walk`, `walk_to`, `back`, `strafe_left`, `strafe_right`, `turn_left`, `turn_right`, `turn_to`, `idle`). Missing kinds read plainly (`walks forward`). |
| `assets.scene` | text | `medium` (default `A 3D video game`), `setting`, `lighting`. |

A skill is *mechanic × attribute*: the mechanic (`ranged_attack`) is engine
code shared by every character, and the entry says how this character's
attribute realizes it (`Fireball`). Writing one mechanic gives a family of
skills across characters.

Write a skill's `text` in the order the proxy shows it: the gesture that
starts it (the proxy's arm movement during the windup: raised to shoulder
height for a cast, pulled back for a strike), the release, what flies and how
it looks, and the effect on `{target}`. A gesture the proxy does not make
(an overhead swing over a proxy that thrusts) fights the proxy and reads as a
mismatch between the hand and the effect. `moves` does the same for
locomotion: the gait, what the clothes and the held item do, the stance at
rest; the engine still decides when and where the character moves.

## 3. Take (produced at run time)

### Proxy video (`<Video 1>`)

- Rendered from the player's camera, any frame rate (resampled to the model's
  fps by whole-frame selection), any 16:9-like size (center-cropped to the
  model's aspect).
- `modality` says what the frames are:

| `modality` | What the engine renders | VM preprocessing |
|---|---|---|
| `style` | Ordinary RGB render: capsule bodies, low-poly or grey-box meshes, voxels, or the shipped renderer | Lanczos resize |
| `depth` | Grey depth, all channels equal | luminance, Lanczos resize; inverted if `near_is_bright` is false |
| `semantic` | Flat class colours (the 12 CWM classes), hard edges | nearest resize |
| `duv` | Packed: R = log depth, G/B = class colours | nearest resize |

- `kind` is informational (`capsule`, `lowpoly`, `voxel`, `blender`, …).
- Coarser proxies give the VM more freedom: a capsule says only where a body
  is, so the VM supplies the limbs; a low-poly mesh also pins the pose.

### Entities and events

- `entities`: id → noun phrase (`"cultist": "the hooded cultist in a black robe"`).
  `player` is implicit.
- `events`: `{t, mechanic, actor, target?, outcome?}`, `t` in seconds from the
  take's first proxy frame. These come from the engine log and are final. An
  event belongs to frame `floor(t·fps + 0.001)`, so `t = frame / fps` written
  with six decimals always lands on its own frame, also on segment boundaries.

### Text

- `action`: what happens, as a clause that continues "The protagonist, *look*,
  …". One string for the whole take, or one per segment. Placeholders:
  - `{skill:<mechanic>}` → that mechanic's skill text. **A segment may use it
    only if an event of that mechanic falls inside the segment**, so the
    caption can never describe an attack the engine did not resolve there.
    The skill's `{target}` becomes the first such event's target.
  - `{skill:<mechanic>><entity>}` → the same, but the event must have that
    target (use it when one segment holds attacks on different targets).
  - `{target:<entity>}` → the entity's noun phrase.
  - `{move:<kind>}` / `{move:<kind>><entity>}` → the character's phrase for
    that movement (`walk_to` and `turn_to` need the entity).
  - A skill used a second time in the same segment reads as its `again`
    phrase (or its text followed by "again").
  - `{skill:<mechanic>.<phase>}` (also with `><entity>`) → one phase of the
    skill, so the action can be several sentences ("{He} {skill:ranged_attack.windup}.
    {skill:ranged_attack.impact}."); `.name` gives the skill's name. The
    sentence starts are capitalised after composition.
  - `{he}` / `{him}` / `{his}` (and `{He}` …) → the character's pronoun.
  - Segment-relative time stamps such as `[2.62s-2.93s]` may appear inside
    the action; the engine's text styles (`chain`, `prose`, `phased`,
    `timed`, `spans`, `coarse`) differ in exactly these choices.
- Events may carry a `detail` object (damage, hit points, …) for logs and
  tools; it never enters the text.
- `camera`: an optional closing sentence about the camera.

## 4. What the VM receives

### Segments

The VM generates fixed-length clips. A longer take is cut into overlapping
segments; segment *k* covers take frames `[k·stride, k·stride + frames)`, and
the proxy holds its last frame if it ends early.

| `model` | Clip | fps | Target | Proxy | Stride | Max segments |
|---|---|---|---|---|---|---|
| `h3-33b` | 124 frames (5.17 s) | 24 | 1280×704 | 640×352 | 85 (39-frame overlap) | 16 |
| `wan2.2-5b` | 121 frames | 24 | 1280×704 | 640×352 | — | 1 |
| `wan2.2-14b` | 81 frames | 16 | 832×480 | 416×240 | — | 1 |

The first segment uses the opening role `w0_omni` (one leading latent fixed
from `<Picture 1>`), later ones the continuation role `wn_omni` (ten latents
carried over). The causal (DMD) students read, for chunk *k*, only proxy
chunks *k−5 … k*: the engine never needs to send the future.

### Caption and document

Each segment's caption is

```
[0.00s-5.17s] {medium} captures {setting}. {lighting}. {Subject}, {look}, {action}. {camera}
```

where `{Subject}` is `The protagonist` / `The vehicle`, followed by
`from the identity sheet` when a sheet is given. The `[0.00s-5.17s]` prefix
is the training captions' span and is kept verbatim for every backbone.

The text the VM encodes is

```
Reference modality: {modality} video.
{identity line for the subject, only with a sheet}
{caption}
```

The identity lines are fixed sentences the VM was trained on
(`SHEET_PROMPTS` in [`protocol.py`](../gamewm_engine/protocol.py)).

### Image preprocessing

`<Picture 1>` is center-cropped to the target aspect and scaled to short edge
2048; `<Picture 2>` is scaled to short edge 1024.

## 5. Delivery

`gamewm_engine.vm.Encoder` encodes a request with the VM's own cache encoders
(VAE + Qwen3-VL, from FastVideo-oc) into `h3-proxy-serve-job.v1` samples, and
`enqueue` drops the job into the spool of GameWM-VM's resident workers
(`python -m solarwm.backends.minimax_h3.proxy_loop serve`). Results land in
`spool/done/<id>.json` and `spool/out/<id>/take.mp4`.

## 6. Not in v1

- Streaming chunk-by-chunk delivery to the causal students. A request is
  take-based; the engine side of streaming is specified separately in
  [STREAM.md](STREAM.md) (`gamewm.engine-stream.v1`), and any recorded stream
  converts back into a v1 request (`TakeSink`), which today's VM runs.
- Event-aware proxy semantics (a dedicated class or marker for projectiles and
  hit frames); today the hit time lives in the event log and the caption.
- Wan delivery through the serve spool; Wan requests run through `solarwm infer`.
