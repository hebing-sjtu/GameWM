# 训练场 / Training ground

The smallest game that exercises the whole engine → VM contract: one
protagonist with three skills, three training dummies, a walled courtyard. It
renders all four proxy modalities, writes the action text from its own event
log, and streams in VAE chunks ([STREAM.md](../../../protocol/STREAM.md)).

## Rules

All numbers live in [`rules.py`](rules.py); times become whole ticks at the
model's fps, one tick per output frame.

| Mechanic | Default skill (fire mage) | Windup | Recovery | Cooldown | Effect |
|---|---|---|---|---|---|
| `melee_attack` | Flame Fist | 0.25 s | 0.35 s | 0.8 s | Nearest standing dummy within 1.3 m of the player and 50° of facing; 34 damage, on the last windup frame |
| `ranged_attack` | Fireball | 0.4 s | 0.3 s | 1.2 s | A projectile leaves the hand at 1.4 m height and flies at 14 m/s; hits the first dummy, wall, pillar or crate; 25 damage |
| `dash` | Ember Dash | — | — | 2.0 s | 4 m in 0.25 s along the move input (or facing), stopped by obstacles |

- Movement: walk 2.2 m/s, turn 150°/s, 30 % speed while an attack winds up or
  recovers. Priority when several buttons are pressed: dash, melee, ranged.
- Dummies: 100 hit points. A hit rocks the dummy on its stand; at 0 it falls
  (0.5 s), stays down 3 s, rises (0.75 s) and is back at full health.
- Events: `melee_attack` hit/miss, `ranged_attack` cast, then hit/miss (target
  a dummy, `wall` or `crate`), `dash` start/end, `knockdown`, `respawn`. Each
  carries `detail.frame` (and damage and remaining hit points on hits).
- Deterministic: the same inputs give the same frames and events
  (`game.inputs` is the replay log, `game.replay()` a fresh game).

The skill tree is data: pass any `character` whose `skills` has the three
mechanics, and the same take becomes a frost mage or a gunslinger.

## Proxy

[`render.py`](render.py) ray-casts a grey-box scene (mannequin of capsules
and spheres, dummies on round stands, box walls, crates and pillars, paved
ground, overcast sky, a glowing orange orb for the fireball) in pure numpy.
One cast gives the shaded image (`style`, framed like the capsule proxies the
VM was trained on), metric z-depth and class ids, from which `depth`,
`semantic` and `duv` take the exact training codes. Classes: player `human`;
dummies, crates and the projectile `prop`; walls and pillars
`building_structure`; ground `terrain`; `sky`.

At 640×352 with 2× supersampling a frame takes about 2 s on one core; the
12.25 s demo (294 frames, all four modalities) takes 22 s with 32 workers.

## Text

[`narrate.py`](narrate.py) turns any frame range into an `action` clause:
locomotion runs from the input as `{move:...}` placeholders (walks toward a
dummy, turns to face one, steps back, sidesteps, and an idle beat for a pause
of 0.6 s or more with no attack under way), and one clause per resolved
attack with an explicit target (`{skill:ranged_attack>dummy_center}`) and
the outcome ("knocking it to the ground", "rocking it back on its stand"). Skills and moves appear only as placeholders: a caption names an
attack only where the engine resolved one, and the character's `skills` and
`moves` decide how the gesture, the effect and the gait read. A skill's text
follows the proxy's arm (raised to shoulder height over the 0.4 s cast,
pulled back over the 0.25 s strike windup), and a repeat in the same segment
uses its shorter `again` phrase.

## Run

Needs numpy; PyAV and Pillow for videos and the take.

```bash
python -m gamewm_engine.games.training_ground --out runs/demo \
  --modalities style,depth,semantic,duv --controller demo --take --preview --workers 32
```

| Output | What |
|---|---|
| `runs/demo/stream/` | The stream: `session.json`, `chunks.jsonl`, `frames/<modality>/<k>.npy` |
| `runs/demo/take/` | The same session as a v1 request (`request.json`, `proxy/style.mp4`, `plan.json`); `first_frame_proxy.png` stands in for `<Picture 1>` until a painted first frame replaces it |
| `runs/demo/preview.mp4` | All modalities side by side with chunk and event labels |
| `runs/demo/inputs.jsonl` | The input log |

## Characters, authored once

[`examples/training_ground/roster.json`](../../../examples/training_ground/roster.json)
lists eight characters (fire mage, ice mage, rifle soldier, dart martial
artist, vampire, thunder warrior, forest druid, arcane sorceress), each with a
look, a themed courtyard, what the grey boxes become, and a skill tree for the
three mechanics (Fireball / Ice Shard / Burst Fire / Steel Darts / ...).
[`author.py`](author.py) turns that plus one turnaround sheet per character
into asset packs ([`authoring.py`](../../authoring.py), `gamewm.asset-pack.v1`):

```bash
python -m gamewm_engine.games.training_ground.author prepare \
  --roster examples/training_ground/roster.json --sheets SHEETS/ --out packs/
# paint packs/<id>/first_frame.png from collage.png with prompt.txt (any image model)
python -m gamewm_engine.games.training_ground.author review --out packs/
python -m gamewm_engine.games.training_ground.author play --out packs/ --runs runs/ --workers 32
```

1. `prepare` renders the opening view at 1344×768 (same 70° field of view, so
   the VM's centre crop is exactly proxy frame 0) with and without the
   player, cuts the back view out of the sheet (GrabCut on the plain
   backdrop) and stands it in the player's box, feet on the box bottom, body
   height matched (a staff or barrel above the head doesn't count). The
   collage is the only reference for the image model, so the character keeps
   its size and place instead of being painted as a portrait.
2. The restyle prompt asks for the same camera and every object in place,
   names the setting, lighting and props, and forbids extra people and UI.
3. `review` writes `review.jpg`: proxy, painting, painting with the proxy's
   class boundaries (red) and player box (blue), sheet. Redo a painting whose
   player leaves the box or whose dummies drift off their stands.
4. `play` runs the demo once per pack (`TrainingGround(character=...,
   scene=..., first_frame=...)`) and writes `runs/<id>/take/request.json`;
   the first pack also gets all four modalities and `runs/proxy_preview.mp4`.

The takes go to a running conversion service through its UI API
(`python -m gamewm_engine submit-ui`, see the top-level README), and
`author overview` tiles the proxy and every render into one labelled grid
with the engine's events burnt in.

First full run (2026-11-05, H3 BD-Sheet R5-B@600, 30 steps, seed 0; the
12.25 s demo is 294 frames in 3 segments, about 33 min per take on one GPU;
results in `/data/binghe/h3_proxy/evals/gamewm-training-ground-oc-20261105/`):
all eight keep their identity and their courtyard, the dummies, pillars and
crates stay where the proxy has them, both knockdowns land on the paving and
the dummies rise again. Ranged effects read clearly for fire, ice, blood,
lightning (a bolt from the sky) and arcane (an orb in the hand); the druid's
thorns are faint and the martial artist's steel darts are not visible, only
the throwing arm.

Second run (same seed, same proxy, the phased skill text and per-character
`moves`; `renders_v2/`, `sbs/<id>_v1_v2.mp4` side by side): frame for frame
the two runs are nearly the same. The pose, the timing of the arm and where
the effect appears follow the proxy, and the text changes only the size and
shape of the effects (the darts are still invisible). With this model the
text describes how things look, but the proxy decides how the body moves: a
gesture or gait the proxy doesn't make cannot be written in.

Third run (`author play --script showcase --camera C --style S`, one 5.17 s
segment per variant, 26 renders on four characters; then the full demo with
`--camera push --style phased`, `renders_v3_demo/`, `sbs_v3/<id>_v1_v3.mp4`):

- Text granularity sets how strong the effects are. `coarse` (only the skill
  name) gives a small fireball, no explosion and a plain dash; `chain` and
  `prose` a medium burst; the per-phase styles (`phased`, `timed`, `spans`)
  a large fireball, a big explosion with smoke, white flashes for thunder and
  frost bursts for ice. The three per-phase styles look alike, so the demo
  uses `phased` (no time stamps inside the caption).
- Write effects as light. Steel darts never show; the same proxy with the
  darts written as "three blazing white streaks of light ... like tiny
  comets" gives three clearly visible white trails that land on the dummy.
  The colour of the proxy's glow (`--glow`) changes nothing.
- The `push` shot (closer, swung 35° to the right during a skill) makes the
  character and the effect much larger than `follow`; `zoom` (narrow lens)
  is in between. With `follow`, the melee target sometimes disappears.
- Known issues: the fire mage's demo grew dummy-like shapes on the wall top;
  melee effects in the push shot are smaller than ranged ones and partly
  hidden by the body; a repeated skill in one segment reads as its short
  `again` phrase, so its effect is weaker than the first.

## Effect specs

The third run says the text decides how an effect looks and the proxy how the
body moves, so both now come from one spec per character
([`effects.py`](effects.py)): per skill a `pose` for the proxy body (ranged:
`point`, `staff`, `throw`, `overhead`, `rifle`, `push`; melee: `punch`,
`palm`, `thrust`, `swing`, `overhead`, `claw`; see `render.POSES`), a `kind`
(`at_target` hides the flying orb, e.g. lightning from the sky), and the
phase texts `windup` / `release` / `travel` / `impact` / `decay` (melee:
`windup` / `impact` / `decay` / `recover`), plus `again` and `text`.
`effects.validate` holds every spec to what worked in hand-written H3 effect
prompts and in our renders: the effect is light with a named colour, the
whole body casts, the hit is short and a `decay` clause clears it off the
target at once, and there are no camera, timing or gore words.

```bash
python scripts/effect_specs.py --roster examples/training_ground/roster.json \
  --refs effect_refs.json --out examples/training_ground/effects \
  --merged examples/training_ground/roster_fx.json      # one ask-claude call per character
python scripts/effect_specs.py ... --files                # or: write <id>.prompt1.txt, read <id>.reply1.txt
python -m gamewm_engine.games.training_ground.author retext \
  --roster examples/training_ground/roster_fx.json --out packs_fx/
python -m gamewm_engine.games.training_ground.author play --out packs_fx/ --runs runs_long/ \
  --script long --camera push --style phased --style-only --workers 32
```

`--refs` is a list of hand-written effect prompts that worked (their
choreography paragraphs are quoted to the model); a reply that fails the
checks goes back with the list of problems. The walk now strides along the
direction of travel with a slight bob and lean, and the arms swing with bent
elbows. In the per-phase styles each window names its first push-in only
(the proxy shows every camera move), a shot still in flight at the end of a
window is told up to its release (the world records the dummy it is headed
for on the `cast` event), and every impact is followed by its `decay`.
`LONG` is a 29.96 s bout (719 frames, eight H3 segments, every segment with
a hit; three knockdowns and three respawns); its captions are 240-330 words.

## From a character card

The effect spec covers the skills only; the look, the themed arena
(`setting`, `lighting`, `props`), the dash phases and the nine moves of the
eight roster characters were written by hand. [`oc.py`](oc.py) turns a card
(identity sheet plus a line naming the role and the three skills, optionally
the pronoun and an arena theme; see
[`cards.json`](../../../examples/training_ground/cards.json)) into the whole
entry. `oc.validate` adds to the effect checks what the captions and the
layouts need: an arena text every layout's rewrites match ("a walled ...
courtyard", "three ... training dummies on round stands", the props sentence
naming walls, ground, dummies, boxes and tall blocks), dash `decay` /
`recover`, all nine moves (`walk_to` / `turn_to` name `{target}`), and no
arena name in skill or move texts except a projectile's "across the
courtyard", which the corridor rewrites. The corridor and rooms layouts now
also rewrite `lighting` ("lights the courtyard" read wrong in the corridor
takes).

```bash
python scripts/oc_author.py --cards examples/training_ground/cards.json --sheets SHEETS/ \
  --out oc_entries/ --merged roster_oc.json            # ask-claude reads the sheet; --backend files for any model
python scripts/oc_regress.py --roster roster_oc.json --runs RUNS_PARENT --report oc_report.md
```

The format example the model sees is a ninth, made-up character
([`oc_example.json`](../../../examples/training_ground/oc_example.json)), so
the eight roster entries stay a held-out reference. `oc_regress.py` swaps an
entry into recorded takes (`runs_dt_<layout>/<id>/take/request.json` keeps
the engine's action text with its placeholders) and recomposes their
captions; with `roster_fx.json` it reproduces the captions of all 24
`dt` takes, except where the lighting and the rifle's "across the yard" /
"scanning the yard" now read right in the corridor.

## Long takes with a VLM gate

The eight `LONG` takes were rendered with the H3 Stage0.5 bidirectional model
(R5-B@600, ref2va-proxy). A segment is 124 frames; each later one starts from
the last 34 frames (10 latents) of the one before, so a wrong object, once
drawn, stays for the rest of the take. Without a check, a few takes grew extra
training dummies or a doorway in a plain wall.

The serve job can now carry a `gate`: after every segment, the Qwen3-VL text
encoder of H3 looks at six moments (proxy | generated) and the segment is
re-rendered with a new seed (`seed + 1000*index + 100*attempt`) up to three
times; the best try is kept and the next segment starts from it. Try 0 is the
ungated render, so a take with no retries is identical to the ungated one.
The judge (`segment_judge.py serve` in FastVideo-oc `scripts/h3_proxy/serve/`)
fails a segment for:

- `structures`: an opening (door, gate, arch, building) where the proxy shows
  a plain wall;
- `dummies`: more dummies than the proxy at every moment where the proxy shows
  any, or two moments short by two or more. Each side is counted in its own
  call; shown both, the VLM copies the proxy count;
- `scene`, `character`, `broken`, or a score below 6.

A grey dummy next to the player in a push-in often reads as none in the
proxy, so moments where the proxy count is zero do not count; before this
rule, the push-in segment of every take failed.

Results on the eight takes: 55 of 64 segments passed on the first try. Four
were fixed by a retry (an arched doorway and buildings behind the fire mage's
wall, a gateway in the dart courtyard wall, an arched doorway in the vampire hall,
a dummy that rose mid-shot for the thunder warrior). Five failed all three
tries and kept the best one: four dummy surpluses (fire, thunder, druid,
arcane) and a wooden gate on the fire mage's left wall. A judge call takes
about 15 s; with retries a take took 27-44 min on 4 GPUs (SP4) and 50-68 min on 2 (SP2).

Re-seeding fixes what the segment itself invents (the doorway, a dummy that
appears mid-shot), but not what it inherits: a dummy already in the 34
carried frames or in the reference frame comes back on every try. Removing
those would need backtracking into the previous segment.

## Other arenas

The arena is a `Layout` (`layout.py`): walls, pillars, crates, dummies, the
start pose, the rectangles the camera may stand in, named marks and regex
rewrites that turn a courtyard roster entry into this arena's text. Rules,
skills and the renderer are shared. Three are defined:

- `courtyard`: the original 20 x 20 m yard with three dummies (default; renders
  byte-identical to before).
- `corridor`: a 6 m wide, 51 m long lane with wall pillars every 10 m, crates
  and six dummies alternating left and right. `CORRIDOR_LONG` works down it.
- `rooms`: two 16 m yards with three dummies each, joined by a 4 m gateway
  between two tall pillars. `ROOMS_LONG` clears the first yard, dashes and walks
  through the gateway (segment 4 is traversal only), then clears the second.
  The camera here is pulled in front of a tall wall that would hide the player
  (`camera_collide`).

Both 29.96 s scripts never miss and knock dummies down four times. `Walk(start,
end, target)` turns toward a dummy or mark and walks once roughly facing it.

```bash
python -m gamewm_engine.games.training_ground.author prepare --layout corridor \
  --roster examples/training_ground/roster_fx.json --sheets SHEETS --out packs_corridor/
python -m gamewm_engine.games.training_ground.author play --out packs_corridor/ --runs runs_corridor/ \
  --script corridor --camera push --style phased --style-only --workers 32
```

`prepare` stores the layout in each pack, `retext` keeps it, and `play` refuses
a script made for another layout.

## Steady camera

`--camera steady` keeps the `follow` framing for the whole take (no skill
push-in) and pans slowly: the yaw eases with a 0.9 s time constant and turns at
most 45°/s, the position trails with 0.2 s. It is meant for footage that will be
cut into a video. Measured on the proxies (`push` → `steady`), the 95th
percentile turn rate drops from 98 / 64 / 97 to 46 / 32 / 45 °/s (courtyard /
corridor / rooms), the fastest turn from 169 / 121 / 198 to 80 / 40 / 62 °/s,
and the 14–20 lens swings per take go to none; every hit's target stays on
screen. In `ROOMS_LONG` the player now squares up at the gateway and waits
0.5 s before dashing (14.6 s instead of 13.5 s) so a slow camera is behind
them, not beside the pillar; the camera of every shot is also pulled in front of
a tall wall when its trail, not only its goal, would cut a corner.

`--controller bot` plays with a simple bot instead of the scripted demo.
From Python, a controller is any function `World -> Input`:

```python
from gamewm_engine.games.training_ground import Face, Hold, Input, Press, Script, TrainingGround

script = Script([Hold(0.0, 1.5, forward=1.0), Face(1.5, 2.0, "dummy_left"), Press(2.0, "ranged")])
game = TrainingGround("h3-33b", ("style", "semantic"))
for message in game.stream(script, seconds=4.0):
    ...
```
