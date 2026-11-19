"""A contact sheet of the same frames from several videos, one row per video.

    python scripts/review_sheet.py --out sheet.jpg --frames 48,64,70 --box 0.35 0.2 0.65 0.8 \\
        proxy=runs/x/take/proxy/style.mp4 render=renders/x/take.mp4

``--box`` is a fraction of each video's width and height, so a 640x352 proxy and a 1280x720
render crop the same region of the picture.
"""

import argparse

import av
from PIL import Image, ImageDraw


def grab(path: str, frames: list[int], box: tuple[float, float, float, float]) -> list[Image.Image]:
    out: dict[int, Image.Image] = {}
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in frames:
                image = frame.to_image().convert("RGB")
                w, h = image.size
                out[i] = image.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
            if i >= max(frames):
                break
    return [out.get(k, Image.new("RGB", (8, 8))) for k in frames]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="label=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--box", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0))
    ap.add_argument("--tile-height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    frames = [int(f) for f in args.frames.split(",")]
    bw, bh = args.box[2] - args.box[0], args.box[3] - args.box[1]
    th = args.tile_height
    tw = round(th * bw * 16 / (bh * 9))
    rows = [(v.split("=", 1)[0], grab(v.split("=", 1)[1], frames, tuple(args.box))) for v in args.videos]
    sheet = Image.new("RGB", (tw * len(frames), th * len(rows)))
    draw = ImageDraw.Draw(sheet)
    for r, (label, tiles) in enumerate(rows):
        for c, tile in enumerate(tiles):
            sheet.paste(tile.resize((tw, th), Image.LANCZOS), (c * tw, r * th))
            draw.text((c * tw + 5, r * th + 4), f"{label} {frames[c] / args.fps:.2f}s", fill=(255, 255, 0),
                      stroke_width=2, stroke_fill=(0, 0, 0))
    sheet.save(args.out, quality=90)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
