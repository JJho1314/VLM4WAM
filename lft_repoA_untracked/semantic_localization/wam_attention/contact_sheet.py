"""Contact sheet of exported panels so the good ones can be spotted without opening 120 folders.

Each row is one scene: RGB-only overlay | SG-WAM overlay, captioned with its concentration numbers
and prompt. Panels stay in the export's ranked order (best concentration gain first).
"""
import os, glob, csv
import numpy as np
from PIL import Image, ImageDraw

PANEL_DIR = os.environ.get("PANEL_DIR",
                           "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs/panels_composite_all")
OUT = os.environ.get("SHEET", PANEL_DIR + "/_contact_sheet.png")
COLS = int(os.environ.get("COLS", 4))          # scenes per row
PER = int(os.environ.get("SHEET_MAX", 40))     # scenes on the sheet
TH = int(os.environ.get("THUMB_H", 150))


def main():
    meta = {}
    idx = os.path.join(PANEL_DIR, "index.tsv")
    if os.path.exists(idx):
        with open(idx) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                meta[row["panel"]] = row
    dirs = sorted(d for d in glob.glob(f"{PANEL_DIR}/*") if os.path.isdir(d))[:PER]
    if not dirs:
        print("no panels"); return

    tiles = []
    for d in dirs:
        name = os.path.basename(d)
        try:
            a = Image.open(f"{d}/rgbonly.png"); b = Image.open(f"{d}/sg.png")
        except Exception:
            continue
        w = int(a.width * TH / a.height)
        a = a.resize((w, TH)); b = b.resize((w, TH))
        pair = Image.new("RGB", (w * 2 + 4, TH), (255, 255, 255))
        pair.paste(a, (0, 0)); pair.paste(b, (w + 4, 0))
        m = meta.get(name, {})
        cap = f"{name.split('_')[0]} SG={m.get('SG_conc','?')[:5]} RGB={m.get('RGB_conc','?')[:5]}  {m.get('prompt','')[:46]}"
        tiles.append((pair, cap))

    tw = max(t.width for t, _ in tiles); th = TH + 16
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * (tw + 8), rows * (th + 6)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (t, cap) in enumerate(tiles):
        x = (i % COLS) * (tw + 8); y = (i // COLS) * (th + 6)
        sheet.paste(t, (x, y)); dr.text((x + 2, y + TH + 2), cap, fill=(0, 0, 0))
    sheet.save(OUT)
    print(f"SAVED {OUT}  ({len(tiles)} scenes, left=RGB-only right=SG-WAM)", flush=True)


if __name__ == "__main__":
    main()
