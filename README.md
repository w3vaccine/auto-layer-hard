# MTD-2396 Auto Layer POC 

Working Magic Layer POC: flat print → motif proposal → review → **Layers Menu** (durable accepted layers) → motif pack export.

This is a **POC**, not product code. Contracts below are what to mirror when integrating.

---

## Quick start

```bash
cd auto-layer-poc
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

.venv/bin/python app.py
# → http://127.0.0.1:8787
```

1. Pick **Seashells** (or showcase / hard)
2. Run **CV only**
3. **Open review** → Accept all high → **Push accepted → Layers Menu**
4. Open Layers Menu document → **Download motif pack**

Optional smoke (no server):

```bash
.venv/bin/python scripts/smoke_handoff.py
# → out/smoke_handoff/smoke_report.json
```

`GOOGLE_API_KEY` is optional (tile gapfill / Gemini inpaint only). Default CV path needs no key. Hard path (soft/camo/busy) needs the key.

### Render

Docker image: `docker/Dockerfile` (CPU torch + sam2_b). Needs Pro / 4GB. Set `GOOGLE_API_KEY`. Blueprint: `render.yaml`. Health: `GET /health`.

Busy/hard path latency (CPU): target ~1–2 min with quality — up to 3 Gemini discover passes until ≥12 motifs, ≤28 SAM boxes at 512px, CV residual spawn (not glue). Override with `HARD_BUSY_BOX_CAP`, `HARD_BUSY_GEMINI_PASSES`, `HARD_SAM_MAX_SIDE`.

---

## What “works how it should” means

| Behavior | How |
|----------|-----|
| Extract motifs | CV connected components + defringe (default) |
| Proposal, not final | `scene.json` kind=`auto_layer_proposal` |
| High vs uncertain | `rank_score` + `confidence_tier`; high listed first; uncertain collapsed in UI |
| Residual never silent | `residual_ink.png` + locked `scene.residual` + baked into `base.png` |
| Original preserved | `original.png` always |
| Review | accept / reject / merge fragments |
| Durable accept | `POST /api/documents/.../accept` → Layers Menu document |
| Rerun = new proposal | attach new proposal; **does not** wipe `document.layers` |
| Handoff pack | ZIP: `motifs/*.png` + `motifs.json` (+ base) |

---

## Output contract (`scene.json`)

```json
{
  "version": 2,
  "kind": "auto_layer_proposal",
  "proposal_id": "...",
  "original": "original.png",
  "background": "background.png",
  "base": "base.png",
  "residual": {
    "id": "residual_sheet",
    "kind": "residual_sheet",
    "src": "residual_ink.png",
    "locked": true,
    "status": "kept_in_base",
    "residual_frac": 0.01
  },
  "layers": [
    {
      "id": "m001",
      "src": "layers/m001.png",
      "bbox_px": [x, y, w, h],
      "confidence": 0.8,
      "matte_score": 0.6,
      "rank_score": 0.65,
      "confidence_tier": "high",
      "status": "proposed"
    }
  ],
  "stats": {
    "high_confidence": 45,
    "uncertain": 23,
    "ink_coverage": 0.99,
    "residual_frac": 0.01,
    "background_quality": "clean|approximate|unreliable",
    "partial": false
  }
}
```

---

## Layers Menu API (POC contract)

Documents live at `out/documents/<doc_id>/`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/documents` | list documents |
| `POST` | `/api/documents` | create empty `{name, width, height}` |
| `POST` | `/api/documents/from-job` | `{job_id, pipeline}` → doc + attach proposal |
| `POST` | `/api/documents/:id/attach-proposal` | attach new proposal (rerun); keeps accepted layers |
| `POST` | `/api/documents/:id/accept` | `{motifs:[...], job_id?, pipeline?}` → durable Layers Menu |
| `POST` | `/api/documents/:id/remove` | `{layer_ids:[...]}` |
| `GET` | `/api/documents/:id/layers-menu` | product-shaped menu payload |
| `GET` | `/api/documents/:id/motif-pack` | download ZIP |
| `GET` | `/documents/:id/` | visual Layers Menu |

**Rule:** proposal ≠ Layers Menu. Accept copies assets into `documents/<id>/motifs/`.

### Motif pack (`motifs.json`)

```json
{
  "version": 1,
  "kind": "motif_pack",
  "canvas": {"width": 1024, "height": 1024},
  "motifs": [
    {
      "id": "m001",
      "asset": "motifs/m001.png",
      "bbox_px": [x, y, w, h],
      "transform": {"x": 0, "y": 0, "width": 100, "height": 100, "rotation": 0},
      "confidence_tier": "high"
    }
  ]
}
```

---

## Pipelines

| Mode | Use |
|------|-----|
| **CV** (default for clean) | Fast; ship/demo path |
| **Hard route** (`soft` / `camo` / `busy`) | VLM boxes → SAM2 → soft alpha |
| SAM2 + QA | Cleaner masks on CV path; slow on CPU |

```bash
# Auto-route (classifies print type)
.venv/bin/python src/run_auto_layer.py -i fixtures/hard/h17_camouflage.png -o out/demo --route auto

# Force hard camo path
.venv/bin/python src/run_auto_layer.py -i fixtures/hard/h17_camouflage.png -o out/demo --route camo

# Legacy CV only
.venv/bin/python src/run_auto_layer.py -i fixtures/06-seashells.png -o out/demo --route off --no-sam --no-qa
```

Bakeoff gallery: `out/hard_route_bakeoff/compare.html`

### Hard-route findings (honest)
Bakeoff v2 (`out/hard_route_bakeoff_v2/`):

| Type | Result |
|------|--------|
| **soft** | CV + interior feather — **parity** with CV (F1 ~0.995, cov ~99%) |
| **busy** | VLM→SAM motifs + CV residual fill — **win**: far fewer layers, higher coverage, more high-tier |
| **camo** | CV primary (stable bg) — **parity** with CV; contrast densify reserved for missed regions |

Auto-route: clean/soft-on-clear-ground → CV; camo → camo path; paisley-like → busy.

---

## Fixtures

- `fixtures/*.png` — good prints (seashells, ditsy, paisley, …)
- `fixtures/showcase/` — 15 Gemini motifs @ 2048
- `fixtures/hard/` — stress (watercolor, camouflage, micro ditsy, …)

See [KNOWN_LIMITS.md](KNOWN_LIMITS.md).

---

## Handoff folder

After smoke:

```
handoff/                     # scripts/build_handoff.py
  README.md                  # this file (copy)
  KNOWN_LIMITS.md
  sample_scene.json
  sample_motifs.json
  sample_layers_menu.json
  CONTRACT.md                # API + schema cheat sheet
```

```bash
.venv/bin/python scripts/build_handoff.py
```

---

## Layout

```
auto-layer-poc/
  app.py                 # server :8787
  src/
    run_auto_layer.py    # pipeline entry
    segment.py / isolate_v2.py / scene.py
    layers_menu.py       # durable document + pack
  web/
    editor.html/.js      # proposal review
    layers_menu.html     # accepted Layers Menu UI
  scripts/smoke_handoff.py
  fixtures/
```
