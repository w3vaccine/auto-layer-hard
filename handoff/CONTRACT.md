# Auto Layer → Layers Menu contract (cheat sheet)

## Two objects

1. **Proposal** (`scene.json`) — ephemeral Auto Layer output. Review only.
2. **Document** (`document.json`) — durable Layers Menu. Accepted motifs live here.

Rerun Auto Layer → attach new proposal. Never wipe `document.layers` on rerun.

## Accept payload

```http
POST /api/documents/:id/accept
Content-Type: application/json

{
  "proposal_id": "…",
  "job_id": "…",          // optional; resolves layer PNG paths
  "pipeline": "cv",
  "motifs": [
    {
      "id": "m001",
      "label": "motif",
      "src": "layers/m001.png",
      "bbox_px": [10, 20, 100, 120],
      "transform": {"x": 10, "y": 20, "width": 100, "height": 120, "rotation": 0},
      "confidence_tier": "high"
    }
  ]
}
```

## Layers Menu view

```http
GET /api/documents/:id/layers-menu
```

Returns base/original URLs + accepted layers with `url` for each asset.

## Motif pack

```http
GET /api/documents/:id/motif-pack
→ ZIP { motifs.json, motifs/*.png, base.png? }
```
