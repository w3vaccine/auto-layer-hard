# Known limits (Auto Layer POC)

Honest failure modes for teammate integration. Partial success is OK when residual is visible.

## Works well
- Separated / lightly overlapping motifs on flat grounds (seashells, coastal shells, berries, geo)
- Solid ink vs clear ground → high tier trustworthy enough for Accept-all-high review
- Residual sheet when leftover ink remains

## Partial / weak (expect banner + residual)
| Case | Fixture | What happens |
|------|---------|----------------|
| Soft watercolor bleed | `hard/h12_soft_watercolor.png` | Fringe merging; higher residual; many uncertain |
| Camouflage | `hard/h17_camouflage.png` | Ground ≈ motif; under-segment |
| Dense interlocking | paisley / ivy trail | Internal filigree left in base |
| Micro ditsy (200+) | `hard/h03_micro_ditsy.png` | Many tiny pieces; slow; noisy tiers |
| Low contrast | `hard/h02_low_contrast.png` | Missed / fused components |

## Hard-route status (VLM→SAM)
Implemented behind `--route auto|soft|camo|busy`. See bakeoff `out/hard_route_bakeoff/`.

| Type | Status |
|------|--------|
| clean | CV remains best (keep) |
| soft on clear ground | Prefer CV; auto-router does this |
| camo | Hard path is the right architecture; needs denser VLM seeds + eval that is not ink_map-vs-bg |
| busy | Hard path wins on *motif count / reviewability*; incomplete ink OK if residual sheet kept |

## Intentionally out of scope for this POC
- Real Print Studio Layers Menu persistence (this API is the **contract mock**)
- SAM2 as default on clean prints (too slow)
- Pantone-accurate recolor (HSV only)
- Perfect generative background inpaint without Gemini key
- Durable cross-session auth / multi-user

## Product rules encoded here
1. Never destroy `original.png`
2. Unextracted ink stays in `base` + residual sheet — never silently discard
3. Rerun Auto Layer = **new proposal**; accepted Layers Menu layers stay until user removes them
4. Background quality `unreliable` → warn; offer original fallback
