"""Layers Menu document model — durable accepted motifs vs Auto Layer proposals.

This is the POC contract your teammate can mirror in product:

  proposal  = ephemeral Auto Layer output (scene.json)
  document  = durable Print Studio–style state (accepted layers + base)

Rerun Auto Layer → new proposal_id. Never clobber document.layers.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any


DOC_ROOT_NAME = "documents"


def documents_root(out_root: Path) -> Path:
    root = out_root / DOC_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _empty_document(
    doc_id: str,
    *,
    name: str,
    width: int = 0,
    height: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "layers_menu_document",
        "id": doc_id,
        "name": name,
        "created_at": _now(),
        "updated_at": _now(),
        "canvas": {"width": width, "height": height},
        "source": source,
        # Flattened original always preserved
        "original": None,
        # Working base (bg + residual ink kept in base)
        "base": None,
        "background": None,
        "residual": None,
        # Durable accepted motif layers (Layers Menu)
        "layers": [],
        # Latest attached proposal (review only — not Layers Menu)
        "active_proposal": None,
        "proposal_history": [],
        "stats": {
            "accepted": 0,
            "last_import_at": None,
        },
    }


def create_document(
    out_root: Path,
    *,
    name: str = "Untitled print",
    width: int = 0,
    height: int = 0,
    source: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    root = documents_root(out_root)
    doc_id = doc_id or uuid.uuid4().hex[:12]
    doc_dir = root / doc_id
    if doc_dir.exists():
        raise FileExistsError(f"document {doc_id} already exists")
    (doc_dir / "assets").mkdir(parents=True)
    (doc_dir / "motifs").mkdir(parents=True)
    doc = _empty_document(doc_id, name=name, width=width, height=height, source=source)
    save_document(out_root, doc)
    return doc


def doc_dir(out_root: Path, doc_id: str) -> Path:
    return documents_root(out_root) / doc_id


def load_document(out_root: Path, doc_id: str) -> dict[str, Any] | None:
    path = doc_dir(out_root, doc_id) / "document.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def save_document(out_root: Path, doc: dict[str, Any]) -> Path:
    d = doc_dir(out_root, doc["id"])
    d.mkdir(parents=True, exist_ok=True)
    doc["updated_at"] = _now()
    doc["stats"] = doc.get("stats") or {}
    doc["stats"]["accepted"] = len(doc.get("layers") or [])
    path = d / "document.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def list_documents(out_root: Path) -> list[dict[str, Any]]:
    root = documents_root(out_root)
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        doc = load_document(out_root, p.name)
        if not doc:
            continue
        out.append(
            {
                "id": doc["id"],
                "name": doc.get("name"),
                "source": doc.get("source"),
                "accepted": len(doc.get("layers") or []),
                "updated_at": doc.get("updated_at"),
                "active_proposal": (doc.get("active_proposal") or {}).get("proposal_id"),
                "url": f"/documents/{doc['id']}/",
                "layers_menu": f"/api/documents/{doc['id']}/layers-menu",
            }
        )
    return out


def _copy_asset(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def attach_proposal_from_scene(
    out_root: Path,
    doc_id: str,
    scene_dir: Path,
    *,
    pipeline: str = "cv",
) -> dict[str, Any]:
    """Attach an Auto Layer proposal to a document without touching accepted layers."""
    doc = load_document(out_root, doc_id)
    if doc is None:
        raise FileNotFoundError(f"unknown document {doc_id}")

    scene_path = scene_dir / "scene.json"
    if not scene_path.is_file():
        raise FileNotFoundError(f"no scene.json in {scene_dir}")
    scene = json.loads(scene_path.read_text())

    d = doc_dir(out_root, doc_id)
    prop_id = str(scene.get("proposal_id") or scene_dir.name)
    prop_assets = d / "proposals" / prop_id
    if prop_assets.exists():
        shutil.rmtree(prop_assets)
    prop_assets.mkdir(parents=True)

    # Copy canvas assets into document (original/base always refreshed from proposal;
    # accepted motif layers are NOT overwritten)
    for key, filename in (
        ("original", "original.png"),
        ("base", "base.png"),
        ("background", "background.png"),
        ("residual", "residual_ink.png"),
    ):
        src = scene_dir / filename
        if src.is_file():
            dest_name = filename
            _copy_asset(src, d / "assets" / dest_name)
            if key == "residual":
                doc["residual"] = {
                    **(scene.get("residual") or {}),
                    "src": f"assets/{dest_name}",
                }
            else:
                doc[key] = f"assets/{dest_name}"

    # Copy layer previews for review (proposal only)
    layers_src = scene_dir / "layers"
    if layers_src.is_dir():
        dest_layers = prop_assets / "layers"
        shutil.copytree(layers_src, dest_layers)

    shutil.copy2(scene_path, prop_assets / "scene.json")

    doc["canvas"] = {
        "width": int(scene.get("width") or doc.get("canvas", {}).get("width") or 0),
        "height": int(scene.get("height") or doc.get("canvas", {}).get("height") or 0),
    }
    doc["source"] = scene.get("source") or doc.get("source")
    proposal_ref = {
        "proposal_id": prop_id,
        "pipeline": pipeline,
        "scene_dir": str(scene_dir.resolve()),
        "local_scene": f"proposals/{prop_id}/scene.json",
        "editor": f"/jobs/{scene_dir.parent.name}/{scene_dir.name}/editor.html"
        if (scene_dir.parent.name and scene_dir.name)
        else None,
        "stats": scene.get("stats"),
        "attached_at": _now(),
        "status": "open",
    }
    # Prefer editor under documents if we can serve it — keep job editor path too
    proposal_ref["document_editor"] = (
        f"/documents/{doc_id}/proposals/{prop_id}/editor.html"
    )
    # Copy editor into proposal folder for self-contained review
    web_root = Path(__file__).resolve().parent.parent / "web"
    for name in ("editor.html", "editor.js"):
        src = web_root / name
        if src.is_file():
            shutil.copy2(src, prop_assets / name)

    hist = list(doc.get("proposal_history") or [])
    hist.append({"proposal_id": prop_id, "attached_at": proposal_ref["attached_at"], "pipeline": pipeline})
    doc["proposal_history"] = hist[-20:]
    doc["active_proposal"] = proposal_ref
    save_document(out_root, doc)
    return doc


def create_document_from_scene(
    out_root: Path,
    scene_dir: Path,
    *,
    name: str | None = None,
    pipeline: str = "cv",
    doc_id: str | None = None,
) -> dict[str, Any]:
    scene = json.loads((scene_dir / "scene.json").read_text())
    doc = create_document(
        out_root,
        name=name or f"Auto Layer · {scene.get('source') or scene_dir.name}",
        width=int(scene.get("width") or 0),
        height=int(scene.get("height") or 0),
        source=scene.get("source"),
        doc_id=doc_id,
    )
    return attach_proposal_from_scene(out_root, doc["id"], scene_dir, pipeline=pipeline)


def _resolve_motif_src(scene_dir: Path, src: str) -> Path | None:
    if not src:
        return None
    if src.startswith("data:"):
        return None  # handled by caller with inline bytes
    path = (scene_dir / src).resolve()
    try:
        path.relative_to(scene_dir.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def accept_motifs(
    out_root: Path,
    doc_id: str,
    motifs: list[dict[str, Any]],
    *,
    scene_dir: Path | None = None,
    proposal_id: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Push accepted motifs into Layers Menu (durable document.layers).

    Motifs may include:
      id, label, src|asset, bbox_px, transform, confidence, confidence_tier, ...
    Asset bytes are copied into documents/<id>/motifs/<id>.png
    """
    import base64
    import re

    doc = load_document(out_root, doc_id)
    if doc is None:
        raise FileNotFoundError(f"unknown document {doc_id}")

    d = doc_dir(out_root, doc_id)
    motifs_dir = d / "motifs"
    motifs_dir.mkdir(exist_ok=True)

    # Resolve scene_dir from active proposal if needed
    if scene_dir is None and doc.get("active_proposal"):
        local = doc["active_proposal"].get("local_scene")
        if local:
            scene_dir = (d / local).parent
        elif doc["active_proposal"].get("scene_dir"):
            scene_dir = Path(doc["active_proposal"]["scene_dir"])

    existing = {L["id"]: L for L in (doc.get("layers") or [])}
    if replace:
        # Clear motif files for removed ids later
        existing = {}

    accepted_ids: list[str] = []
    for m in motifs:
        mid = str(m.get("id") or f"m{uuid.uuid4().hex[:8]}")
        dest = motifs_dir / f"{mid}.png"
        src = m.get("src") or m.get("asset") or ""
        wrote = False
        if isinstance(src, str) and src.startswith("data:image"):
            # data:image/png;base64,...
            match = re.match(r"data:image/[^;]+;base64,(.+)", src, re.DOTALL)
            if match:
                dest.write_bytes(base64.b64decode(match.group(1)))
                wrote = True
        elif scene_dir is not None:
            src_path = _resolve_motif_src(scene_dir, src)
            if src_path is not None:
                _copy_asset(src_path, dest)
                wrote = True
            else:
                # Try motifs/ or layers/ by id
                for candidate in (
                    scene_dir / "layers" / f"{mid}.png",
                    scene_dir / src,
                ):
                    if candidate.is_file():
                        _copy_asset(candidate, dest)
                        wrote = True
                        break

        if not wrote and not dest.is_file():
            raise FileNotFoundError(f"cannot resolve asset for motif {mid} (src={src!r})")

        bbox = m.get("bbox_px") or m.get("transform") and [
            m["transform"].get("x", 0),
            m["transform"].get("y", 0),
            m["transform"].get("width", 0),
            m["transform"].get("height", 0),
        ]
        transform = m.get("transform") or {}
        if bbox and len(bbox) == 4 and not transform:
            transform = {
                "x": bbox[0],
                "y": bbox[1],
                "width": bbox[2],
                "height": bbox[3],
                "rotation": float(m.get("rotation") or 0),
            }

        layer = {
            "id": mid,
            "label": m.get("label") or "motif",
            "asset": f"motifs/{mid}.png",
            "bbox_px": list(bbox) if bbox else [0, 0, 0, 0],
            "transform": transform,
            "confidence": m.get("confidence"),
            "confidence_tier": m.get("confidence_tier"),
            "matte_score": m.get("matte_score"),
            "hue": m.get("hue", 0),
            "saturation": m.get("saturation", 100),
            "status": "accepted",
            "proposal_id": proposal_id
            or (doc.get("active_proposal") or {}).get("proposal_id"),
            "accepted_at": _now(),
            "method": m.get("method"),
        }
        existing[mid] = layer
        accepted_ids.append(mid)

    doc["layers"] = list(existing.values())
    # Stable order: by y then x
    doc["layers"].sort(
        key=lambda L: (
            int((L.get("transform") or {}).get("y") or (L.get("bbox_px") or [0, 0])[1]),
            int((L.get("transform") or {}).get("x") or (L.get("bbox_px") or [0])[0]),
        )
    )
    doc["stats"]["last_import_at"] = _now()
    if doc.get("active_proposal"):
        doc["active_proposal"]["status"] = "reviewed"
    save_document(out_root, doc)
    return {"document": doc, "accepted_ids": accepted_ids}


def remove_layers(out_root: Path, doc_id: str, layer_ids: list[str]) -> dict[str, Any]:
    doc = load_document(out_root, doc_id)
    if doc is None:
        raise FileNotFoundError(f"unknown document {doc_id}")
    remove = set(layer_ids)
    d = doc_dir(out_root, doc_id)
    kept = []
    for L in doc.get("layers") or []:
        if L["id"] in remove:
            asset = d / L.get("asset", "")
            if asset.is_file():
                asset.unlink()
        else:
            kept.append(L)
    doc["layers"] = kept
    save_document(out_root, doc)
    return doc


def layers_menu_view(out_root: Path, doc_id: str) -> dict[str, Any]:
    """Product-shaped Layers Menu payload."""
    doc = load_document(out_root, doc_id)
    if doc is None:
        raise FileNotFoundError(f"unknown document {doc_id}")
    base = f"/documents/{doc_id}"
    layers = []
    for L in doc.get("layers") or []:
        layers.append(
            {
                **L,
                "url": f"{base}/{L['asset']}",
            }
        )
    residual = doc.get("residual")
    if residual and residual.get("src"):
        residual = {**residual, "url": f"{base}/{residual['src']}"}
    return {
        "kind": "layers_menu",
        "document_id": doc_id,
        "name": doc.get("name"),
        "canvas": doc.get("canvas"),
        "original": f"{base}/{doc['original']}" if doc.get("original") else None,
        "base": f"{base}/{doc['base']}" if doc.get("base") else None,
        "background": f"{base}/{doc['background']}" if doc.get("background") else None,
        "residual": residual,
        "layers": layers,
        "active_proposal": doc.get("active_proposal"),
        "count": len(layers),
    }


def build_motif_pack_zip(out_root: Path, doc_id: str, zip_path: Path) -> Path:
    doc = load_document(out_root, doc_id)
    if doc is None:
        raise FileNotFoundError(f"unknown document {doc_id}")
    d = doc_dir(out_root, doc_id)
    pack = {
        "version": 1,
        "kind": "motif_pack",
        "document_id": doc_id,
        "source": doc.get("source"),
        "canvas": doc.get("canvas"),
        "base": "base.png" if doc.get("base") else None,
        "background": "background.png" if doc.get("background") else None,
        "original": "original.png" if doc.get("original") else None,
        "residual": doc.get("residual"),
        "motifs": [
            {
                "id": L["id"],
                "label": L.get("label"),
                "asset": f"motifs/{L['id']}.png",
                "bbox_px": L.get("bbox_px"),
                "transform": L.get("transform"),
                "confidence": L.get("confidence"),
                "confidence_tier": L.get("confidence_tier"),
                "matte_score": L.get("matte_score"),
                "hue": L.get("hue"),
                "saturation": L.get("saturation"),
            }
            for L in (doc.get("layers") or [])
        ],
        "note": "Layers Menu / Pattern Placement handoff pack",
    }
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("motifs.json", json.dumps(pack, indent=2))
        for key, name in (
            ("base", "base.png"),
            ("background", "background.png"),
            ("original", "original.png"),
        ):
            if doc.get(key):
                src = d / doc[key]
                if src.is_file():
                    zf.write(src, name)
        for L in doc.get("layers") or []:
            src = d / L["asset"]
            if src.is_file():
                zf.write(src, f"motifs/{L['id']}.png")
    return zip_path
