/* Auto Layer review canvas — proposal → Layers Menu handoff (Konva) */
(async function () {
  const $ = (id) => document.getElementById(id);

  if (typeof Konva === "undefined") {
    const banner = $("banner");
    if (banner) {
      banner.hidden = false;
      banner.className = "banner bad";
      banner.textContent =
        "Review failed to load Konva (CDN blocked). Try refresh, or open via the local app at :8787.";
    }
    console.error("Konva missing");
    return;
  }

  const sceneUrl = new URLSearchParams(location.search).get("scene") || "scene.json";
  const qs = new URLSearchParams(location.search);
  let documentId = qs.get("document_id") || qs.get("doc") || null;
  let scene;
  try {
    const r = await fetch(sceneUrl);
    if (!r.ok) throw new Error(`Failed to load ${sceneUrl} (${r.status})`);
    scene = await r.json();
  } catch (err) {
    const banner = $("banner");
    if (banner) {
      banner.hidden = false;
      banner.className = "banner bad";
      banner.textContent = String(err);
    }
    console.error(err);
    return;
  }

  const stats = scene.stats || {};
  const bgQuality = stats.background_quality || "approximate";
  const residualMeta = scene.residual || null;
  $("meta").textContent =
    `${scene.source || "print"} · ${scene.layers.length} proposed · ` +
    `${stats.high_confidence ?? "?"} high / ${stats.uncertain ?? "?"} uncertain · ` +
    `${scene.width}×${scene.height} · loading layers…`;

  const banner = $("banner");
  if (bgQuality === "unreliable") {
    banner.hidden = false;
    banner.className = "banner bad";
    banner.textContent =
      "Background reconstruction looks unreliable — holes/ghosts may remain. " +
      "Use Original fallback or keep residual ink in the working base.";
  } else if (stats.partial || bgQuality === "approximate") {
    banner.hidden = false;
    banner.className = "banner";
    banner.textContent =
      `Partial result: ${(stats.residual_frac * 100 || 0).toFixed(1)}% ink left in base · ` +
      `bg quality=${bgQuality}. Reject false positives; merge fragments before accepting.`;
  } else {
    banner.hidden = false;
    banner.className = "banner ok";
    banner.textContent = "High-confidence extraction. Review uncertain items before accepting.";
  }

  const stageW = scene.width;
  const stageH = scene.height;
  const stage = new Konva.Stage({ container: "container", width: stageW, height: stageH });
  const bgLayer = new Konva.Layer();
  const residualLayer = new Konva.Layer();
  const motifLayer = new Konva.Layer();
  const uiLayer = new Konva.Layer();
  stage.add(bgLayer);
  stage.add(residualLayer);
  stage.add(motifLayer);
  stage.add(uiLayer);

  const transformer = new Konva.Transformer({
    rotateEnabled: true,
    boundBoxFunc: (oldBox, newBox) => {
      if (newBox.width < 8 || newBox.height < 8) return oldBox;
      return newBox;
    },
  });
  uiLayer.add(transformer);

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  async function tryLoadImage(src) {
    try {
      return await loadImage(src);
    } catch {
      return null;
    }
  }

  const baseSrc = scene.base || scene.background;
  const residualSrc = residualMeta && residualMeta.src ? residualMeta.src : "residual_ink.png";
  const [baseImg, origImg, bgOnlyImg, residualImg] = await Promise.all([
    loadImage(baseSrc),
    loadImage(scene.original || scene.background),
    loadImage(scene.background),
    tryLoadImage(residualSrc),
  ]);

  const bgNode = new Konva.Image({
    image: baseImg,
    x: 0,
    y: 0,
    width: stageW,
    height: stageH,
    listening: false,
    name: "background",
  });
  bgLayer.add(bgNode);
  bgLayer.draw();

  let residualNode = null;
  let showResidual = false;
  if (residualImg && residualMeta && !residualMeta.empty) {
    residualNode = new Konva.Image({
      image: residualImg,
      x: 0,
      y: 0,
      width: stageW,
      height: stageH,
      listening: false,
      name: "residual_sheet",
      opacity: 1,
      visible: false,
    });
    residualLayer.add(residualNode);
    residualLayer.draw();
    showResidual = true;
  }

  let view = "base";
  function setView(next) {
    view = next;
    document.querySelectorAll("#viewMode button").forEach((b) => {
      b.classList.toggle("on", b.dataset.view === next);
    });
    if (next === "original") bgNode.image(origImg);
    else if (next === "bg") bgNode.image(bgOnlyImg);
    else if (next === "residual") {
      // Clean bg + residual sheet so leftover ink is visible as its own layer
      bgNode.image(bgOnlyImg);
    } else bgNode.image(baseImg);

    if (residualNode) {
      // Overlay residual only when inspecting residual / bg (base already bakes it in)
      residualNode.visible(next === "residual" || (next === "bg" && showResidual));
    }
    bgLayer.batchDraw();
    residualLayer.batchDraw();
  }
  document.querySelectorAll("#viewMode button").forEach((b) => {
    b.addEventListener("click", () => setView(b.dataset.view));
  });

  /** @type {{node: Konva.Image, meta: object, hue: number, sat: number}[]} */
  const items = [];
  /** @type {Set<object>} */
  const multi = new Set();
  let selected = null;
  let idSeq = scene.layers.length;
  let uncertainOpen = !(scene.review && scene.review.uncertain_collapsed_by_default);

  function applyFilters(item) {
    const node = item.node;
    node.cache();
    node.filters([Konva.Filters.HSV]);
    node.hue(((item.hue % 360) + 360) % 360);
    node.saturation((item.sat - 100) / 100);
    node.value(0);
    motifLayer.batchDraw();
  }

  function selectItem(item, { additive = false } = {}) {
    if (!additive) multi.clear();
    selected = item;
    if (item) multi.add(item);
    if (!item || multi.size === 0) {
      transformer.nodes([]);
      uiLayer.draw();
      renderList();
      return;
    }
    transformer.nodes([...multi].map((i) => i.node));
    uiLayer.draw();
    if (multi.size === 1) {
      $("hue").value = String(item.hue);
      $("hueVal").textContent = `${item.hue}°`;
      $("sat").value = String(item.sat);
      $("satVal").textContent = `${item.sat}%`;
    }
    renderList();
  }

  function renderList() {
    const ul = $("layerList");
    ul.innerHTML = "";
    const accepted = items.filter((i) => i.meta.status === "accepted").length;
    const proposed = items.filter((i) => i.meta.status === "proposed").length;
    const rejected = items.filter((i) => i.meta.status === "rejected").length;
    $("layerStats").textContent = `(${accepted} accepted · ${proposed} proposed · ${rejected} rejected)`;

    // Locked residual sheet (not a motif — kept in base)
    if (residualMeta) {
      const li = document.createElement("li");
      li.className = "locked";
      const thumb = document.createElement("img");
      thumb.src = residualMeta.src || residualSrc;
      thumb.alt = "";
      const label = document.createElement("span");
      const pct = ((residualMeta.residual_frac || stats.residual_frac || 0) * 100).toFixed(1);
      label.textContent = residualMeta.empty
        ? "Residual · empty"
        : `Residual · ${pct}% kept in base`;
      const tier = document.createElement("span");
      tier.className = "tier locked";
      tier.textContent = "locked";
      li.append(thumb, label, tier);
      li.addEventListener("click", () => setView("residual"));
      ul.appendChild(li);
    }

    const high = items.filter((i) => i.meta.confidence_tier === "high" || i.meta.status === "accepted");
    const unc = items.filter(
      (i) => i.meta.confidence_tier !== "high" && i.meta.status !== "accepted"
    );

    function appendItem(item) {
      const li = document.createElement("li");
      if (item === selected || multi.has(item)) li.classList.add("active");
      if (item.meta.status === "rejected") li.classList.add("rejected");
      const thumb = document.createElement("img");
      thumb.src = item.meta.src;
      thumb.alt = "";
      const label = document.createElement("span");
      label.textContent = `${item.meta.id} · ${item.meta.label || "motif"}`;
      if (!item.node.visible()) label.style.opacity = "0.45";
      const tier = document.createElement("span");
      const t =
        item.meta.status === "accepted"
          ? "accepted"
          : item.meta.confidence_tier || "uncertain";
      tier.className =
        "tier" + (t === "uncertain" ? " uncertain" : t === "accepted" ? " accepted" : "");
      tier.textContent = t;
      li.append(thumb, label, tier);
      li.addEventListener("click", (e) => selectItem(item, { additive: e.shiftKey }));
      ul.appendChild(li);
    }

    if (high.length) {
      const hdr = document.createElement("li");
      hdr.className = "section";
      hdr.textContent = `High confidence (${high.length})`;
      ul.appendChild(hdr);
      high.forEach(appendItem);
    }

    if (unc.length) {
      const hdr = document.createElement("li");
      hdr.className = "section toggle";
      hdr.textContent = `Uncertain (${unc.length}) ${uncertainOpen ? "▾" : "▸"}`;
      hdr.addEventListener("click", () => {
        uncertainOpen = !uncertainOpen;
        renderList();
      });
      ul.appendChild(hdr);
      if (uncertainOpen) unc.forEach(appendItem);
    }
  }

  async function addLayer(meta, opts = {}) {
    const img = await loadImage(meta.src);
    const [x, y, w, h] = meta.bbox_px || [0, 0, img.width, img.height];
    const node = new Konva.Image({
      image: img,
      x: opts.x ?? x,
      y: opts.y ?? y,
      width: opts.width ?? w,
      height: opts.height ?? h,
      rotation: opts.rotation ?? 0,
      draggable: true,
      name: meta.id,
      opacity: meta.status === "rejected" ? 0.25 : 1,
    });
    node.on("click tap", (e) => {
      const item = items.find((i) => i.node === node);
      selectItem(item, { additive: e.evt.shiftKey });
    });
    node.on("dragend transformend", () => motifLayer.batchDraw());
    motifLayer.add(node);
    const item = {
      node,
      meta: {
        status: "proposed",
        confidence_tier: "uncertain",
        ...meta,
      },
      hue: opts.hue ?? 0,
      sat: opts.sat ?? 100,
    };
    if (item.meta.status === "rejected") node.visible(false);
    items.push(item);
    if (item.hue !== 0 || item.sat !== 100) applyFilters(item);
    return item;
  }

  // Scene layers arrive already sorted high-first
  const total = (scene.layers || []).length;
  for (let i = 0; i < total; i++) {
    await addLayer(scene.layers[i]);
    if (i % 20 === 19 || i === total - 1) {
      $("meta").textContent =
        `${scene.source || "print"} · loaded ${i + 1}/${total} · ` +
        `${stats.high_confidence ?? "?"} high / ${stats.uncertain ?? "?"} uncertain`;
    }
  }
  motifLayer.draw();
  renderList();
  $("meta").textContent =
    `${scene.source || "print"} · ${items.length} proposed · ` +
    `${stats.high_confidence ?? "?"} high / ${stats.uncertain ?? "?"} uncertain · ` +
    `${scene.width}×${scene.height}`;

  stage.on("click tap", (e) => {
    if (e.target === stage || e.target.name() === "background") selectItem(null);
  });

  $("hue").addEventListener("input", () => {
    if (!selected) return;
    selected.hue = Number($("hue").value);
    $("hueVal").textContent = `${selected.hue}°`;
    applyFilters(selected);
  });
  $("sat").addEventListener("input", () => {
    if (!selected) return;
    selected.sat = Number($("sat").value);
    $("satVal").textContent = `${selected.sat}%`;
    applyFilters(selected);
  });
  $("btnReset").addEventListener("click", () => {
    if (!selected) return;
    selected.hue = 0;
    selected.sat = 100;
    $("hue").value = "0";
    $("sat").value = "100";
    $("hueVal").textContent = "0°";
    $("satVal").textContent = "100%";
    selected.node.clearCache();
    selected.node.filters([]);
    motifLayer.batchDraw();
  });
  $("btnHide").addEventListener("click", () => {
    if (!selected) return;
    selected.node.visible(!selected.node.visible());
    if (!selected.node.visible()) selectItem(null);
    else {
      motifLayer.batchDraw();
      renderList();
    }
  });

  function acceptItems(list) {
    list.forEach((item) => {
      if (item.meta.status === "rejected") return;
      item.meta.status = "accepted";
      item.node.opacity(1);
      item.node.visible(true);
    });
    renderList();
    motifLayer.batchDraw();
  }
  $("btnAccept").addEventListener("click", () => {
    const list = multi.size ? [...multi] : selected ? [selected] : [];
    acceptItems(list);
  });
  $("btnAcceptHigh").addEventListener("click", () => {
    acceptItems(items.filter((i) => i.meta.confidence_tier === "high"));
  });
  $("btnReject").addEventListener("click", () => {
    const list = multi.size ? [...multi] : selected ? [selected] : [];
    list.forEach((item) => {
      item.meta.status = "rejected";
      item.node.visible(false);
      item.node.opacity(0.25);
    });
    selectItem(null);
    motifLayer.batchDraw();
    renderList();
  });

  $("btnMerge").addEventListener("click", async () => {
    const list = [...multi];
    if (list.length < 2) {
      alert("Shift-click 2+ fragments to merge.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = stageW;
    canvas.height = stageH;
    const ctx = canvas.getContext("2d");
    let minX = Infinity,
      minY = Infinity,
      maxX = 0,
      maxY = 0;
    for (const item of list) {
      const n = item.node;
      const box = n.getClientRect({ relativeTo: stage });
      minX = Math.min(minX, box.x);
      minY = Math.min(minY, box.y);
      maxX = Math.max(maxX, box.x + box.width);
      maxY = Math.max(maxY, box.y + box.height);
      const img = n.image();
      ctx.save();
      ctx.translate(n.x(), n.y());
      ctx.rotate((n.rotation() * Math.PI) / 180);
      ctx.drawImage(img, 0, 0, n.width() * n.scaleX(), n.height() * n.scaleY());
      ctx.restore();
    }
    minX = Math.max(0, Math.floor(minX));
    minY = Math.max(0, Math.floor(minY));
    maxX = Math.min(stageW, Math.ceil(maxX));
    maxY = Math.min(stageH, Math.ceil(maxY));
    const tw = Math.max(1, maxX - minX);
    const th = Math.max(1, maxY - minY);
    const crop = document.createElement("canvas");
    crop.width = tw;
    crop.height = th;
    crop.getContext("2d").drawImage(canvas, minX, minY, tw, th, 0, 0, tw, th);
    const dataUrl = crop.toDataURL("image/png");

    list.forEach((item) => {
      item.meta.status = "rejected";
      item.node.visible(false);
      item.node.destroy();
      const idx = items.indexOf(item);
      if (idx >= 0) items.splice(idx, 1);
    });

    idSeq += 1;
    const meta = {
      id: `merge${idSeq}`,
      label: "merged motif",
      src: dataUrl,
      bbox_px: [minX, minY, tw, th],
      confidence: 0.8,
      matte_score: 0.8,
      confidence_tier: "high",
      rank_score: 0.85,
      status: "proposed",
      method: "user_merge",
    };
    const item = await addLayer(meta);
    motifLayer.draw();
    selectItem(item);
  });

  $("btnDup").addEventListener("click", async () => {
    if (!selected) return;
    idSeq += 1;
    const src = selected.meta;
    const node = selected.node;
    const meta = {
      ...src,
      id: `copy${idSeq}`,
      label: `${src.label || "motif"} (copy)`,
      status: "proposed",
    };
    const item = await addLayer(meta, {
      x: node.x() + 16,
      y: node.y() + 16,
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
      rotation: node.rotation(),
      hue: selected.hue,
      sat: selected.sat,
    });
    motifLayer.draw();
    selectItem(item);
  });

  function acceptedMotifs() {
    return items
      .filter((i) => i.meta.status === "accepted")
      .map((i) => {
        const n = i.node;
        const x = Math.round(n.x());
        const y = Math.round(n.y());
        const w = Math.round(n.width() * n.scaleX());
        const h = Math.round(n.height() * n.scaleY());
        return {
          id: i.meta.id,
          label: i.meta.label || "motif",
          asset: `motifs/${i.meta.id}.png`,
          src: i.meta.src,
          bbox_px: [x, y, w, h],
          transform: {
            x,
            y,
            width: w,
            height: h,
            rotation: n.rotation(),
          },
          confidence: i.meta.confidence,
          confidence_tier: i.meta.confidence_tier,
          matte_score: i.meta.matte_score,
          hue: i.hue,
          saturation: i.sat,
          method: i.meta.method,
        };
      });
  }

  $("btnExport").addEventListener("click", () => {
    const accepted = acceptedMotifs().map(({ src, ...rest }) => ({
      ...rest,
      src: undefined,
      asset: rest.asset,
      status: "accepted",
    }));
    // Strip undefined
    const clean = accepted.map((m) => {
      const o = { ...m };
      delete o.src;
      return o;
    });
    const payload = {
      version: 2,
      kind: "auto_layer_accepted",
      source: scene.source,
      proposal_id: scene.proposal_id,
      original: scene.original,
      base: scene.base,
      background: scene.background,
      residual: scene.residual,
      width: scene.width,
      height: scene.height,
      layers: clean,
      note: "Accepted motifs ready for Layers Menu / Pattern Placement handoff",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `accepted_${scene.proposal_id || "layers"}.json`;
    a.click();
  });

  async function fetchAsBlob(url) {
    if (url.startsWith("data:")) {
      const res = await fetch(url);
      return res.blob();
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}`);
    return res.blob();
  }

  $("btnMotifPack").addEventListener("click", async () => {
    const motifs = acceptedMotifs();
    if (!motifs.length) {
      alert("Accept at least one motif before exporting a motif pack.");
      return;
    }
    if (typeof JSZip === "undefined") {
      alert("JSZip failed to load — check network / CDN.");
      return;
    }
    const btn = $("btnMotifPack");
    btn.disabled = true;
    btn.textContent = "Packing…";
    try {
      const zip = new JSZip();
      const folder = zip.folder("motifs");
      const manifestMotifs = [];
      for (const m of motifs) {
        const blob = await fetchAsBlob(m.src);
        folder.file(`${m.id}.png`, blob);
        manifestMotifs.push({
          id: m.id,
          label: m.label,
          asset: m.asset,
          bbox_px: m.bbox_px,
          transform: m.transform,
          confidence: m.confidence,
          confidence_tier: m.confidence_tier,
          matte_score: m.matte_score,
          hue: m.hue,
          saturation: m.saturation,
        });
      }
      const pack = {
        version: 1,
        kind: "motif_pack",
        source: scene.source,
        proposal_id: scene.proposal_id,
        canvas: { width: scene.width, height: scene.height },
        base: scene.base,
        background: scene.background,
        original: scene.original,
        residual: residualMeta
          ? {
              src: residualMeta.src,
              residual_frac: residualMeta.residual_frac,
              status: "kept_in_base",
            }
          : null,
        motifs: manifestMotifs,
        note: "Drop into Layers Menu / Pattern Placement — assets under motifs/",
      };
      zip.file("motifs.json", JSON.stringify(pack, null, 2));
      // Include working base for placement context
      try {
        zip.file("base.png", await fetchAsBlob(scene.base || "base.png"));
      } catch {
        /* optional */
      }
      const out = await zip.generateAsync({ type: "blob" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(out);
      a.download = `motif_pack_${scene.proposal_id || "layers"}.zip`;
      a.click();
    } catch (err) {
      console.error(err);
      alert(String(err));
    } finally {
      btn.disabled = false;
      btn.textContent = "Download motif pack";
    }
  });

  function setDocStatus(text, href) {
    const el = $("docStatus");
    if (!el) return;
    if (href) {
      el.innerHTML = `${text} · <a href="${href}" target="_blank" rel="noopener">open Layers Menu</a>`;
    } else {
      el.textContent = text;
    }
  }

  if (documentId) setDocStatus(`Document ${documentId}`);

  $("btnPushLayers").addEventListener("click", async () => {
    const motifs = acceptedMotifs();
    if (!motifs.length) {
      alert("Accept at least one motif first (or Accept all high).");
      return;
    }
    const btn = $("btnPushLayers");
    btn.disabled = true;
    btn.textContent = "Pushing…";
    try {
      const parts = location.pathname.split("/").filter(Boolean);
      let jobId = null;
      let pipeline = "cv";
      const ji = parts.indexOf("jobs");
      if (ji >= 0 && parts.length >= ji + 3) {
        jobId = parts[ji + 1];
        pipeline = parts[ji + 2];
      }

      if (!documentId) {
        if (!jobId) {
          throw new Error("Open review from a job URL, or pass ?document_id=…");
        }
        const createRes = await fetch("/api/documents/from-job", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            job_id: jobId,
            pipeline,
            name: `Auto Layer · ${scene.source || scene.proposal_id}`,
          }),
        });
        const data = await createRes.json();
        if (!createRes.ok) throw new Error(data.error || createRes.statusText);
        documentId = data.id;
      }

      const acceptRes = await fetch(`/api/documents/${documentId}/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          motifs: motifs.map((m) => ({
            id: m.id,
            label: m.label,
            src: m.src,
            asset: m.asset,
            bbox_px: m.bbox_px,
            transform: m.transform,
            confidence: m.confidence,
            confidence_tier: m.confidence_tier,
            matte_score: m.matte_score,
            hue: m.hue,
            saturation: m.saturation,
            method: m.method,
          })),
          proposal_id: scene.proposal_id,
          job_id: jobId,
          pipeline,
        }),
      });
      const acceptData = await acceptRes.json();
      if (!acceptRes.ok) throw new Error(acceptData.error || acceptRes.statusText);
      setDocStatus(
        `Pushed ${acceptData.accepted_ids?.length || motifs.length} motifs`,
        `/documents/${documentId}/`
      );
      const u = new URL(location.href);
      u.searchParams.set("document_id", documentId);
      history.replaceState(null, "", u.toString());
    } catch (e) {
      console.error(e);
      alert(String(e));
    } finally {
      btn.disabled = false;
      btn.textContent = "Push accepted → Layers Menu";
    }
  });
})().catch((err) => {
  document.getElementById("meta").textContent = String(err);
  console.error(err);
});
