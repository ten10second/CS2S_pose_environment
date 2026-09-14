#!/usr/bin/env python3
"""Build an offline HTML gallery for fixed KITTI test-sequence samples."""

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


Record = Dict[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def _format_checkpoint_step(value: Any) -> str:
    if value is None:
        return "checkpoint unknown"
    try:
        step = int(value)
    except (TypeError, ValueError):
        return str(value)
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_relative_png(path_value: Any, record_id: str, field_name: str) -> str:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{record_id} missing {field_name}")
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{record_id} {field_name} must be a safe relative path: {path_value}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"{record_id} {field_name} must point to a PNG: {path_value}")
    return path_value


def _normalise_record(record: Mapping[str, Any], cfg_scales: Sequence[str]) -> Record:
    sample_id = str(record.get("sample_id", ""))
    if not sample_id:
        raise ValueError("Record is missing sample_id")
    if record.get("split") != "test2":
        raise ValueError(f"{sample_id} split must be test2")

    sources = _require_mapping(record.get("sources"), f"{sample_id}.sources")
    outputs = _require_mapping(record.get("outputs"), f"{sample_id}.outputs")
    normalised_outputs = {}
    for cfg in cfg_scales:
        normalised_outputs[cfg] = _require_relative_png(outputs.get(cfg), sample_id, f"outputs.{cfg}")

    return {
        "sample_id": sample_id,
        "split": "test2",
        "drive": str(record["drive"]),
        "frame_index": int(record["frame_index"]),
        "clip_index": int(record["clip_index"]),
        "seed": int(record["seed"]),
        "initial_noise_sha256": str(record["initial_noise_sha256"]),
        "sources": {
            "satellite": _require_relative_png(sources.get("satellite"), sample_id, "sources.satellite"),
            "lidar_overlay": _require_relative_png(
                sources.get("lidar_overlay"), sample_id, "sources.lidar_overlay"
            ),
            "gt": _require_relative_png(sources.get("gt"), sample_id, "sources.gt"),
        },
        "outputs": normalised_outputs,
    }


def _cfg_scales(metadata: Mapping[str, Any]) -> List[str]:
    scales = metadata.get("cfg_scales", ["3.0", "7.5"])
    if not isinstance(scales, Sequence) or isinstance(scales, (str, bytes)):
        raise ValueError("metadata.cfg_scales must be a list")
    result = [str(scale) for scale in scales]
    for required in ("3.0", "7.5"):
        if required not in result:
            raise ValueError(f"metadata.cfg_scales must include {required}")
    return result


def _group_records(records: Sequence[Record], frames_per_clip: int, num_clips: int) -> List[List[Record]]:
    clips: List[List[Record]] = [[] for _ in range(num_clips)]
    for record in records:
        clip_index = record["clip_index"]
        if clip_index < 0 or clip_index >= num_clips:
            raise ValueError(f"{record['sample_id']} clip_index out of range: {clip_index}")
        clips[clip_index].append(record)

    for clip_index, clip in enumerate(clips):
        clip.sort(key=lambda item: item["frame_index"])
        if len(clip) != frames_per_clip:
            raise ValueError(f"Clip {clip_index} has {len(clip)} records; expected {frames_per_clip}")
        seeds = {item["seed"] for item in clip}
        if len(seeds) != 1:
            raise ValueError(f"Clip {clip_index} must use one seed across all frames")
        for position, item in enumerate(clip):
            item["frame_position"] = position
    return clips


def _gallery_html(metadata: Mapping[str, Any], clips: Sequence[Sequence[Record]], cfg_scales: Sequence[str]) -> str:
    checkpoint_label = _format_checkpoint_step(metadata.get("checkpoint_step"))
    data = {
        "checkpointLabel": checkpoint_label,
        "checkpointStep": metadata.get("checkpoint_step"),
        "cfgScales": list(cfg_scales),
        "framesPerClip": metadata["frames_per_clip"],
        "numClips": metadata["num_clips"],
        "clips": clips,
        "selectionReport": metadata.get("selection_report", {}),
    }
    data_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
    script_json = data_json.replace("<", "\\u003c")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KITTI test2 continuous-frame comparison</title>
  <style>
    :root {{
      --border: #d8dde6;
      --ink: #111827;
      --muted: #5b6472;
      --soft: #f4f6f8;
      --accent: #1f6feb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--ink);
      font-family: Arial, "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    }}
    main {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 22px 28px 34px;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subhead {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      padding: 12px;
      border: 1px solid var(--border);
      background: #fff;
      margin-bottom: 16px;
    }}
    .controls label {{
      color: var(--muted);
      font-size: 13px;
    }}
    button, select {{
      height: 34px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      padding: 0 10px;
      border-radius: 4px;
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }}
    input[type="range"] {{
      width: min(520px, 42vw);
      accent-color: var(--accent);
    }}
    .status {{
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }}
    .panel {{
      border: 1px solid var(--border);
      background: #fff;
      min-width: 0;
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      min-height: 38px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
      font-weight: 700;
      line-height: 1.25;
    }}
    .panel-title span:last-child {{
      color: var(--muted);
      font-weight: 400;
    }}
    .image-wrap {{
      height: min(52vh, 560px);
      min-height: 320px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #fff;
      overflow: hidden;
    }}
    .image-wrap img {{
      max-width: 100%;
      max-height: 100%;
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 14px 0;
      padding: 10px 12px;
      border: 1px solid var(--border);
      background: var(--soft);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .meta strong {{
      display: block;
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .filmstrip {{
      display: grid;
      grid-template-columns: repeat(16, minmax(0, 1fr));
      gap: 6px;
      margin-top: 12px;
    }}
    .thumb {{
      border: 2px solid transparent;
      background: #fff;
      padding: 0;
      height: auto;
      cursor: pointer;
      border-radius: 0;
    }}
    .thumb.active {{
      border-color: var(--accent);
    }}
    .thumb img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border: 1px solid var(--border);
    }}
    .thumb span {{
      display: block;
      padding-top: 3px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
    }}
    @media (max-width: 980px) {{
      main {{ padding: 16px; }}
      header {{ display: block; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .meta {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filmstrip {{ grid-template-columns: repeat(8, minmax(0, 1fr)); }}
      input[type="range"] {{ width: 100%; }}
      .status {{ margin-left: 0; width: 100%; }}
    }}
    @media (max-width: 620px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .meta {{ grid-template-columns: 1fr; }}
      .image-wrap {{ height: auto; min-height: 0; aspect-ratio: 1 / 1; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>KITTI test2 连续帧对比</h1>
        <div class="subhead">固定测试集片段，逐帧独立生成；片段内固定 seed。LiDAR 叠加 GT 仅用于查看投影，GT RGB 不作为生成条件。深度由近到远：红 → 黄 → 绿 → 青 → 蓝。</div>
      </div>
      <div class="subhead">checkpoint <strong>{html.escape(checkpoint_label)}</strong></div>
    </header>

    <section class="controls" aria-label="Gallery controls">
      <label>片段
        <select id="clipSelect"></select>
      </label>
      <button id="prevButton" type="button">上一帧</button>
      <button id="playButton" type="button">播放</button>
      <button id="nextButton" type="button">下一帧</button>
      <label>帧 <span id="frameLabel">0</span>
        <input id="frameSlider" type="range" min="0" max="{int(metadata['frames_per_clip']) - 1}" value="0">
      </label>
      <button class="cfgButton active" type="button" data-cfg="3.0">CFG 3.0</button>
      <button class="cfgButton" type="button" data-cfg="7.5">CFG 7.5</button>
      <div class="status" id="statusLine"></div>
    </section>

    <section class="grid" aria-label="Frame comparison">
      <article class="panel">
        <div class="panel-title"><span>卫星</span><span>原比例</span></div>
        <div class="image-wrap"><img id="satelliteImage" alt="Satellite input"></div>
      </article>
      <article class="panel">
        <div class="panel-title"><span>LiDAR 投影到 RGB</span><span>按深度着色</span></div>
        <div class="image-wrap"><img id="lidarImage" alt="LiDAR depth overlay"></div>
      </article>
      <article class="panel">
        <div class="panel-title"><span>GT</span><span>目标图像</span></div>
        <div class="image-wrap"><img id="gtImage" alt="Ground truth"></div>
      </article>
      <article class="panel">
        <div class="panel-title"><span>CFG 结果</span><span id="cfgLabel">CFG 3.0</span></div>
        <div class="image-wrap"><img id="outputImage" alt="Generated output"></div>
      </article>
    </section>

    <section class="meta" aria-label="Current frame metadata">
      <div>split<strong id="splitMeta"></strong></div>
      <div>drive<strong id="driveMeta"></strong></div>
      <div>frame<strong id="frameMeta"></strong></div>
      <div>sample<strong id="sampleMeta"></strong></div>
      <div>clip seed<strong id="seedMeta"></strong></div>
      <div>checkpoint<strong id="checkpointMeta"></strong></div>
      <div>initial noise sha256<strong id="noiseMeta"></strong></div>
      <div>生成方式<strong>逐帧独立生成</strong></div>
    </section>

    <section class="filmstrip" id="filmstrip" aria-label="16-frame filmstrip"></section>
  </main>

  <script id="gallery-data" type="application/json">{script_json}</script>
  <script>
    const gallery = JSON.parse(document.getElementById("gallery-data").textContent);
    let clipIndex = 0;
    let frameIndex = 0;
    let cfg = gallery.cfgScales.includes("3.0") ? "3.0" : gallery.cfgScales[0];
    let timer = null;

    const els = {{
      clipSelect: document.getElementById("clipSelect"),
      frameSlider: document.getElementById("frameSlider"),
      frameLabel: document.getElementById("frameLabel"),
      statusLine: document.getElementById("statusLine"),
      playButton: document.getElementById("playButton"),
      satelliteImage: document.getElementById("satelliteImage"),
      lidarImage: document.getElementById("lidarImage"),
      gtImage: document.getElementById("gtImage"),
      outputImage: document.getElementById("outputImage"),
      cfgLabel: document.getElementById("cfgLabel"),
      splitMeta: document.getElementById("splitMeta"),
      driveMeta: document.getElementById("driveMeta"),
      frameMeta: document.getElementById("frameMeta"),
      sampleMeta: document.getElementById("sampleMeta"),
      seedMeta: document.getElementById("seedMeta"),
      checkpointMeta: document.getElementById("checkpointMeta"),
      noiseMeta: document.getElementById("noiseMeta"),
      filmstrip: document.getElementById("filmstrip"),
    }};

    function currentRecord() {{
      return gallery.clips[clipIndex][frameIndex];
    }}

    function clampFrame(value) {{
      const maxFrame = gallery.framesPerClip - 1;
      return Math.max(0, Math.min(maxFrame, value));
    }}

    function setFrame(value) {{
      frameIndex = clampFrame(value);
      render();
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        window.clearInterval(timer);
        timer = null;
      }}
      els.playButton.textContent = "播放";
      els.playButton.classList.remove("active");
    }}

    function togglePlayback() {{
      if (timer !== null) {{
        stopPlayback();
        return;
      }}
      els.playButton.textContent = "暂停";
      els.playButton.classList.add("active");
      timer = window.setInterval(() => {{
        frameIndex = (frameIndex + 1) % gallery.framesPerClip;
        render();
      }}, 200);
    }}

    function renderFilmstrip() {{
      els.filmstrip.innerHTML = "";
      gallery.clips[clipIndex].forEach((record, index) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "thumb" + (index === frameIndex ? " active" : "");
        button.title = `frame ${{record.frame_index}}`;
        button.addEventListener("click", () => {{
          stopPlayback();
          setFrame(index);
        }});
        const img = document.createElement("img");
        img.src = record.sources.gt;
        img.alt = `frame ${{record.frame_index}} thumbnail`;
        const label = document.createElement("span");
        label.textContent = String(index).padStart(2, "0");
        button.appendChild(img);
        button.appendChild(label);
        els.filmstrip.appendChild(button);
      }});
    }}

    function render() {{
      const record = currentRecord();
      els.frameSlider.value = frameIndex;
      els.frameLabel.textContent = frameIndex;
      els.satelliteImage.src = record.sources.satellite;
      els.lidarImage.src = record.sources.lidar_overlay;
      els.gtImage.src = record.sources.gt;
      els.outputImage.src = record.outputs[cfg];
      els.cfgLabel.textContent = `CFG ${{cfg}}`;
      els.statusLine.textContent = `clip ${{clipIndex + 1}}/${{gallery.numClips}} · frame ${{frameIndex}}/${{gallery.framesPerClip - 1}} · ${{record.drive}} · ${{record.frame_index}}`;
      els.splitMeta.textContent = record.split;
      els.driveMeta.textContent = record.drive;
      els.frameMeta.textContent = record.frame_index;
      els.sampleMeta.textContent = record.sample_id;
      els.seedMeta.textContent = record.seed;
      els.checkpointMeta.textContent = gallery.checkpointLabel;
      els.noiseMeta.textContent = record.initial_noise_sha256.slice(0, 16) + "...";
      [...document.querySelectorAll(".cfgButton")].forEach((button) => {{
        button.classList.toggle("active", button.dataset.cfg === cfg);
      }});
      renderFilmstrip();
    }}

    gallery.clips.forEach((clip, index) => {{
      const first = clip[0];
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `片段 ${{index + 1}} · ${{first.drive}} · seed ${{first.seed}}`;
      els.clipSelect.appendChild(option);
    }});
    els.clipSelect.addEventListener("change", (event) => {{
      stopPlayback();
      clipIndex = Number(event.target.value);
      frameIndex = 0;
      render();
    }});
    els.frameSlider.addEventListener("input", (event) => {{
      stopPlayback();
      setFrame(Number(event.target.value));
    }});
    document.getElementById("prevButton").addEventListener("click", () => {{
      stopPlayback();
      setFrame(frameIndex - 1);
    }});
    document.getElementById("nextButton").addEventListener("click", () => {{
      stopPlayback();
      setFrame(frameIndex + 1);
    }});
    els.playButton.addEventListener("click", togglePlayback);
    [...document.querySelectorAll(".cfgButton")].forEach((button) => {{
      button.addEventListener("click", () => {{
        cfg = button.dataset.cfg;
        render();
      }});
    }});
    render();
  </script>
</body>
</html>
"""


def build_gallery(output_dir: Path) -> Path:
    """Write comparison.html for an inference output directory and return its path."""

    output_dir = Path(output_dir)
    metadata = _require_mapping(_read_json(output_dir / "metadata.json"), "metadata.json")
    records_payload = _read_json(output_dir / "records.json")
    if not isinstance(records_payload, list):
        raise ValueError("records.json must contain a list")

    frames_per_clip = int(metadata.get("frames_per_clip", 16))
    num_clips = int(metadata.get("num_clips", 2))
    if frames_per_clip != 16:
        raise ValueError(f"metadata.frames_per_clip must be 16, got {frames_per_clip}")
    if num_clips != 2:
        raise ValueError(f"metadata.num_clips must be 2, got {num_clips}")
    selection_report = _require_mapping(metadata.get("selection_report"), "metadata.selection_report")
    if not isinstance(selection_report.get("clips"), list):
        raise ValueError("metadata.selection_report.clips must be a list")

    cfg_scales = _cfg_scales(metadata)
    records = [_normalise_record(_require_mapping(record, f"records[{index}]"), cfg_scales) for index, record in enumerate(records_payload)]
    clips = _group_records(records, frames_per_clip=frames_per_clip, num_clips=num_clips)

    page_path = output_dir / "comparison.html"
    page_path.write_text(_gallery_html(metadata, clips, cfg_scales), encoding="utf-8")
    return page_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing records.json and metadata.json")
    args = parser.parse_args()
    print(build_gallery(args.input_dir))


if __name__ == "__main__":
    main()
