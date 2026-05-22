#!/usr/bin/env python3
"""Remote experiment harness for controllable text-to-music generation."""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "home" / "fundwotsai" / "Deep_MIR_hw2"
TARGETS, REFS = DATA / "target_music_list_60s", DATA / "referecne_music_list_60s"
SOURCE_ROOT = ROOT / "vendor" if (ROOT / "vendor").exists() else ROOT / "third_party"
FAMILY = SOURCE_ROOT / "Text-to-Music_control_family"
RESULTS, AUDIO = ROOT / "results", ROOT / "generated_audio"
MANIFEST, CAPTIONS, RETRIEVAL = RESULTS / "manifest.json", RESULTS / "captions.json", RESULTS / "retrieval.json"
GENERATIONS, METRICS, REPORT = RESULTS / "generation.json", RESULTS / "metrics.json", ROOT / "report.html"
BASE, CONTROL = "stabilityai/stable-audio-open-1.0", "fundwotsai2001/Text-to-Music_control_family"
SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rpath(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def device(value: str) -> str:
    import torch
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def targets(path: Path) -> list[dict[str, str]]:
    return load(path)["targets"]


def captions(path: Path) -> dict[str, str]:
    return {row["id"]: row["caption"] for row in load(path)["captions"]}


def inventory(_: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    files = lambda folder: sorted(path for path in folder.iterdir() if path.suffix.lower() in SUFFIXES)
    target_files, ref_files = files(TARGETS), files(REFS)
    save(MANIFEST, {
        "dataset_dir": rel(DATA),
        "target_count": len(target_files),
        "reference_count": len(ref_files),
        "targets": [{"id": f"t{i:02d}", "path": rel(path)} for i, path in enumerate(target_files)],
        "references": [{"id": f"r{i:03d}", "path": rel(path)} for i, path in enumerate(ref_files)],
    })
    print(f"Wrote {MANIFEST}: {len(target_files)} targets, {len(ref_files)} references")


def caption(args: argparse.Namespace) -> None:
    import librosa
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    dev = device(args.device)
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=dev if dev.startswith("cuda") else None
    )
    model = model if dev.startswith("cuda") else model.to(dev)
    model.eval()
    old = load(CAPTIONS) if CAPTIONS.exists() else {"captions": []}
    rows = {row["id"]: row for row in old["captions"]}
    chosen = targets(args.manifest)[:args.limit] if args.limit else targets(args.manifest)
    for target in chosen:
        if target["id"] in rows and not args.overwrite:
            continue
        wav, _ = librosa.load(rpath(target["path"]), sr=processor.feature_extractor.sampling_rate, mono=True)
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "audio_url": target["path"]}, {"type": "text", "text": args.prompt}],
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=prompt, audios=[wav], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )
        inputs = {key: value.to(dev) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        rows[target["id"]] = {
            "id": target["id"], "target_path": target["path"], "model": args.model,
            "prompt": args.prompt, "caption": text,
        }
        save(CAPTIONS, {"caption_model": args.model, "captions": [rows[key] for key in sorted(rows)]})
        print(f"Captioned {target['id']}: {' '.join(text.split())[:120]}")


class Clap:
    def __init__(self, model_name: str, dev: str):
        import torch
        from transformers import AutoProcessor, ClapModel
        self.torch, self.dev = torch, device(dev)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.dev).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def audio(self, paths: Iterable[Path]) -> np.ndarray:
        import librosa
        out = []
        for path in paths:
            wav, _ = librosa.load(path, sr=self.sr, mono=True)
            inputs = self.processor(audios=wav, sampling_rate=self.sr, return_tensors="pt")
            inputs = {key: value.to(self.dev) for key, value in inputs.items()}
            with self.torch.inference_mode():
                vector = self.model.get_audio_features(**inputs)
            out.append(self.torch.nn.functional.normalize(vector, dim=-1)[0].float().cpu().numpy())
        return np.stack(out)

    def text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.dev) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs)
        return self.torch.nn.functional.normalize(vector, dim=-1).float().cpu().numpy()


def retrieve(args: argparse.Namespace) -> None:
    manifest, scorer = load(args.manifest), Clap(args.clap_model, args.device)
    target_vec = scorer.audio([rpath(row["path"]) for row in manifest["targets"]])
    ref_vec = scorer.audio([rpath(row["path"]) for row in manifest["references"]])
    similarity, rows = target_vec @ ref_vec.T, []
    for index, target in enumerate(manifest["targets"]):
        order = np.argsort(-similarity[index])[:args.top_k]
        matches = [{
            "rank": rank + 1, "reference_id": manifest["references"][ref]["id"],
            "reference_path": manifest["references"][ref]["path"],
            "clap_audio_cosine": float(similarity[index, ref]),
        } for rank, ref in enumerate(order)]
        rows.append({"id": target["id"], "target_path": target["path"], "matches": matches})
        print(f"{target['id']} -> {matches[0]['reference_id']} {matches[0]['clap_audio_cosine']:.4f}")
    save(RETRIEVAL, {"clap_model": args.clap_model, "top_k": args.top_k, "retrieval": rows})


def pipe_for(args: argparse.Namespace, dev: str):
    import torch
    from diffusers import DiffusionPipeline
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.mode == "text":
        return DiffusionPipeline.from_pretrained(BASE, torch_dtype=dtype).to(dev)
    return DiffusionPipeline.from_pretrained(
        BASE, custom_pipeline=CONTROL, attn_processor_type=args.method, torch_dtype=dtype,
        adapter_repo_id=CONTROL, adapter_checkpoint=f"{args.method}/adapters.safetensors",
        melody_encoder_checkpoint=f"{args.method}/melody_encoder.safetensors",
    ).to(dev)


def generate(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch
    cap, dev = captions(args.captions), device(args.device)
    pipe = pipe_for(args, dev)
    current = load(GENERATIONS)["generations"] if GENERATIONS.exists() else []
    rows = {(row["id"], row["variant"]): row for row in current}
    folder = AUDIO / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    chosen = targets(args.manifest)[:args.limit] if args.limit else targets(args.manifest)
    for target in chosen:
        key, out, prompt = (target["id"], args.variant), folder / f"{target['id']}.wav", cap[target["id"]]
        if key in rows and out.exists() and not args.overwrite:
            continue
        generator = torch.Generator(device=dev).manual_seed(args.seed)
        with torch.inference_mode():
            if args.mode == "melody":
                # Official MuseControlLite CQT MIR melody condition, not target audio conditioning.
                condition = pipe.encode_melody_from_path(str(rpath(target["path"])))
                wav = pipe(
                    extracted_condition=condition, prompt=prompt, negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps, guidance_scale_text=args.text_guidance,
                    guidance_scale_con=args.condition_guidance, num_waveforms_per_prompt=1,
                    audio_end_in_s=args.seconds, generator=generator,
                ).audios
                source = "target CQT melody extracted by MuseControlLite"
            else:
                wav = pipe(
                    prompt=prompt, negative_prompt=args.negative_prompt, num_inference_steps=args.steps,
                    guidance_scale=args.text_guidance, num_waveforms_per_prompt=1,
                    audio_end_in_s=args.seconds, generator=generator,
                ).audios
                source = "text only"
        sf.write(out, wav[0].T.float().cpu().numpy(), pipe.vae.sampling_rate)
        rows[key] = {
            "id": target["id"], "target_path": target["path"], "generated_path": rel(out),
            "caption": prompt, "variant": args.variant, "mode": args.mode, "base_model": BASE,
            "control_method": args.method if args.mode == "melody" else None, "steps": args.steps,
            "seed": args.seed, "seconds": args.seconds, "text_guidance": args.text_guidance,
            "condition_guidance": args.condition_guidance if args.mode == "melody" else None,
            "condition_source": source,
        }
        save(GENERATIONS, {"generations": [rows[key] for key in sorted(rows)]})
        print(f"Generated {out}")


def assemble_generation(args: argparse.Namespace) -> None:
    cap, rows = captions(args.captions), []
    variants = {
        "stable_text": ("text", None, None, "text only"),
        "muse_melody_cfg1": ("melody", "MuseControlLite", 1.0, "target CQT melody extracted by MuseControlLite"),
        "muse_melody_cfg15": ("melody", "MuseControlLite", 1.5, "target CQT melody extracted by MuseControlLite"),
    }
    for target in targets(args.manifest):
        for variant, (mode, method, condition_guidance, source) in variants.items():
            output = AUDIO / variant / f"{target['id']}.wav"
            if not output.exists():
                raise SystemExit(f"Missing completed generation {output}")
            rows.append({
                "id": target["id"], "target_path": target["path"], "generated_path": rel(output),
                "caption": cap[target["id"]], "variant": variant, "mode": mode, "base_model": BASE,
                "control_method": method, "steps": args.steps, "seed": args.seed, "seconds": args.seconds,
                "text_guidance": args.text_guidance, "condition_guidance": condition_guidance,
                "condition_source": source,
            })
    save(GENERATIONS, {"generations": rows})
    print(f"Wrote {GENERATIONS}: {len(rows)} full generation rows")


def melody(target: Path, sample: Path) -> float:
    sys.path.insert(0, str(FAMILY))
    from utils.extract_conditions import extract_melody_one_hot
    expected, got = extract_melody_one_hot(str(target)), extract_melody_one_hot(str(sample))
    length = min(expected.shape[1], got.shape[1])
    return float(((got[:, :length] == expected[:, :length]) & (got[:, :length] == 1)).sum() / length)


def aes(ckpt: str | None):
    from audiobox_aesthetics.infer import initialize_predictor
    return initialize_predictor(ckpt_path=ckpt) if ckpt else initialize_predictor()


def aes_rows(predictor: Any, paths: list[Path]) -> list[dict[str, float]]:
    return [{key: float(value) for key, value in row.items()}
            for row in predictor.forward([{"path": str(path)} for path in paths])]


def evaluate(args: argparse.Namespace) -> None:
    scorer, predictor, ret_rows, gen_rows = Clap(args.clap_model, args.device), aes(args.aes_ckpt), [], []
    if args.retrieval.exists():
        items = load(args.retrieval)["retrieval"]
        best = [row["matches"][0] for row in items]
        for item, match, axes in zip(items, best, aes_rows(predictor, [rpath(row["reference_path"]) for row in best])):
            ret_rows.append({
                "id": item["id"], "target_path": item["target_path"], "reference_id": match["reference_id"],
                "reference_path": match["reference_path"], "clap_audio_cosine": match["clap_audio_cosine"],
                "melody_accuracy": melody(rpath(item["target_path"]), rpath(match["reference_path"])),
                "audiobox_aesthetics": axes,
            })
            print(f"Retrieval metrics {item['id']}")
    if args.generations.exists():
        items = load(args.generations)["generations"]
        for item, axes in zip(items, aes_rows(predictor, [rpath(row["generated_path"]) for row in items])):
            target, generated = scorer.audio([rpath(item["target_path"]), rpath(item["generated_path"])])
            text = scorer.text([item["caption"]])[0]
            gen_rows.append({
                "id": item["id"], "target_path": item["target_path"], "generated_path": item["generated_path"],
                "caption": item["caption"], "variant": item["variant"], "mode": item["mode"],
                "condition_source": item["condition_source"], "clap_target_text": float(target @ text),
                "clap_text_generated": float(text @ generated), "clap_generated_target": float(generated @ target),
                "melody_accuracy": melody(rpath(item["target_path"]), rpath(item["generated_path"])),
                "audiobox_aesthetics": axes,
            })
            print(f"Generation metrics {item['id']} {item['variant']}")
    save(METRICS, {"clap_model": args.clap_model, "retrieval": ret_rows, "generation": gen_rows})
    print(f"Wrote {METRICS}")


def td(value: Any) -> str:
    if isinstance(value, float):
        value = "" if math.isnan(value) else f"{value:.4f}"
    return f"<td>{html.escape(str(value))}</td>"


def acells(row: dict[str, Any]) -> str:
    return "".join(td(row["audiobox_aesthetics"].get(key, "")) for key in ["CE", "CU", "PC", "PQ"])


def report(args: argparse.Namespace) -> None:
    manifest = load(args.manifest)
    cap = captions(args.captions) if args.captions.exists() else {}
    metrics = load(args.metrics) if args.metrics.exists() else {"retrieval": [], "generation": []}
    ret = {row["id"]: row for row in metrics["retrieval"]}
    gen = {(row["id"], row["variant"]): row for row in metrics["generation"]}
    variants = sorted({row["variant"] for row in metrics["generation"]})
    sections = []
    for target in manifest["targets"]:
        rid, got, grows = target["id"], ret.get(target["id"]), []
        rhtml = "<p>Retrieval metrics not generated yet.</p>"
        if got:
            rhtml = (
                f"<p>Top reference: <code>{html.escape(Path(got['reference_path']).stem)}</code></p>"
                "<table><tr><th>CLAP target-reference</th><th>Melody</th><th>CE</th><th>CU</th><th>PC</th><th>PQ</th></tr>"
                f"<tr>{td(got['clap_audio_cosine'])}{td(got['melody_accuracy'])}{acells(got)}</tr></table>"
            )
        for variant in variants:
            row = gen.get((rid, variant))
            if row:
                grows.append(
                    "<tr>" + td(variant) + td(row["condition_source"])
                    + f'<td><audio controls preload="none" src="{html.escape(row["generated_path"])}"></audio></td>'
                    + td(row["clap_target_text"]) + td(row["clap_text_generated"]) + td(row["clap_generated_target"])
                    + td(row["melody_accuracy"]) + acells(row) + "</tr>"
                )
        ghtml = "<p>Generation metrics not generated yet.</p>"
        if grows:
            ghtml = (
                "<table><tr><th>Variant</th><th>Condition</th><th>Audio</th><th>CLAP target-text</th>"
                "<th>CLAP text-generated</th><th>CLAP generated-target</th><th>Melody</th>"
                "<th>CE</th><th>CU</th><th>PC</th><th>PQ</th></tr>" + "".join(grows) + "</table>"
            )
        sections.append(
            f"<section><h2>{html.escape(rid)}: {html.escape(Path(target['path']).stem)}</h2>"
            f"<p><strong>Qwen2-Audio caption:</strong> {html.escape(cap.get(rid, 'Caption not generated yet.'))}</p>"
            f"<h3>Retrieval</h3>{rhtml}<h3>Generation</h3>{ghtml}</section>"
        )
    source = html.escape(args.source_code_url)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Text-to-Music Report</title><style>
body{{background:#f7f7f5;color:#1f252a;font-family:system-ui,sans-serif;line-height:1.45;margin:0}}main{{margin:auto;max-width:1320px;padding:36px 28px 72px}}section{{border-top:1px solid #ccd3d7;margin-top:32px;padding-top:24px}}h1,h2,h3{{letter-spacing:0;line-height:1.18}}h2{{overflow-wrap:anywhere}}code{{background:#e8eeef;padding:2px 5px}}table{{background:white;border-collapse:collapse;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid #d7dde0;padding:9px 10px;text-align:left;vertical-align:top}}th{{background:#dfe9e7}}audio{{height:34px;width:248px}}.meta{{max-width:950px}}
</style></head><body><main><h1>Controllable Text-to-Music Generation</h1><p class="meta">Qwen2-Audio captions target music. CLAP retrieves provided references. Stable Audio Open text-only generation is compared with MuseControlLite using official target CQT MIR melody features; target audio is never an audio condition.</p><p>Source code: <a href="{source}">{source}</a></p><p>Metrics are CLAP cosines, Meta Audiobox Aesthetics CE/CU/PC/PQ, and assignment chromagram melody accuracy.</p>{''.join(sections)}</main></body></html>'''
    REPORT.write_text(page, encoding="utf-8")
    print(f"Wrote {REPORT}")


def export_site(args: argparse.Namespace) -> None:
    report_html = REPORT.read_text(encoding="utf-8")
    site_dir, audio_dir = ROOT / "site", ROOT / "site" / "audio"
    site_dir.mkdir(exist_ok=True)
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    variants = ["stable_text", "muse_melody_cfg1", "muse_melody_cfg15"]
    for variant in variants:
        source_dir, output_dir = AUDIO / variant, audio_dir / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        for wav in sorted(source_dir.glob("*.wav")):
            mp3 = output_dir / f"{wav.stem}.mp3"
            subprocess.run([
                args.ffmpeg, "-y", "-i", str(wav), "-vn", "-codec:a", "libmp3lame",
                "-b:a", args.bitrate, str(mp3),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            report_html = report_html.replace(
                f"generated_audio/{variant}/{wav.name}", f"audio/{variant}/{mp3.name}"
            )
    root_html = report_html.replace('src="audio/', 'src="site/audio/')
    (site_dir / "index.html").write_text(report_html, encoding="utf-8")
    (ROOT / "index.html").write_text(root_html, encoding="utf-8")
    REPORT.write_text(root_html, encoding="utf-8")
    print(f"Wrote {site_dir / 'index.html'}, {ROOT / 'index.html'}, and {sum(1 for _ in audio_dir.rglob('*.mp3'))} MP3 files")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory").set_defaults(func=inventory)
    p = sub.add_parser("caption")
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument("--device", default="auto"); p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--prompt", default="Generate a detailed English music caption. Describe instruments, genre, tempo or rhythm, melody, mood, and recording texture. Do not identify the song title.")
    p.set_defaults(func=caption)
    p = sub.add_parser("retrieve")
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.add_argument("--clap-model", default="laion/clap-htsat-unfused")
    p.add_argument("--device", default="auto"); p.add_argument("--top-k", type=int, default=3); p.set_defaults(func=retrieve)
    p = sub.add_parser("generate")
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.add_argument("--captions", type=Path, default=CAPTIONS)
    p.add_argument("--mode", choices=["text", "melody"], required=True); p.add_argument("--variant", required=True)
    p.add_argument("--method", choices=["MuseControlLite", "SongEcho_base", "SongEcho_large"], default="MuseControlLite")
    p.add_argument("--device", default="cuda"); p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--steps", type=int, default=50); p.add_argument("--seconds", type=float, default=2097152 / 44100)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--text-guidance", type=float, default=7.0)
    p.add_argument("--condition-guidance", type=float, default=1.0); p.add_argument("--negative-prompt", default="")
    p.add_argument("--limit", type=int); p.add_argument("--overwrite", action="store_true"); p.set_defaults(func=generate)
    p = sub.add_parser("assemble-generation")
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.add_argument("--captions", type=Path, default=CAPTIONS)
    p.add_argument("--steps", type=int, default=50); p.add_argument("--seconds", type=float, default=2097152 / 44100)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--text-guidance", type=float, default=7.0)
    p.set_defaults(func=assemble_generation)
    p = sub.add_parser("evaluate")
    p.add_argument("--retrieval", type=Path, default=RETRIEVAL); p.add_argument("--generations", type=Path, default=GENERATIONS)
    p.add_argument("--clap-model", default="laion/clap-htsat-unfused"); p.add_argument("--aes-ckpt")
    p.add_argument("--device", default="auto"); p.set_defaults(func=evaluate)
    p = sub.add_parser("report")
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.add_argument("--captions", type=Path, default=CAPTIONS)
    p.add_argument("--metrics", type=Path, default=METRICS); p.add_argument("--source-code-url", default="TODO: public repository URL")
    p.set_defaults(func=report)
    p = sub.add_parser("export-site")
    p.add_argument("--ffmpeg", default="/home/jtan/miniconda3/envs/cttm/bin/ffmpeg")
    p.add_argument("--bitrate", default="160k")
    p.set_defaults(func=export_site)
    return top


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
