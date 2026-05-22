# Controllable Text-to-Music Experiments

This remote workspace holds the Deep MIR assignment pipeline.

## Method

- Captioning: `Qwen/Qwen2-Audio-7B-Instruct` captions each target music file
  with the official Qwen2-Audio chat template.
- Retrieval: CLAP embeddings from `laion/clap-htsat-unfused` rank each item in
  the provided reference list.
- Generation: Stable Audio Open text-only captions are compared with
  MuseControlLite from the official `Text-to-Music_control_family` pipeline.
  MuseControlLite receives only its official CQT MIR melody feature extracted
  from target music, never a target audio condition.
- Metrics: CLAP cosine scores, Meta Audiobox Aesthetics CE/CU/PC/PQ, and the
  assignment chromagram melody accuracy.

## Environment

The server has no sudo audio tooling, so this workspace uses conda:

```bash
cd /home/jtan/controllable_text_to_music_generation
/home/jtan/miniconda3/bin/conda create -n cttm -y python=3.11 pip "ffmpeg>=4,<8" -c conda-forge
/home/jtan/miniconda3/envs/cttm/bin/python -m pip install \
  -r third_party/Text-to-Music_control_family/requirements.txt \
  -e third_party/audiobox-aesthetics
/home/jtan/miniconda3/envs/cttm/bin/python -m pip install --force-reinstall \
  torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
/home/jtan/miniconda3/envs/cttm/bin/python -m pip install --force-reinstall \
  --no-deps torchcodec==0.2
```

The final CUDA 12.4 pin matters on this host. The unpinned upstream requirement
currently resolves to CUDA 13 wheels, which the installed GPU driver cannot run.
TorchCodec `0.2` is the official compatibility row for PyTorch `2.6`.
The FFmpeg `>=4,<8` pin keeps the shared FFmpeg libraries within TorchCodec
0.2's supported set.

`stabilityai/stable-audio-open-1.0` is gated on Hugging Face. After accepting
its model access terms, log in before generation:

```bash
/home/jtan/miniconda3/envs/cttm/bin/huggingface-cli login
```

## Remote Commands

Use `tmux` for model downloads and GPU runs:

```bash
cd /home/jtan/controllable_text_to_music_generation
PY=/home/jtan/miniconda3/envs/cttm/bin/python
$PY experiments/music_pipeline.py inventory

tmux new -s cttm_caption
$PY experiments/music_pipeline.py caption
$PY experiments/music_pipeline.py retrieve

$PY experiments/music_pipeline.py generate --mode text --variant stable_text
$PY experiments/music_pipeline.py generate --mode melody --variant muse_melody_cfg1 --method MuseControlLite --condition-guidance 1.0
$PY experiments/music_pipeline.py generate --mode melody --variant muse_melody_cfg15 --method MuseControlLite --condition-guidance 1.5

$PY experiments/music_pipeline.py assemble-generation
$PY experiments/music_pipeline.py evaluate
$PY experiments/music_pipeline.py report --source-code-url PUBLIC_REPOSITORY_URL
$PY experiments/music_pipeline.py export-site
```

For a short smoke test, append `--limit 1 --steps 8` to generation commands.
Artifacts land in `results`, `generated_audio`, and `report.html`.
`export-site` transcodes the final generated WAV files to browser-playable MP3
files under `site/audio` and writes `site/index.html` with relative audio
links for a standalone site bundle. It also writes root `index.html` and
updates `report.html` to use the same MP3 files through `site/audio`, so a
branch-root GitHub Pages site and a checkout both expose playable audio.
