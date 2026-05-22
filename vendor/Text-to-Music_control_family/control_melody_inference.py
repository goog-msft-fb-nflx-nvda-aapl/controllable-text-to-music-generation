from diffusers import DiffusionPipeline
import torch
from utils.extract_conditions import compute_melody_v2, extract_melody_one_hot
import torch.nn.functional as F
from config_inference import get_config
import os 
import soundfile as sf
from utils.stable_audio_dataset_utils import load_audio_file

# Initialize pipeline and download weights from huggingface
base_model = "stabilityai/stable-audio-open-1.0"
method = "SongEcho_base" # Options: SongEcho_base, SongEcho_large, MuseControlLite
prompt = "A vibrant MIDI electronic composition with a hopeful and optimistic vibe."

pipe = DiffusionPipeline.from_pretrained(
    base_model,
    custom_pipeline="fundwotsai2001/Text-to-Music_control_family",
    attn_processor_type = method,
    # force_download=True,
    torch_dtype=torch.float16,
    adapter_repo_id="fundwotsai2001/Text-to-Music_control_family",
    adapter_checkpoint=f"{method}/adapters.safetensors",
    melody_encoder_checkpoint=f"{method}/melody_encoder.safetensors",
)
pipe = pipe.to("cuda")
audio_file = "./melody_condition_audio/610_bass.mp3"

extracted_condition = pipe.encode_melody_from_path(audio_file)

# Inference stable-audio open + adapters
waveform = pipe(
    extracted_condition = extracted_condition, 
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale_text=7.0,
    guidance_scale_con=1.0, # modify to adjust the control strength
    num_waveforms_per_prompt=1,
    audio_end_in_s=2097152 / 44100,
    generator=torch.Generator().manual_seed(42),
).audios 

# Save audio 
output = waveform[0].T.float().cpu().numpy()
sf.write("test.wav", output, pipe.vae.sampling_rate)

# Calculate melody accuracy
melody_condition = extract_melody_one_hot(audio_file)      
gen_melody = extract_melody_one_hot("test.wav")
min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
accuracy = matches / min_len_melody
print("melody accuracy", accuracy)