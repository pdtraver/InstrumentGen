import numpy as np
import librosa
import torch
import laion_clap
import os

# Change your audio files & checkpoints here
audio_filename = 'Continue?.wav'
ckpt_filename = '630k-best.pt'

# Get path locations of audio & ckpt files
audio_loc = os.path.join(os.getcwd(), audio_filename)
ckpt_loc = os.path.join(os.getcwd(), ckpt_filename)

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1, a_max=1.)
    return (x * 32767.).astype(np.int16)

# Change enable fusion to False if fusion of features not desired (need to load different .pt file too)
# Was getting errors with fusion -- seemed to get Nones in mel spectrogram -- default to non-fusion now
# Limitation of non-fusion is needing to use audio of the same length (and less than 10 seconds)
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt(ckpt = ckpt_loc)

test_audio, _ = librosa.load(audio_loc)
test_text = ['Sick dubstep bass sound']

# Generate audio embeddings from audio data, returning tensor
audio_data = test_audio.reshape(1, -1)
audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() #quantize data
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Generate text embeddings from text, returning tensor
text_embed = model.get_text_embedding(test_text, use_tensor=True)
print(text_embed)
print(text_embed.shape)