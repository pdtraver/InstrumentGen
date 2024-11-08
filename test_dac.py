import dac
from audiotools import AudioSignal

#import d_models
import torch

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

# model.to('cuda')

# Load audio signal file
signal = AudioSignal('/Users/bcarone/PycharmProjects/InstrumentGen/Continue?.wav')

signal = signal.resample(44100).to_mono().truncate_samples(44100)

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)
# print(signal.audio_data)
# print(signal.sample_rate)
# print(signal.audio_data.shape)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

z_fromcodes = model.quantizer.from_codes(codes)[0]
z_fromlatents = model.quantizer.from_latents(latents)[0]

# rec = model.decode(z)
# rec_fromcodes = model.decode(z_fromcodes)
# rec_fromlatents = model.decode(z_fromlatents)

# AudioSignal(rec.detach(),44100).write("z.wav")
# AudioSignal(rec_fromcodes.detach(),44100).write("z_fromcodes.wav")
# AudioSignal(rec_fromlatents.detach(),44100).write("z_fromlatents.wav")

# print(rec.shape)
# print(z_fromcodes == z)
# print(z_fromlatents == z)
# print(latents.view(1,9,-1,87).shape)

print(z.shape)
print(codes.shape)
print(latents.shape)
