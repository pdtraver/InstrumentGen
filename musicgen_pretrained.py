from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


print('Loading model')
model = MusicGen.get_pretrained("small")
print('Setting model parameters')
model.set_generation_params(duration=4)  # generate 8 seconds.

descriptions = ["happy rock", "energetic EDM"]

print('Generating audio')
wav = model.generate(descriptions)  # generates 2 samples.

print('Preparing Tokens and Attributes')
attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
print('Generating tokens')
tokens = model._generate_tokens(attributes, prompt_tokens)

print(tokens.shape)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
