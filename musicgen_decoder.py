from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.models.lm import LMModel
from audiocraft.models.encodec import DAC
from audiocraft.models.musicgen import MusicGen

# Instrument Gen Params
print('Loading parameters...')
NUM_CODEBOOKS = 9
DELAY_SAMPLES = None
FLATTEN_FIRST = 0
EMPTY_INITIAL = 0
NUM_CODES = 1024 # cardinality/vocab size
DIM = 512 # dim of transformer layer
NUM_HEADS = 16
NUM_LAYERS = 12
DAC_VERSION = "44khz"

# Chosen Params
MAX_DURATION = 20 # s -- has to be more than extend_stride in musicgen.py

# Prepare LM
# Providers
print('Building providers...')
condition_provider = None
fuser = None
pattern_provider = DelayedPatternProvider(n_q = NUM_CODEBOOKS,
                                          delays = DELAY_SAMPLES,
                                          flatten_first = FLATTEN_FIRST,
                                          empty_initial = EMPTY_INITIAL)

# Full LM
print('Defining LM...')
lm_model = LMModel(pattern_provider = pattern_provider,
                   condition_provider = condition_provider,
                   fuser = fuser,
                   n_q = NUM_CODEBOOKS,
                   card = NUM_CODES,
                   dim = DIM,
                   num_heads = NUM_HEADS,
                   num_layers = NUM_LAYERS)

# Prepare DAC
print('Importing DAC...')
dac = DAC(model_type = DAC_VERSION)

# Prepare MusicGen
print('Building MusicGen...')
music_gen = MusicGen('Instrument_Gen_Transformer', dac, lm_model, MAX_DURATION)
print(type(music_gen))

