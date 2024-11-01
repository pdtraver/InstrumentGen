from audiocraft.modules.transformer import StreamingTransformer
from audiocraft.modules.codebook_patterns import DelayedPatternProvider
from torch import 

## DAC
# These will be outputted from DAC
codes = None # shape B x N x T

num_codes = 9
delays = None # 1 step delay
flatten_first = 0 # Timestamps to flatten
empty_initial = 0 # Empty coordinates to prepend
timestamps = codes.shape[2] # Number of frames in audio/codes

special_token = -1

## Delayed pattern
pattern_provider = DelayedPatternProvider(n_q = num_codes,
                                         delays = delays,
                                         flatten_first = flatten_first,
                                         empty_initial = empty_initial)

delayed_pattern = pattern_provider.get_pattern(timestamps = timestamps)
delayed_pattern_sequence, _, _ = delayed_pattern.build_pattern_sequence(z = codes,
                                                                  special_token = special_token)

## Transformer -- instrument gen version
data_dim = 512
num_heads = 16
num_layers = 12

transformer = StreamingTransformer(d_model = data_dim,
                                   num_heads = num_heads,
                                   num_layers = num_layers)

delayed_predictions = transformer.forward(delayed_pattern_sequence)

## De-interleave
predictions = delayed_pattern.revert_pattern_sequence(s = delayed_predictions,
                                                      special_token = special_token)

## Cross entropy
loss = nn.CrossEntropyLoss



