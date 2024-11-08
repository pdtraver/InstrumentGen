# InstrumentGen
Instrument Gen Python Implementation

DAC: https://github.com/descriptinc/descript-audio-codec

CLAP: https://github.com/LAION-AI/CLAP

MusicGen: https://github.com/facebookresearch/audiocraft/tree/main

Setting up environment from scratch:
1. Create virtual environment with Python 3.9.0 (prerequisite: install conda)
   ```
   conda create -n instrument_gen python==3.9.0
   conda activate instrument_gen
   python --version
   ```
2. Install DAC & test with script
   ```
   pip install git+https://github.com/descriptinc/descript-audio-codec
   python test_dac.py
   ```
3. Install CLAP + extraneous dependencies & test with script
   ```
   pip install laion-clap
   pip install torchvision==0.16.0
   pip install torchaudio==2.1.0
   python clap_test.py
   ```

4. Clone Audiocraft & run musicgen test script
   ```
   git clone https://github.com/facebookresearch/audiocraft.git
   pip install -r ./audiocraft/requirements.txt # this should fail
   python -m pip install -e .
   ```
   If you are not logged into git you may have to configure this before running the above.
   If you run into errors based on dependencies after this, refer to the updated venv.txt in the repo & check what you are missing.
   I had to install the following, for example:
   ```
   pip install spacy==3.7.5
   pip install flashy==0.0.2
   ```
   Finally, run the pretrained MusicGen model. Feel free to change the descriptions for different audio outputs.
   ```
   python musicgen_pretrained.py
   ```
   
