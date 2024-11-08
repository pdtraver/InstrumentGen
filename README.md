# InstrumentGen
## Instrument Gen Python Implementation

### Links to components of InstrumentGen
DAC: https://github.com/descriptinc/descript-audio-codec

CLAP: https://github.com/LAION-AI/CLAP

MusicGen: https://github.com/facebookresearch/audiocraft/tree/main

### Setting up environment
Brandon was nice enough to compile a requirements.txt. To setup your environment from scratch do the following:
```
conda create -n instrument_gen python==3.9.0
conda activate instrument_gen
python --version
pip install -r requirements.txt
```
If you still have issues after trying this, follow the steps below to install things manually.

Setting up environment manually:
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
   ```
   You'll also need to download the checkpoint used in the test script & change the location in the file to where it's stored on your machine.
   You can download the 630k-best.pt file here: https://huggingface.co/lukewys/laion_clap/blob/main/630k-best.pt. Other checkpoints that we might explore
   in the future can be found here: https://huggingface.co/lukewys/laion_clap/tree/main.
   Now you should be able to run your test script:
   ```
   python clap_test.py
   ```

4. Clone Audiocraft & run musicgen test script
   ```
   git clone https://github.com/facebookresearch/audiocraft.git
   pip install -r ./audiocraft/requirements.txt # this should fail
   python -m pip install -e .
   ```
   If you are not logged into git you may have to configure this before running the above.
   If you run into errors based on dependencies after this, refer to the updated peters_venv.txt or requirements.txt in the repo & check what you are missing.
   I had to install the following, for example:
   ```
   pip install spacy==3.7.5
   pip install flashy==0.0.2
   ```
   Finally, run the pretrained MusicGen model. Feel free to change the descriptions for different audio outputs.
   ```
   python musicgen_pretrained.py
   ```

### Getting ready to train
Luckily DAC and CLAP are frozen modules in InstrumentGen, so we do not need to train these. However, the MusicGen module we will need to train on the NSynth dataset. We have a few steps before we'll be ready to train, which include:
1. NSynth preparation (NSynth into DAC)
2. CLAP preparation (NSynth + text into CLAP)
3. RVQ on CLAP output (Quantizing CLAP embeddings)
4. MusicGen setup (Codebook interleaving patterns)
5. Timbral Consistency Loss (Implementing TC measure)

Each of these components will be important to matching the implementation provided in the paper. Some of the details (particularly regarding the text conditioning and TC measure) are a bit hazy as Native Instruments is a proprietary company, so we will likely have room for creativity and innovation on these fronts. Let's discuss at our meeting what people are most interested in and we can prepare for each subtask.
