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
   Let's first clone the repo:
   ```
   git clone https://github.com/facebookresearch/audiocraft.git
   ```
   Now, before we install the requirements, please go into the requirements.txt file within the audiocraft repo. Please change the following dependency versions to what is listed         below:
   ```
   flashy==0.0.2            # previously >=0.0.1
   spacy==3.7.5             # previously ==3.7.6
   xformers==0.0.22.post7   # previously <0.0.23
   ```
   This should help with the following two commands, which should install all the dependencies we need correctly. You'll need to run these from within the audiocraft repo (i.e. change    your terminal location):
   ```
   pip install -r ./audiocraft/requirements.txt # this should fail
   python -m pip install -e .
   ```
   If you are not logged into git you may have to configure this before running the above.
   If you run into errors based on dependencies after this, refer to the updated peters_venv.txt or requirements.txt in the repo & check what you are missing.
   For example, I had to manually install the following (these are now changed in the requirements.txt file) the first time I set the repo up:
   ```
   pip install spacy==3.7.5
   pip install flashy==0.0.2
   ```
   Finally, run the pretrained MusicGen model. Feel free to change the descriptions for different audio outputs.
   ```
   python musicgen_pretrained.py
   ```

#### If all else fails
We've had some folks struggling to do the above on their own machines due to annoying Mac things (some dependencies not being available easily on M1/M2 chips) or the steps above not starting from a fresh machine outlook. So, here are an alternative set of steps that start from the very beginning and hopefully should allow for installation of M1/M2 chips.
1. Setup homebrew, virtualenv & pyenv
   
   This is homebrew; it should only be needed on MacOS. Run the command in your terminal (in any location), hit enter twice and let it install.
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Let's now install virtualenv. You might already have it, but just in case. We won't need homebrew just yet:
   ```
   pip install virtualenv
   ```
   Next is pyenv. This is a python version manager that allows for easy switching between versions on Unix machines:
   ```
   brew update
   brew install pyenv
   ```
   We need to set a few things up in your terminal before you can use pyenv in the correct way. This depends on which type of shel you're in: bash, zsh or fish. I'll paste the             directions below for a zsh and bash shells below. For MacOS users, it's probably helpful to go through both sets of directions. It's a little tricky so follow closely:

   For bash shells, run the following:
   ```
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   ```
   Then navigate to ```~/.bash_profile``` file. If you type just ```cd``` in the terminal it'll take you to the home directory, otherwise known as ```~```. From there, if you have        vscode installed you can type ```code .bash_profile``` and it should pop up in a vscode window. You can use another text editor or run vim directly in the terminal, but you'll have    to figure that out yourselves. Once you have the file open, add the following lines to the end of it:
   ```
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
   echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
   ```
   If you have a ```~/.profile``` file as well, you can add the following lines there:
   ```
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
   echo 'eval "$(pyenv init -)"' >> ~/.profile
   ```
   I did not so you might not as well.

   For zsh shells, run the following in the terminal:
   ```
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   ```

   After doing the above, you'll need to run an important command to activate pyenv in your shell environment:
   ```
   exec "$SHELL"
   ```
   This should activate everything we just set up. You will have to do this each time you open a new terminal to make pyenv work, so remember it by heart (or set an alias).
   
   Now we should be able to install a variety of versions of python & manage them easily. Let's try it with python 3.9.1 (for some reason 3.9.0 doesn't work for me, but try this too      in case it works for you):
   ```
   pyenv install 3.9.1
   ```
   If all goes well, you should see ```Installed Python-3.9.1 to /Users/your_username/.pyenv/versions/3.9.1```.
   If you have multiple python versions installed, you now can switch easily between them by typing:
   ```
   pyenv global [python-verion]
   ```
   For us, you can write:
   ```
   pyenv global 3.9.1
   ```
   Lastly, we'll install the plug between pyenv and virtualenv so we can control our virtual environments within pyenv:
   ```
   git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
   ```
   This comes from this repo, for those interested: https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv
3. Create a venv
   
   It seemed like conda was giving people trouble last time, so here's a non-conda way to make a virtual environment. You can do this by using virtualenv directly in the location         where the InstrumentGen repo is located. I'm going to suggest using pyenv, though, so you can reuse environments across projects. It might be futile for now, but in the future it      could help us and doesn't take much extra effort.

   You'll need to make sure you have the correct version of python you want installed. If you went through the steps above you should have 3.9.1 (or 3.9.0) already installed in pyenv.    Now we just need to make a virtualenv with that python version; I'm going to call it instrument_gen:
   ```
   pyenv virtualenv 3.9.1 instrument_gen
   pyenv activate instrument_gen
   ```
   If all the above went well, you should be able to see the following:
   ```
   which python        # returns /Users/your_username/.pyenv/shims/python
   python --version    # returns Python 3.9.1
   ```
4. Run steps 2-4 from the above help guide.
   These steps will install DAC, CLAP & MusicGen, and will run the test files. If your environment is set up correctly you should be able to do this smoothly.


If you made it this far, hopefully you've been able to set up an environment and run the test files for DAC, CLAP and MusicGen. Now we can start to think about training the model!

#### Simple venv (no conda or pyenv)
I wrote the above "If all else fails" before I wrote this section, but I realized using pyenv & such might be overkill. Here's a really simple way to set up a virtual environment on a Mac and get ready to install the packages above.
1. Install homebrew
   You might have it already but we need it for this option:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install virtualenv
   Let's install virtualenv. You might already have it, but just in case:
   ```
   pip install virtualenv
   ```
4. Navigate to folder where InstrumentGen will live
   In a terminal window, cd to the folder you want to store InstrumentGen. If you've already cloned it, great. If not, clone the repo there using the following command:
   ```
   git clone https://github.com/pdtraver/InstrumentGen.git
   ```
   You may need to login to git first. This you should be able to figure out on your own (you might need to make an SSH key, but you can find directions on github about this).
5. Make sure the correct python version is installed on your computer
   Since we're using homebrew and not pyenv, we're limited to the version of python homebrew can install, which is 3.9.20. Everything should still work with this, but we might run        into issues down the road. For now, though, let's just get you up and running:
   ```
   brew install python@3.9
   ```
6. Make a virtual environment in the same folder as InstrumentGen
   Within the InstrumentGen repo, run the following in the command line:
   ```
   virtualenv --python="/opt/homebrew/bin/python3.9" "./instrument_gen"
   ```
   A new folder inside the repo should be made. This is the folder with your virtual environment, now called instrument_gen. To activate this, run:
   ```
   source ./instrument_gen/bin/activate
   ```
   If you type the following, you'll see which python you're using:
   ```
   which python       # /Users/your_username/path/to/repo/InstrumentGen/instrument_gen/bin/python
   python --version   # Python 3.9.20
   ```

   Now like the above, you should run steps 2-4 of the original setup code to get DAC, CLAP and MusicGen. If the test files compile, you're good to go to next step!
   

### Getting ready to train
Luckily DAC and CLAP are frozen modules in InstrumentGen, so we do not need to train these. However, the MusicGen module we will need to train on the NSynth dataset. We have a few steps before we'll be ready to train, which include:
1. NSynth preparation (NSynth into DAC)
2. CLAP preparation (NSynth + text into CLAP)
3. RVQ on CLAP output (Quantizing CLAP embeddings)
4. MusicGen setup (Codebook interleaving patterns)
5. Timbral Consistency Loss (Implementing TC measure)

Each of these components will be important to matching the implementation provided in the paper. Some of the details (particularly regarding the text conditioning and TC measure) are a bit hazy as Native Instruments is a proprietary company, so we will likely have room for creativity and innovation on these fronts. Let's discuss at our meeting what people are most interested in and we can prepare for each subtask.
