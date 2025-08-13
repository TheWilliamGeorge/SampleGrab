# SampleGrab
### Simple, Universal Instrument Sampler
Available for Windows, MacOS and Linux
Python version available

## Versions
MacOS (Intel and ARM)
Linux
Python Universal (requires XXXXXXXX)

A Windows build is under development, so for now, please use the Python Universal version ([SampleGrab.py](https://github.com/TheWilliamGeorge/SampleGrab/blob/main/SampleGrab.py))

Python dependencies:
```bash
 pip install PySide6 sounddevice soundfile numpy
```

## Installation
#### MacOS: 
Unzip the file and double-click SampleGrab.app
#### Linux:
Unzip the file and run the .bin file. Requires PortAudio. If you do not have this installed, run: 
```bash
sudo apt-get install libportaudio2
```


## Features
- Keyboard layout with labelled keys for intuitive recording
- Click-to-record functionality on each key to automatically assign the pitch to the sound
- Algorithm to determine when the sound begins and automatically trim the start of the sample 
- Adjustable buffer time at start of sample to pick up any sounds such as keys being pressed, strings being plucked etc 
- Each key can have as many samples recorded as you want 
- Project system to manage each set of samples 
- Sample player to play back and delete recorded samples 
- Export all samples to .zip 
- Completely free and open-source


## The story of SampleGrab
Recently, I was clearing out some old stuff and came across an old keyboard I had as a child. The volume slider was semi-functional and I didn’t really need it anymore (I have a full size piano now), but the sounds were very nostalgic, and I felt that it would be a shame to lose them (I couldn’t find them anywhere on the internet). So, I decided to sample the entire sound bank. 

Sampling 100 sounds over three octaves was going to take a long time, and I didn’t want to manually record each sound, trim it and name it. So, I began to search for a piece of software that would allow me to sample faster and easier. But I couldn’t find any such software. 

So, as any good programmer does, I decided to make it myself…
Built with the PySide6 wrapper for the QT GUI framework, SampleGrab is lightweight and uses built-in system UI elements for maximum performance. Contributions are welcome! 


## Future development
- [ ] Windows version
- [x] Assign Root Note Metadata for simple importing to samplers
- [ ] Waveforms
- [ ] Set default name for samples
- [ ] When default names are enabled, numbers applied to multi-samples
