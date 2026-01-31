# TUNES - Audio Stem Separation

Separate audio files into individual stems (vocals, drums, bass, other) using Facebook's Demucs deep learning model.

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** The first run will download the model weights (~1.5GB for htdemucs). This only happens once.

## Usage

### Basic Usage

```bash
# Separate all stems (vocals, drums, bass, other)
python separate.py song.mp3

# Output will be saved to ./stems/ by default
```

### Extract Specific Stems

```bash
# Extract only vocals
python separate.py song.mp3 --stems vocals

# Extract bass and drums
python separate.py song.mp3 --stems bass,drums

# Extract everything except vocals
python separate.py song.mp3 --stems drums,bass,other
```

### Custom Output Directory

```bash
python separate.py song.mp3 --output ./my_stems/
```

### Extract a Time Range

Process only a portion of the audio for faster results:

```bash
# Extract just the first 15 seconds
python separate.py song.mp3 --duration 15

# Extract 30 seconds starting at 1:30
python separate.py song.mp3 --start 1:30 --duration 30

# Extract from 2 minutes to the end
python separate.py song.mp3 --start 2:00

# Time formats: "30" = 30 seconds, "1:30" = 1 minute 30 seconds
```

This is useful for:
- Quick testing before processing a full song
- Analyzing a specific section (e.g., just the chorus)
- Faster iteration when you know where the interesting part is

### Use Different Models

```bash
# Fine-tuned model (better quality, slower)
python separate.py song.mp3 --model htdemucs_ft

# 6-stem model (adds piano and guitar)
python separate.py song.mp3 --model htdemucs_6s

# List all available models
python separate.py --list-models
```

### Force CPU Processing

```bash
python separate.py song.mp3 --cpu
```

## Available Models

| Model | Description |
|-------|-------------|
| `htdemucs` | Default hybrid transformer (recommended) |
| `htdemucs_ft` | Fine-tuned version (best quality, slower) |
| `htdemucs_6s` | 6 stems: vocals, drums, bass, other, piano, guitar |
| `mdx` | MDX competition model |
| `mdx_extra` | MDX with extra training |
| `mdx_q` | Quantized MDX (faster, less memory) |
| `mdx_extra_q` | Quantized MDX extra |

## Available Stems

- **vocals** - Main vocals and harmonies
- **drums** - Percussion and drums
- **bass** - Bass guitar and low-frequency instruments
- **other** - Everything else (guitars, synths, etc.)
- **piano** - Piano (only with `htdemucs_6s`)
- **guitar** - Guitar (only with `htdemucs_6s`)

## Performance Tips

1. **GPU Acceleration**: The script automatically uses CUDA (NVIDIA) or MPS (Apple Silicon) if available
2. **Memory**: If you run out of memory, try `mdx_q` or `mdx_extra_q` models
3. **Quality vs Speed**: `htdemucs_ft` gives best quality but is slower; `htdemucs` is a good balance

## Example Output

```
============================================================
Audio Stem Separation
============================================================
Input:  /path/to/song.mp3
Output: /path/to/stems
Model:  htdemucs
Device: mps
============================================================

[1/4] Loading model 'htdemucs'...
       Model loaded in 2.3s
       Stems to extract: vocals, drums, bass, other

[2/4] Loading audio file...
       Duration: 3m 45.2s
       Sample rate: 44100 Hz
       Channels: 2
       Audio loaded in 0.8s

[3/4] Separating stems (this may take a while)...
       Estimated time: ~3m 45.2s
       Processing...
       Separation completed in 2m 12.5s

[4/4] Saving stems...
       Saved: /path/to/stems/song_vocals.wav
       Saved: /path/to/stems/song_drums.wav
       Saved: /path/to/stems/song_bass.wav
       Saved: /path/to/stems/song_other.wav
       All stems saved in 1.2s

============================================================
Separation complete!
============================================================
```
