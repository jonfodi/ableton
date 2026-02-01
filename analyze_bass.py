#!/usr/bin/env python3
"""
Bass Analyzer for Synth Recreation

Extracts per-note features that map directly to Ableton synth parameters.
Outputs structured JSON ready for LLM interpretation.

Usage:
    python analyze_bass_v2.py input.wav
    python analyze_bass_v2.py input.wav --output ./analysis/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("Error: librosa required. Install with: pip install librosa")
    sys.exit(1)


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, return mono waveform and sample rate."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr


def detect_notes(y: np.ndarray, sr: int, min_note_length_ms: float = 50) -> list[dict]:
    """
    Detect note boundaries using onset detection.
    
    Returns list of notes with start/end times and audio segments.
    """
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    min_samples = int(sr * min_note_length_ms / 1000)
    
    notes = []
    for i, start in enumerate(onset_samples):
        # End is next onset or end of file
        if i + 1 < len(onset_samples):
            end = onset_samples[i + 1]
        else:
            end = len(y)
        
        # Skip if too short
        if end - start < min_samples:
            continue
        
        notes.append({
            "index": len(notes),
            "start_sample": int(start),
            "end_sample": int(end),
            "start_time": float(start / sr),
            "end_time": float(end / sr),
            "audio": y[start:end]
        })
    
    return notes


def extract_adsr(note_audio: np.ndarray, sr: int) -> dict:
    """
    Extract ADSR envelope parameters from a single note.
    
    Returns attack_ms, decay_ms, sustain_level (0-1), release_ms
    """
    # Compute amplitude envelope
    envelope = np.abs(note_audio)
    
    # Smooth the envelope
    hop = max(1, len(envelope) // 500)
    envelope_smooth = np.array([
        envelope[i:i+hop].mean() 
        for i in range(0, len(envelope), hop)
    ])
    
    if len(envelope_smooth) < 4:
        return {"attack_ms": 0, "decay_ms": 0, "sustain_level": 0, "release_ms": 0}
    
    # Normalize
    peak_val = envelope_smooth.max()
    if peak_val < 1e-6:
        return {"attack_ms": 0, "decay_ms": 0, "sustain_level": 0, "release_ms": 0}
    
    envelope_norm = envelope_smooth / peak_val
    
    # Find peak position
    peak_idx = np.argmax(envelope_norm)
    
    # Attack: start to peak
    attack_samples = peak_idx * hop
    attack_ms = (attack_samples / sr) * 1000
    
    # Find sustain level (average of middle 50% after peak)
    post_peak = envelope_norm[peak_idx:]
    if len(post_peak) > 4:
        mid_start = len(post_peak) // 4
        mid_end = 3 * len(post_peak) // 4
        sustain_level = float(post_peak[mid_start:mid_end].mean())
    else:
        sustain_level = float(post_peak.mean()) if len(post_peak) > 0 else 0
    
    # Decay: peak to sustain level
    decay_end_idx = peak_idx
    for i in range(peak_idx, len(envelope_norm)):
        if envelope_norm[i] <= sustain_level * 1.1:  # Within 10% of sustain
            decay_end_idx = i
            break
    
    decay_samples = (decay_end_idx - peak_idx) * hop
    decay_ms = (decay_samples / sr) * 1000
    
    # Release: last 20% of note (approximation)
    release_portion = len(envelope_norm) // 5
    if release_portion > 1:
        release_start = len(envelope_norm) - release_portion
        release_samples = release_portion * hop
        release_ms = (release_samples / sr) * 1000
    else:
        release_ms = 0
    
    return {
        "attack_ms": round(attack_ms, 1),
        "decay_ms": round(decay_ms, 1),
        "sustain_level": round(sustain_level, 3),
        "release_ms": round(release_ms, 1)
    }


def extract_pitch(note_audio: np.ndarray, sr: int) -> dict:
    """
    Extract fundamental frequency using YIN algorithm.
    """
    # Use pyin for better pitch tracking
    f0, voiced_flag, voiced_prob = librosa.pyin(
        note_audio, 
        fmin=30,  # Low E on bass
        fmax=400,  # Upper range for bass
        sr=sr
    )
    
    # Get median of voiced frames
    voiced_f0 = f0[voiced_flag]
    
    if len(voiced_f0) == 0:
        return {"fundamental_hz": None, "midi_note": None, "note_name": None, "confidence": 0}
    
    fundamental = float(np.median(voiced_f0))
    midi_note = int(round(librosa.hz_to_midi(fundamental)))
    note_name = librosa.midi_to_note(midi_note)
    confidence = float(voiced_prob[voiced_flag].mean()) if len(voiced_prob[voiced_flag]) > 0 else 0
    
    return {
        "fundamental_hz": round(fundamental, 1),
        "midi_note": midi_note,
        "note_name": note_name,
        "confidence": round(confidence, 2)
    }


def extract_harmonics(note_audio: np.ndarray, sr: int, fundamental_hz: float, num_harmonics: int = 8) -> dict:
    """
    Extract harmonic content relative to fundamental.
    
    Returns amplitude of each harmonic relative to fundamental (in dB).
    This maps directly to Operator's oscillator levels.
    """
    if fundamental_hz is None or fundamental_hz < 20:
        return {"harmonic_amplitudes_db": [], "brightness_estimate": None}
    
    # Compute spectrum
    n_fft = min(4096, len(note_audio))
    spectrum = np.abs(np.fft.rfft(note_audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Find amplitude at each harmonic
    harmonic_amps = []
    fundamental_amp = None
    
    for h in range(1, num_harmonics + 1):
        target_freq = fundamental_hz * h
        
        # Find closest bin
        bin_idx = np.argmin(np.abs(freqs - target_freq))
        
        # Average nearby bins for stability
        start = max(0, bin_idx - 2)
        end = min(len(spectrum), bin_idx + 3)
        amp = spectrum[start:end].mean()
        
        if h == 1:
            fundamental_amp = amp
        
        harmonic_amps.append(amp)
    
    # Convert to dB relative to fundamental
    if fundamental_amp and fundamental_amp > 1e-10:
        harmonic_db = []
        for amp in harmonic_amps:
            if amp > 1e-10:
                db = 20 * np.log10(amp / fundamental_amp)
                harmonic_db.append(round(db, 1))
            else:
                harmonic_db.append(-60)  # Floor
    else:
        harmonic_db = [0] * num_harmonics
    
    # Brightness estimate: weighted average of harmonic presence
    weights = np.array(range(1, num_harmonics + 1))
    amps_linear = np.array([10 ** (db / 20) for db in harmonic_db])
    brightness = float(np.average(weights, weights=amps_linear)) if amps_linear.sum() > 0 else 1.0
    
    return {
        "harmonic_amplitudes_db": harmonic_db,
        "brightness_estimate": round(brightness, 2)
    }


def extract_transient(note_audio: np.ndarray, sr: int) -> dict:
    """
    Analyze transient character - punchy vs soft.
    
    Peak/RMS ratio indicates transient punch.
    Higher = more punch = less compression needed in Ableton.
    """
    peak = np.abs(note_audio).max()
    rms = np.sqrt(np.mean(note_audio ** 2))
    
    if rms < 1e-10:
        return {"peak_to_rms_db": 0, "transient_character": "silent"}
    
    crest_factor = peak / rms
    crest_db = 20 * np.log10(crest_factor)
    
    # Interpret
    if crest_db > 12:
        character = "punchy"
    elif crest_db > 8:
        character = "moderate"
    else:
        character = "compressed"
    
    return {
        "peak_to_rms_db": round(crest_db, 1),
        "transient_character": character
    }


def analyze_note(note: dict, sr: int) -> dict:
    """Analyze a single note, extract all features."""
    audio = note["audio"]
    
    # Get pitch first (needed for harmonics)
    pitch = extract_pitch(audio, sr)
    
    return {
        "index": note["index"],
        "start_time": round(note["start_time"], 3),
        "duration_ms": round((note["end_time"] - note["start_time"]) * 1000, 1),
        "pitch": pitch,
        "envelope": extract_adsr(audio, sr),
        "harmonics": extract_harmonics(audio, sr, pitch["fundamental_hz"]),
        "transient": extract_transient(audio, sr)
    }


def aggregate_notes(analyzed_notes: list[dict]) -> dict:
    """
    Aggregate per-note analysis into typical values.
    
    This is what you'd feed to an LLM for synth recreation advice.
    """
    if not analyzed_notes:
        return {}
    
    # Collect values
    attacks = [n["envelope"]["attack_ms"] for n in analyzed_notes]
    decays = [n["envelope"]["decay_ms"] for n in analyzed_notes]
    sustains = [n["envelope"]["sustain_level"] for n in analyzed_notes]
    releases = [n["envelope"]["release_ms"] for n in analyzed_notes]
    
    fundamentals = [n["pitch"]["fundamental_hz"] for n in analyzed_notes if n["pitch"]["fundamental_hz"]]
    
    # Average harmonics
    all_harmonics = [n["harmonics"]["harmonic_amplitudes_db"] for n in analyzed_notes 
                     if n["harmonics"]["harmonic_amplitudes_db"]]
    if all_harmonics:
        avg_harmonics = np.mean(all_harmonics, axis=0).tolist()
        avg_harmonics = [round(h, 1) for h in avg_harmonics]
    else:
        avg_harmonics = []
    
    crests = [n["transient"]["peak_to_rms_db"] for n in analyzed_notes]
    
    return {
        "typical_envelope": {
            "attack_ms": round(np.median(attacks), 1),
            "decay_ms": round(np.median(decays), 1),
            "sustain_level": round(np.median(sustains), 3),
            "release_ms": round(np.median(releases), 1)
        },
        "pitch_range": {
            "lowest_hz": round(min(fundamentals), 1) if fundamentals else None,
            "highest_hz": round(max(fundamentals), 1) if fundamentals else None,
            "typical_hz": round(np.median(fundamentals), 1) if fundamentals else None
        },
        "typical_harmonics_db": avg_harmonics,
        "typical_transient": {
            "peak_to_rms_db": round(np.median(crests), 1),
            "character": "punchy" if np.median(crests) > 12 else "moderate" if np.median(crests) > 8 else "compressed"
        },
        "note_count": len(analyzed_notes)
    }


def analyze_bass(input_path: Path, output_dir: Path) -> dict:
    """Main analysis pipeline."""
    print(f"\nAnalyzing: {input_path.name}")
    print("-" * 50)
    
    # Load
    y, sr = load_audio(input_path)
    duration = len(y) / sr
    print(f"Loaded: {duration:.2f}s @ {sr}Hz")
    
    # Detect notes
    print("Detecting notes...")
    notes = detect_notes(y, sr)
    print(f"Found {len(notes)} notes")
    
    # Analyze each note
    print("Analyzing notes...")
    analyzed_notes = []
    for note in notes:
        analyzed = analyze_note(note, sr)
        analyzed_notes.append(analyzed)
    
    # Aggregate
    print("Aggregating...")
    aggregated = aggregate_notes(analyzed_notes)
    
    # Build output
    analysis = {
        "metadata": {
            "file": input_path.name,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr
        },
        "summary": aggregated,
        "notes": analyzed_notes
    }
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_analysis.json"
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print("-" * 50)
    
    # Print summary for quick review
    print("\nSUMMARY (for Ableton):")
    print(f"  Envelope: A={aggregated['typical_envelope']['attack_ms']}ms "
          f"D={aggregated['typical_envelope']['decay_ms']}ms "
          f"S={aggregated['typical_envelope']['sustain_level']:.0%} "
          f"R={aggregated['typical_envelope']['release_ms']}ms")
    print(f"  Harmonics (dB vs fundamental): {aggregated['typical_harmonics_db'][:5]}...")
    print(f"  Transient: {aggregated['typical_transient']['character']} "
          f"({aggregated['typical_transient']['peak_to_rms_db']}dB crest)")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze bass for synth recreation")
    parser.add_argument("input", type=str, help="Path to bass audio file")
    parser.add_argument("-o", "--output", type=str, default="./analysis", help="Output directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output).resolve()
    analyze_bass(input_path, output_dir)


if __name__ == "__main__":
    main()