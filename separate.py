#!/usr/bin/env python3
"""
Audio Stem Separation using Demucs

This script separates audio files into individual stems (vocals, drums, bass, other)
using Facebook's Demucs deep learning model.

Usage:
    python separate.py input.mp3 --stems bass,drums --output ./stems/
    python separate.py input.wav --model htdemucs_ft
    python separate.py input.mp3  # extracts all stems with default model
    python separate.py input.mp3 --start 1:30 --duration 15  # extract 15s starting at 1:30
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

# We'll handle import errors gracefully
try:
    import torch
    import torchaudio
except ImportError:
    print("Error: PyTorch and torchaudio are required.")
    print("Install with: pip install torch torchaudio")
    sys.exit(1)

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import save_audio
except ImportError:
    print("Error: Demucs is required.")
    print("Install with: pip install demucs")
    sys.exit(1)


# Available stems in Demucs models
AVAILABLE_STEMS = ["vocals", "drums", "bass", "other"]

# Available Demucs models (most common ones)
AVAILABLE_MODELS = [
    "htdemucs",       # Default hybrid transformer model
    "htdemucs_ft",    # Fine-tuned version (better quality, slower)
    "htdemucs_6s",    # 6-stem version (adds piano and guitar)
    "hdemucs_mmi",    # Hybrid with MMI training
    "mdx",            # MDX competition model
    "mdx_extra",      # MDX with extra training
    "mdx_q",          # Quantized MDX (faster, less memory)
    "mdx_extra_q",    # Quantized MDX extra
]


def validate_audio_file(file_path: Path) -> None:
    """
    Validate that the input file exists and has a supported format.
    
    Args:
        file_path: Path to the audio file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is not supported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    supported_formats = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    if file_path.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: {file_path.suffix}\n"
            f"Supported formats: {', '.join(supported_formats)}"
        )


def validate_stems(stems: list[str], model_name: str) -> list[str]:
    """
    Validate that requested stems are available in the model.
    
    Args:
        stems: List of stem names to extract
        model_name: Name of the Demucs model
        
    Returns:
        Validated list of stems
        
    Raises:
        ValueError: If any stem is not available
    """
    # For 6-stem model, add piano and guitar to available stems
    available = AVAILABLE_STEMS.copy()
    if "6s" in model_name:
        available.extend(["piano", "guitar"])
    
    invalid_stems = [s for s in stems if s not in available]
    if invalid_stems:
        raise ValueError(
            f"Invalid stem(s): {', '.join(invalid_stems)}\n"
            f"Available stems for {model_name}: {', '.join(available)}"
        )
    
    return stems


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def parse_time(time_str: str) -> float:
    """
    Parse a time string into seconds.
    
    Supports formats:
        - "30" or "30.5" -> 30 or 30.5 seconds
        - "1:30" -> 1 minute 30 seconds (90 seconds)
        - "1:30.5" -> 1 minute 30.5 seconds
    
    Args:
        time_str: Time string to parse
        
    Returns:
        Time in seconds as a float
        
    Raises:
        ValueError: If format is invalid
    """
    time_str = time_str.strip()
    
    if ":" in time_str:
        # Format: mm:ss or mm:ss.ms
        parts = time_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {time_str}. Use 'ss' or 'mm:ss'")
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use 'ss' or 'mm:ss'")
    else:
        # Format: ss or ss.ms
        try:
            return float(time_str)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use 'ss' or 'mm:ss'")


def get_device() -> torch.device:
    """
    Determine the best available device for processing.
    
    Returns:
        torch.device: CUDA if available, MPS for Apple Silicon, else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU acceleration
        return torch.device("mps")
    else:
        return torch.device("cpu")


def separate_audio(
    input_path: Path,
    output_dir: Path,
    stems: Optional[list[str]] = None,
    model_name: str = "htdemucs",
    device: Optional[torch.device] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> dict[str, Path]:
    """
    Separate an audio file into stems using Demucs.
    
    Args:
        input_path: Path to the input audio file
        output_dir: Directory to save the separated stems
        stems: List of stems to extract (None = all stems)
        model_name: Name of the Demucs model to use
        device: Torch device to use (None = auto-detect)
        start_time: Start time in seconds (None = from beginning)
        duration: Duration in seconds to process (None = until end)
        
    Returns:
        Dictionary mapping stem names to their output file paths
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    print(f"\n{'='*60}")
    print(f"Audio Stem Separation")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Model:  {model_name}")
    print(f"Device: {device}")
    
    # Show time range if specified
    if start_time is not None or duration is not None:
        start_str = format_time(start_time) if start_time else "0s"
        dur_str = format_time(duration) if duration else "end"
        print(f"Range:  {start_str} -> {dur_str}")
    
    print(f"{'='*60}\n")
    
    # Step 1: Load the model
    print(f"[1/4] Loading model '{model_name}'...")
    start_time = time.time()
    
    try:
        model = get_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    load_time = time.time() - start_time
    print(f"       Model loaded in {format_time(load_time)}")
    
    # Determine which stems to extract
    model_stems = model.sources  # Get stems available in this model
    if stems is None:
        stems_to_extract = list(model_stems)
    else:
        stems_to_extract = stems
    
    print(f"       Stems to extract: {', '.join(stems_to_extract)}")
    
    # Step 2: Load the audio file
    print(f"\n[2/4] Loading audio file...")
    start_time = time.time()
    
    try:
        # Load audio using torchaudio
        # Try different backends in order of preference
        backends_to_try = ["ffmpeg", "soundfile", "sox"]
        waveform = None
        last_error = None
        
        for backend in backends_to_try:
            try:
                waveform, sample_rate = torchaudio.load(input_path, backend=backend)
                break
            except Exception as e:
                last_error = e
                continue
        
        if waveform is None:
            raise last_error or RuntimeError("No suitable audio backend found")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")
    
    # Get audio duration for progress estimation
    duration = waveform.shape[1] / sample_rate
    print(f"       Duration: {format_time(duration)}")
    print(f"       Sample rate: {sample_rate} Hz")
    print(f"       Channels: {waveform.shape[0]}")
    
    # Resample if necessary (Demucs models typically expect 44100 Hz)
    if sample_rate != model.samplerate:
        print(f"       Resampling from {sample_rate} Hz to {model.samplerate} Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, model.samplerate)
        waveform = resampler(waveform)
        sample_rate = model.samplerate
    
    # Convert mono to stereo if necessary
    if waveform.shape[0] == 1:
        print("       Converting mono to stereo...")
        waveform = waveform.repeat(2, 1)
    
    # Slice the audio if start_time or duration is specified
    if start_time is not None or duration is not None:
        total_samples = waveform.shape[1]
        
        # Calculate start sample
        start_sample = 0
        if start_time is not None:
            start_sample = int(start_time * sample_rate)
            if start_sample >= total_samples:
                raise ValueError(
                    f"Start time ({format_time(start_time)}) is beyond audio duration "
                    f"({format_time(total_samples / sample_rate)})"
                )
        
        # Calculate end sample
        if duration is not None:
            end_sample = start_sample + int(duration * sample_rate)
            end_sample = min(end_sample, total_samples)  # Don't exceed file length
        else:
            end_sample = total_samples
        
        # Slice the waveform
        waveform = waveform[:, start_sample:end_sample]
        
        # Update duration for the sliced audio
        duration = (end_sample - start_sample) / sample_rate
        actual_start = start_sample / sample_rate
        print(f"       Extracted: {format_time(actual_start)} to {format_time(actual_start + duration)}")
        print(f"       Segment duration: {format_time(duration)}")
    
    load_time = time.time() - start_time
    print(f"       Audio loaded in {format_time(load_time)}")
    
    # Step 3: Perform separation
    print(f"\n[3/4] Separating stems (this may take a while)...")
    start_time = time.time()
    
    # Add batch dimension and move to device
    # Shape: (batch, channels, samples)
    audio_tensor = waveform.unsqueeze(0).to(device)
    
    # Estimate processing time based on audio duration and device
    if device.type == "cuda":
        estimated_factor = 0.5  # GPU is fast
    elif device.type == "mps":
        estimated_factor = 1.0  # Apple Silicon is reasonably fast
    else:
        estimated_factor = 3.0  # CPU is slower
    
    estimated_time = duration * estimated_factor
    print(f"       Estimated time: ~{format_time(estimated_time)}")
    print(f"       Processing...", end="", flush=True)
    
    with torch.no_grad():
        # Apply the model to separate sources
        # Output shape: (batch, sources, channels, samples)
        sources = apply_model(model, audio_tensor, device=device, progress=True)
    
    separation_time = time.time() - start_time
    print(f"\n       Separation completed in {format_time(separation_time)}")
    
    # Step 4: Save the stems
    print(f"\n[4/4] Saving stems...")
    start_time = time.time()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the base name of the input file (without extension)
    base_name = input_path.stem
    
    # Dictionary to store output paths
    output_paths = {}
    
    # sources shape: (1, num_sources, 2, samples)
    # Remove batch dimension
    sources = sources.squeeze(0)
    
    for idx, stem_name in enumerate(model_stems):
        if stem_name not in stems_to_extract:
            continue
        
        # Get the stem audio (shape: channels, samples)
        stem_audio = sources[idx].cpu()
        
        # Create output filename
        output_file = output_dir / f"{base_name}_{stem_name}.wav"
        
        # Save the stem as WAV
        save_audio(stem_audio, output_file, samplerate=sample_rate)
        
        output_paths[stem_name] = output_file
        print(f"       Saved: {output_file}")
    
    save_time = time.time() - start_time
    print(f"       All stems saved in {format_time(save_time)}")
    
    return output_paths


def parse_stems(stems_str: str) -> list[str]:
    """
    Parse comma-separated stem names.
    
    Args:
        stems_str: Comma-separated string of stem names (e.g., "bass,drums")
        
    Returns:
        List of stem names
    """
    stems = [s.strip().lower() for s in stems_str.split(",")]
    return [s for s in stems if s]  # Remove empty strings


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Separate audio files into stems using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python separate.py song.mp3
      Extract all stems with default model

  python separate.py song.mp3 --stems vocals,drums
      Extract only vocals and drums

  python separate.py song.mp3 --start 1:30 --duration 15
      Extract 15 seconds starting at 1:30

  python separate.py song.mp3 --duration 30
      Extract just the first 30 seconds

  python separate.py song.wav --model htdemucs_ft --output ./my_stems/
      Use fine-tuned model and custom output directory

Available stems: vocals, drums, bass, other
  (htdemucs_6s also supports: piano, guitar)

Available models: htdemucs, htdemucs_ft, htdemucs_6s, mdx, mdx_extra, mdx_q
        """,
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input audio file (MP3, WAV, FLAC, etc.)",
    )
    
    parser.add_argument(
        "-s", "--stems",
        type=str,
        default=None,
        help="Comma-separated list of stems to extract (e.g., 'bass,drums'). "
             "If not specified, all stems are extracted.",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./stems",
        help="Output directory for separated stems (default: ./stems)",
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="htdemucs",
        choices=AVAILABLE_MODELS,
        help="Demucs model to use (default: htdemucs)",
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing (even if GPU is available)",
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start time for extraction (e.g., '30' for 30s, '1:30' for 1m30s). "
             "Default: start of file.",
    )
    
    parser.add_argument(
        "-d", "--duration",
        type=str,
        default=None,
        help="Duration to extract (e.g., '15' for 15s, '0:30' for 30s). "
             "Default: until end of file.",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    # Handle --list-models flag
    if args.list_models:
        print("Available Demucs models:")
        print("-" * 40)
        for model in AVAILABLE_MODELS:
            print(f"  {model}")
        print("-" * 40)
        print("\nRecommended:")
        print("  htdemucs    - Good balance of speed and quality")
        print("  htdemucs_ft - Best quality, but slower")
        print("  mdx_extra_q - Fast with good quality")
        sys.exit(0)
    
    try:
        # Validate input file
        input_path = Path(args.input).resolve()
        validate_audio_file(input_path)
        
        # Parse and validate stems
        stems = None
        if args.stems:
            stems = parse_stems(args.stems)
            stems = validate_stems(stems, args.model)
        
        # Set up output directory
        output_dir = Path(args.output).resolve()
        
        # Determine device
        device = torch.device("cpu") if args.cpu else get_device()
        
        # Parse time arguments
        start_time = None
        duration = None
        if args.start:
            start_time = parse_time(args.start)
        if args.duration:
            duration = parse_time(args.duration)
        
        # Perform separation
        output_paths = separate_audio(
            input_path=input_path,
            output_dir=output_dir,
            stems=stems,
            model_name=args.model,
            device=device,
            start_time=start_time,
            duration=duration,
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("Separation complete!")
        print(f"{'='*60}")
        print(f"Output files:")
        for stem, path in output_paths.items():
            print(f"  {stem}: {path}")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
