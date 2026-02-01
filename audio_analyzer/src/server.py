"""
Audio Analyzer MCP Server

Provides tools for stem separation and audio analysis that can be used by LLMs
to analyze music files.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Get the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SEPARATE_SCRIPT = PROJECT_ROOT / "separate.py"
ANALYZE_BASS_SCRIPT = PROJECT_ROOT / "analyze_bass.py"
DEFAULT_STEMS_DIR = PROJECT_ROOT / "stems"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# ─────────────────────────────────────────────────────────────
# Server Initialization
# ─────────────────────────────────────────────────────────────

mcp = FastMCP("audio-analyzer")


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def run_command(cmd: list[str], cwd: Optional[Path] = None) -> tuple[bool, str, str]:
    """
    Run a command and return (success, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for long operations
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out after 10 minutes"
    except Exception as e:
        return False, "", str(e)


# ─────────────────────────────────────────────────────────────
# Resources
# ─────────────────────────────────────────────────────────────

@mcp.resource("audio://analysis/{filename}")
def get_analysis(filename: str) -> str:
    """
    Get the analysis JSON for a previously analyzed stem.
    
    Args:
        filename: Name of the analysis file (e.g., "believe_it_bass_analysis.json")
    """
    analysis_path = DEFAULT_ANALYSIS_DIR / filename
    if not analysis_path.exists():
        return f"Analysis file not found: {filename}"
    
    with open(analysis_path) as f:
        return f.read()


@mcp.resource("audio://stems")
def list_stems() -> str:
    """List all available stem files."""
    stems = []
    for stems_dir in [DEFAULT_STEMS_DIR, PROJECT_ROOT / "stems_ft"]:
        if stems_dir.exists():
            for f in stems_dir.glob("*.wav"):
                stems.append(f"{stems_dir.name}/{f.name}")
    
    if not stems:
        return "No stems found. Use separate_audio to create stems first."
    
    return "Available stems:\n" + "\n".join(f"- {s}" for s in sorted(stems))


@mcp.resource("audio://analyses")
def list_analyses() -> str:
    """List all available analysis files."""
    if not DEFAULT_ANALYSIS_DIR.exists():
        return "No analyses found. Use analyze_bass (or other analyzers) first."
    
    analyses = []
    for f in DEFAULT_ANALYSIS_DIR.glob("*_analysis.json"):
        analyses.append(f.name)
    
    if not analyses:
        return "No analyses found."
    
    return "Available analyses:\n" + "\n".join(f"- {a}" for a in sorted(analyses))


# ─────────────────────────────────────────────────────────────
# Tools: Stem Separation
# ─────────────────────────────────────────────────────────────

@mcp.tool()
def separate_audio(
    input_path: str,
    stems: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = "htdemucs",
    start: Optional[str] = None,
    duration: Optional[str] = None,
) -> str:
    """
    Separate an audio file into individual stems (vocals, drums, bass, other).
    
    This uses Facebook's Demucs deep learning model for high-quality separation.
    
    Args:
        input_path: Path to the input audio file (MP3, WAV, FLAC, etc.)
        stems: Comma-separated stems to extract (e.g., "bass,drums"). 
               If not specified, extracts all stems.
        output_dir: Directory to save stems. Defaults to ./stems/
        model: Demucs model to use. Options:
               - htdemucs (default, good balance)
               - htdemucs_ft (best quality, slower)
               - htdemucs_6s (6 stems: adds piano, guitar)
        start: Start time for extraction (e.g., "30" or "1:30")
        duration: Duration to extract (e.g., "15" for 15 seconds)
    
    Returns:
        Status message with paths to created stem files
    """
    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_path
    
    if not input_file.exists():
        return f"Error: Input file not found: {input_file}"
    
    # Build command (always use CPU to avoid GPU memory issues)
    cmd = [sys.executable, str(SEPARATE_SCRIPT), str(input_file), "--cpu"]
    
    if stems:
        cmd.extend(["--stems", stems])
    
    if output_dir:
        cmd.extend(["--output", output_dir])
    else:
        cmd.extend(["--output", str(DEFAULT_STEMS_DIR)])
    
    cmd.extend(["--model", model])
    
    if start:
        cmd.extend(["--start", start])
    
    if duration:
        cmd.extend(["--duration", duration])
    
    # Run separation
    success, stdout, stderr = run_command(cmd)
    
    if success:
        return f"Separation complete!\n\n{stdout}"
    else:
        return f"Separation failed:\n{stderr}\n\n{stdout}"


# ─────────────────────────────────────────────────────────────
# Tools: Audio Analysis
# ─────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_bass(
    input_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Analyze a bass stem to extract musical features.
    
    Extracts:
    - Tempo (BPM) and beat positions
    - Dynamics (RMS energy envelope)
    - Spectral features (frequency range, brightness)
    - Note onsets
    
    Also generates visualization plots (spectrogram, waveform, etc.)
    
    Args:
        input_path: Path to the bass stem WAV file
        output_dir: Directory for output files. Defaults to ./analysis/
    
    Returns:
        Summary of the analysis with key findings
    """
    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_path
    
    if not input_file.exists():
        return f"Error: Input file not found: {input_file}"
    
    # Build command
    cmd = [sys.executable, str(ANALYZE_BASS_SCRIPT), str(input_file)]
    
    if output_dir:
        cmd.extend(["--output", output_dir])
    else:
        cmd.extend(["--output", str(DEFAULT_ANALYSIS_DIR)])
    
    # Run analysis
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        return f"Analysis failed:\n{stderr}\n\n{stdout}"
    
    # Load and summarize the analysis
    analysis_file = (Path(output_dir) if output_dir else DEFAULT_ANALYSIS_DIR) / f"{input_file.stem}_analysis.json"
    
    if analysis_file.exists():
        with open(analysis_file) as f:
            analysis = json.load(f)
        
        summary = f"""Bass Analysis Complete!

## Summary

**File**: {analysis['metadata']['file']}
**Duration**: {analysis['metadata']['duration_seconds']}s

### Rhythm
- **Tempo**: {analysis['rhythm']['tempo_bpm']} BPM
- **Beats detected**: {len(analysis['rhythm']['beat_times'])}

### Dynamics
- **Peak RMS**: {analysis['dynamics']['peak_rms']}
- **Average RMS**: {analysis['dynamics']['average_rms']}

### Spectral
- **Frequency range**: {analysis['spectral']['frequency_range_hz'][0]} - {analysis['spectral']['frequency_range_hz'][1]} Hz
- **Average brightness**: {analysis['spectral']['average_brightness_hz']} Hz

### Onsets
- **Notes/hits detected**: {analysis['onsets']['count']}

## Output Files
- {input_file.stem}_analysis.json (full data)
- {input_file.stem}_spectrogram.png
- {input_file.stem}_waveform_onsets.png
- {input_file.stem}_brightness.png
- {input_file.stem}_rms.png

Use get_full_analysis() to retrieve the complete JSON data."""
        
        return summary
    else:
        return f"Analysis ran but output file not found.\n\n{stdout}"


@mcp.tool()
def get_full_analysis(stem_name: str) -> str:
    """
    Get the complete analysis JSON for a stem.
    
    Args:
        stem_name: Base name of the stem (e.g., "believe_it_bass")
    
    Returns:
        The full JSON analysis data
    """
    analysis_file = DEFAULT_ANALYSIS_DIR / f"{stem_name}_analysis.json"
    
    if not analysis_file.exists():
        # Try without _analysis suffix
        analysis_file = DEFAULT_ANALYSIS_DIR / f"{stem_name}.json"
    
    if not analysis_file.exists():
        return f"Analysis not found for: {stem_name}\n\nAvailable analyses:\n{list_analyses()}"
    
    with open(analysis_file) as f:
        return f.read()


@mcp.tool()
def list_available_files() -> str:
    """
    List all available audio files, stems, and analyses in the project.
    
    Returns:
        Overview of available files for processing
    """
    output = ["# Available Files\n"]
    
    # Source audio files
    output.append("## Source Audio Files")
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = [f for f in PROJECT_ROOT.iterdir() 
                   if f.is_file() and f.suffix.lower() in audio_extensions]
    
    if audio_files:
        for f in sorted(audio_files):
            output.append(f"- {f.name}")
    else:
        output.append("- (none found in project root)")
    
    output.append("")
    
    # Stems
    output.append("## Separated Stems")
    for stems_dir in [DEFAULT_STEMS_DIR, PROJECT_ROOT / "stems_ft"]:
        if stems_dir.exists():
            stems = list(stems_dir.glob("*.wav"))
            if stems:
                output.append(f"\n### {stems_dir.name}/")
                for f in sorted(stems):
                    output.append(f"- {f.name}")
    
    if len(output) == 4:  # Only headers, no stems
        output.append("- (none found - use separate_audio first)")
    
    output.append("")
    
    # Analyses
    output.append("## Analyses")
    if DEFAULT_ANALYSIS_DIR.exists():
        analyses = list(DEFAULT_ANALYSIS_DIR.glob("*_analysis.json"))
        if analyses:
            for f in sorted(analyses):
                output.append(f"- {f.name}")
        else:
            output.append("- (none found - use analyze_bass first)")
    else:
        output.append("- (analysis directory doesn't exist)")
    
    return "\n".join(output)


# ─────────────────────────────────────────────────────────────
# Tools: Full Pipeline
# ─────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_song(
    input_path: str,
    stems: str = "bass",
    model: str = "htdemucs",
    start: Optional[str] = None,
    duration: Optional[str] = None,
) -> str:
    """
    Run the complete analysis pipeline: separate stems, then analyze each.
    
    This is a convenience tool that combines separation and analysis in one step.
    
    Args:
        input_path: Path to the input audio file
        stems: Comma-separated stems to extract and analyze (default: "bass")
        model: Demucs model (htdemucs, htdemucs_ft, htdemucs_6s)
        start: Start time for extraction
        duration: Duration to extract
    
    Returns:
        Combined results from separation and all analyses
    """
    results = []
    
    # Step 1: Separate
    results.append("# Step 1: Stem Separation\n")
    sep_result = separate_audio(
        input_path=input_path,
        stems=stems,
        model=model,
        start=start,
        duration=duration,
    )
    results.append(sep_result)
    
    if "failed" in sep_result.lower() or "error" in sep_result.lower():
        return "\n".join(results)
    
    # Step 2: Analyze each stem
    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_path
    
    base_name = input_file.stem
    stem_list = [s.strip() for s in stems.split(",")]
    
    for stem in stem_list:
        results.append(f"\n# Step 2: Analyze {stem.title()}\n")
        
        stem_file = DEFAULT_STEMS_DIR / f"{base_name}_{stem}.wav"
        
        if stem == "bass":
            analysis_result = analyze_bass(str(stem_file))
            results.append(analysis_result)
        else:
            results.append(f"(Analysis for {stem} not yet implemented - stem file created at {stem_file})")
    
    return "\n".join(results)


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────

@mcp.prompt()
def analyze_audio_prompt(audio_file: str) -> str:
    """
    Generate a prompt for comprehensive audio analysis.
    
    Args:
        audio_file: Path to the audio file to analyze
    """
    return f"""Please analyze the audio file: {audio_file}

## Steps to Follow

1. First, use `list_available_files()` to see what's already available
2. Use `separate_audio()` to split the song into stems (bass, drums, vocals, other)
3. Use `analyze_bass()` on the bass stem to get detailed musical analysis
4. Review the analysis data and provide musical insights

## Analysis to Provide

After running the tools, please provide:

1. **Tempo & Rhythm Analysis**
   - What is the tempo (BPM)?
   - What's the rhythmic feel (straight, swung, syncopated)?
   - How consistent is the tempo?

2. **Bass Character**
   - What frequency range does the bass occupy?
   - Is it bright/punchy or warm/round?
   - How does the dynamics (RMS) evolve through the track?

3. **Performance Observations**
   - How many notes/onsets were detected?
   - Is the playing busy or sparse?
   - Any notable patterns in the beat alignment?

4. **Production Notes**
   - What does the spectrogram reveal about the bass sound?
   - Any observations about the mix/processing?
"""


@mcp.prompt()
def compare_stems_prompt(stem1: str, stem2: str) -> str:
    """
    Generate a prompt for comparing two stems or analyses.
    """
    return f"""Please compare these two audio stems/analyses:

1. {stem1}
2. {stem2}

Use `get_full_analysis()` to retrieve the analysis data for each, then compare:

- Tempo/timing relationship
- Frequency range overlap or separation
- Dynamic contour similarities/differences
- Onset/attack alignment
- Any interesting interactions between the parts
"""


# ─────────────────────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
