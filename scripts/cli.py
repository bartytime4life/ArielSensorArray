import typer
from pathlib import Path
from typing import Optional

from .paths import RAW, PROCESSED, MODELS, OUTPUTS, LOGS
from .logging_utils import console
from .calibration import run_calibration
from .model_baseline import train_baseline, predict_baseline
from .diagnostics import plot_example_spectrum
from .submit import build_submission

app = typer.Typer(add_completion=True, help="SpectraMind V50 CLI (lightweight)")

@app.callback()
def main(
    raw_dir: Path = typer.Option(RAW, help="Directory with raw .npy cubes"),
    processed_dir: Path = typer.Option(PROCESSED, help="Processed data directory"),
    models_dir: Path = typer.Option(MODELS, help="Models directory"),
    outputs_dir: Path = typer.Option(OUTPUTS, help="Outputs directory"),
    logs_dir: Path = typer.Option(LOGS, help="Logs directory"),
):
    # Ensure dirs exist
    for p in (raw_dir, processed_dir, models_dir, outputs_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    console().print(f"🏁 Using RAW={raw_dir}, PROCESSED={processed_dir}, MODELS={models_dir}, OUTPUTS={outputs_dir}")

@app.command()
def calibrate(
    bias: Optional[Path] = typer.Option(None, help="Optional bias frame (.npy)"),
    dark: Optional[Path] = typer.Option(None, help="Optional dark frame (.npy)"),
    flat: Optional[Path] = typer.Option(None, help="Optional flat field (.npy)"),
    wave: Optional[Path] = typer.Option(None, help="Optional wavelength mapping (.npy)"),
):
    """Run spectroscopic calibration over RAW → PROCESSED/calibrated."""
    run_calibration(bias_path=bias, dark_path=dark, flat_path=flat, wave_path=wave)

@app.command()
def train(
    labels: Optional[Path] = typer.Option(None, help="Optional training labels .npy (N, n_wavelengths)"),
    alpha: float = typer.Option(1e-2, help="Ridge regularization strength"),
):
    """Train a tiny baseline (ridge or template, depending on label availability)."""
    train_baseline(labels_path=labels, alpha=alpha)

@app.command()
def predict():
    """Run inference and save per-cube spectra predictions."""
    predict_baseline()

@app.command()
def diagnose():
    """Generate quick-look figures for sanity checks."""
    plot_example_spectrum()

@app.command()
def submit():
    """Bundle predictions into submission.csv and submission.zip."""
    build_submission()

def _entry():
    app()

if __name__ == "__main__":
    _entry()