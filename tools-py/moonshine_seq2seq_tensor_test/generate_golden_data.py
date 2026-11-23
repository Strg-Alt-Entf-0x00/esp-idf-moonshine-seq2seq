import argparse
import os
import numpy as np
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
from datetime import datetime

def write_bin(filepath, np_array):
    with open(filepath, 'wb') as f:
        f.write(np_array.astype('<f4', copy=False).tobytes())
    print(f"[OK] Wrote {filepath} (Shape: {np_array.shape}, dtype: {np_array.dtype})")

def main():
    parser = argparse.ArgumentParser(description="Generate Golden Data for Moonshine ASR")
    parser.add_argument("--wav", type=str, required=True, help="Input WAV file")
    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    parser.add_argument("--model", type=str, default=str(project_root / "models" / "py" / "moonshine-seq2seq-py-tiny-de"), help="Path to local HF Moonshine model")
    parser.add_argument("--outdir", type=str, help="Output directory for reference tensors (default: project .data/tensor_test)")
    args = parser.parse_args()

    model_label = Path(args.model).name
    output_dir = Path(args.outdir).resolve() if args.outdir else (
        project_root / ".data" / "tensor_test" / f"{timestamp}_{model_label}"
    ).resolve()
    data_root = (project_root / ".data").resolve()
    if not output_dir.is_relative_to(data_root):
        parser.error("--outdir must be inside the project .data directory")
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    args.outdir = str(output_dir)

    print(f"Loading model from: {args.model}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    print(f"Loading audio from: {args.wav}")
    audio, sr = sf.read(args.wav, dtype='float32', always_2d=True)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz")
    
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all() or not len(audio):
        raise ValueError("Audio must be non-empty and contain only finite values")

    print(f"Audio loaded: {len(audio)} samples")
    write_bin(os.path.join(args.outdir, "audio_input.bin"), audio.astype(np.float32))

    audio_tensor = torch.tensor(audio).unsqueeze(0)

    print("Running Encoder...")
    with torch.no_grad():
        encoder_out = model.model.encoder(audio_tensor)
        enc_hidden = encoder_out.last_hidden_state

    enc_hidden_np = enc_hidden.numpy().astype(np.float32)
    print(f"Encoder output shape: {enc_hidden_np.shape}")
    
    write_bin(os.path.join(args.outdir, "encoder_out.bin"), enc_hidden_np)

    print("Running Full Decode...")
    with torch.no_grad():
        out = model.generate(audio_tensor, max_new_tokens=50)
    
    transcript = processor.decode(out[0], skip_special_tokens=True)
    print(f"\nFinal Transcript: {transcript}")
    
    with open(os.path.join(args.outdir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)

    print("\nGolden data generation complete!")

if __name__ == '__main__':
    main()
