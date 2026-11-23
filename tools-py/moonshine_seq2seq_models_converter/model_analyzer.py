#!/usr/bin/env python3
"""Inspect PyTorch checkpoints and validate native ESPM models."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

if __package__:
    from .espm_format import read_espm
else:
    from espm_format import read_espm

# Optional imports with error handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available. PyTorch analysis disabled.")

try:
    from transformers import MoonshineForConditionalGeneration, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] transformers not available. Some features disabled.")

import numpy as np


# =============================================================================
# PyTorch Model Analyzer
# =============================================================================

class PyTorchAnalyzer:
    """Analyzer for PyTorch/Transformers models."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.config = None
        
    def load(self):
        """Load model from directory."""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise RuntimeError("PyTorch and transformers required for PyTorch analysis")
        
        print(f"Loading model from {self.model_dir}...")
        self.config = AutoConfig.from_pretrained(
            str(self.model_dir), local_files_only=True)
        self.model = MoonshineForConditionalGeneration.from_pretrained(
            str(self.model_dir), local_files_only=True)
        self.model.eval()
        print("Model loaded")
    
    def analyze_architecture(self) -> Dict:
        """Analyze model architecture."""
        if self.model is None:
            self.load()
        
        layers = []
        total_params = 0
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layers.append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': params,
                    })
                    total_params += params
        
        # Count operation types
        op_types = defaultdict(int)
        for name, module in self.model.named_modules():
            op_types[type(module).__name__] += 1
        
        return {
            'total_params': total_params,
            'total_params_m': total_params / 1e6,
            'num_layers': len(layers),
            'operation_types': dict(op_types),
            'layers': layers,
            'config': self.config.to_dict() if self.config else {},
        }
    
    def list_state_dict(self) -> List[Dict]:
        """List all tensors in model state dict."""
        if self.model is None:
            self.load()
        
        tensors = []
        for name, tensor in self.model.state_dict().items():
            tensors.append({
                'name': name,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'num_elements': tensor.numel(),
                'size_bytes': tensor.numel() * tensor.element_size(),
            })
        
        return tensors

    def list_deployment_state_dict(self):
        if self.model is None:
            self.load()
        return {
            (part, name): tensor.detach().cpu()
            for part, module in (
                (1, self.model.get_encoder()),
                (2, self.model.get_decoder()),
            )
            for name, tensor in module.state_dict().items()
        }


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_analyze_pytorch(args):
    """Analyze PyTorch model."""
    print("=" * 80)
    print("PyTorch Model Analysis")
    print("=" * 80)
    print()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1
    
    analyzer = PyTorchAnalyzer(model_dir)
    
    try:
        result = analyzer.analyze_architecture()
        
        print(f"Model: {model_dir.name}")
        print(f"Total Parameters: {result['total_params']:,} ({result['total_params_m']:.2f}M)")
        print(f"Layers: {result['num_layers']}")
        print()
        
        print("Operation Types:")
        for op_type, count in sorted(result['operation_types'].items(), key=lambda x: -x[1]):
            print(f"  {op_type:30} x {count}")
        print()
        
        if args.verbose:
            print("Layer Details:")
            for layer in result['layers'][:20]:  # Show first 20
                print(f"  {layer['name']:60} {layer['params']:>12,} params  ({layer['type']})")
            if len(result['layers']) > 20:
                print(f"  ... and {len(result['layers']) - 20} more layers")
            print()
        
        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Analysis saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_analyze_espm(args):
    """Validate and summarize a native ESPM container."""
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: Model file not found: {model_path}")
        return 1
    try:
        parsed = read_espm(model_path.read_bytes())
    except (OSError, ValueError) as error:
        print(f"ERROR: Invalid ESPM model: {error}")
        return 1

    quantization = "INT8" if parsed["quant_type"] == 1 else "INT8-per-channel"
    config = parsed["model_config"]
    print(f"ESPM: {model_path}")
    print(f"  Name: {parsed['model_name']}")
    print(f"  Quantization: {quantization}")
    print(f"  Size: {model_path.stat().st_size:,} bytes")
    print(f"  Resident PSRAM estimate: {parsed['psram_required_kb']:,} KiB")
    print(f"  Vocabulary: {config['vocab_size']} tokens")
    print(f"  Hidden size / layers: {config['hidden_size']} / "
          f"{config['encoder_layers']} encoder, {config['decoder_layers']} decoder")
    print(f"  Tensors: {len(parsed['tensors'])}")
    if args.list_tensors:
        for tensor in parsed["tensors"]:
            part = "encoder" if tensor["part"] == 1 else "decoder"
            print(f"  {part:7} {tensor['name']:<56} "
                  f"shape={tensor['shape']} scales={tensor['num_channels']}")
    if args.output:
        report = {
            "model_path": str(model_path),
            "model_name": parsed["model_name"],
            "quant_type": parsed["quant_type"],
            "psram_required_kb": parsed["psram_required_kb"],
            "model_config": config,
            "metadata": parsed["metadata"],
            "tensors": [
                {key: tensor[key] for key in
                 ("name", "part", "rank", "quant_type", "shape", "num_elements",
                  "num_channels", "channel_axis")}
                for tensor in parsed["tensors"]
            ] if args.list_tensors else [],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def cmd_compare(args):
    """Compare a PyTorch checkpoint with its native ESPM export."""
    print("=" * 80)
    print("PyTorch and ESPM Comparison")
    print("=" * 80)
    print()
    
    pytorch_dir = Path(args.pytorch)
    espm_path = Path(args.espm)
    
    if not pytorch_dir.is_dir():
        print(f"ERROR: PyTorch model not found: {pytorch_dir}")
        return 1
    
    if not espm_path.is_file():
        print(f"ERROR: ESPM model not found: {espm_path}")
        return 1
    
    try:
        parsed = read_espm(espm_path.read_bytes())
        analyzer = PyTorchAnalyzer(pytorch_dir)
        reference = analyzer.list_deployment_state_dict()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: Comparison failed: {error}")
        return 1

    exported = {
        (tensor["part"], tensor["name"]): tensor
        for tensor in parsed["tensors"]
    }
    missing = sorted(set(reference) - set(exported))
    unexpected = sorted(set(exported) - set(reference))
    mismatched_shapes = [
        (key, tuple(reference[key].shape), tuple(exported[key]["shape"]))
        for key in set(reference) & set(exported)
        if tuple(reference[key].shape) != tuple(exported[key]["shape"])
    ]

    squared_error = 0.0
    compared_values = 0
    max_error = 0.0
    for key in set(reference) & set(exported):
        source = reference[key].numpy().astype(np.float32, copy=False)
        target = exported[key]
        if tuple(source.shape) != tuple(target["shape"]):
            continue
        quantized = np.frombuffer(target["data"], dtype=np.int8).reshape(source.shape)
        scales = np.frombuffer(target["scales"], dtype="<f4")
        if target["channel_axis"] == 0xFF:
            restored = quantized.astype(np.float32) * scales[0]
        else:
            scale_shape = [1] * source.ndim
            scale_shape[target["channel_axis"]] = len(scales)
            restored = quantized.astype(np.float32) * scales.reshape(scale_shape)
        difference = np.abs(source - restored)
        squared_error += float(np.square(difference, dtype=np.float64).sum())
        compared_values += difference.size
        max_error = max(max_error, float(difference.max()))

    print(f"PyTorch tensors: {len(reference)}")
    print(f"ESPM tensors:    {len(exported)}")
    print(f"Compared values: {compared_values:,}")
    if compared_values:
        print(f"Quantization max absolute error: {max_error:.7g}")
        print(f"Quantization RMSE: {np.sqrt(squared_error / compared_values):.7g}")
    if missing:
        print(f"Missing ESPM tensors: {len(missing)}")
        for part, name in missing[:20]:
            print(f"  part={part} {name}")
    if unexpected:
        print(f"Unexpected ESPM tensors: {len(unexpected)}")
        for part, name in unexpected[:20]:
            print(f"  part={part} {name}")
    if mismatched_shapes:
        print(f"Tensor shape mismatches: {len(mismatched_shapes)}")
        for (part, name), expected, actual in mismatched_shapes[:20]:
            print(f"  part={part} {name}: PyTorch {expected}, ESPM {actual}")
    return int(bool(missing or unexpected or mismatched_shapes))


def cmd_memory(args):
    """Report storage and declared resident-memory requirements for ESPM."""
    print("=" * 80)
    print("ESPM Memory Profile")
    print("=" * 80)
    print()
    
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: ESPM model not found: {model_path}")
        return 1
    
    try:
        parsed = read_espm(model_path.read_bytes())
    except (OSError, ValueError) as error:
        print(f"ERROR: Invalid ESPM model: {error}")
        return 1
    tensor_data = sum(len(tensor["data"]) for tensor in parsed["tensors"])
    scales_data = sum(len(tensor["scales"]) for tensor in parsed["tensors"])
    params = sum(tensor["num_elements"] for tensor in parsed["tensors"])
    print(f"Model: {parsed['model_name']}")
    print(f"File size: {model_path.stat().st_size:,} bytes")
    print(f"INT8 tensor payload: {tensor_data:,} bytes")
    print(f"Quantization scales: {scales_data:,} bytes")
    print(f"Vocabulary: {len(parsed['vocab']):,} encoded bytes")
    print(f"Parameters: {params:,}")
    print(f"Declared PSRAM requirement: {parsed['psram_required_kb']:,} KiB")
    return 0


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze PyTorch checkpoints and native ESPM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze PyTorch model
  python model_analyzer.py pytorch --model-dir ../../original_models/moonshine-tiny
  
  # Validate ESPM and list tensor metadata
  python model_analyzer.py espm --model model.espm --list-tensors
  
  # Compare checkpoint tensors and quantization error
  python model_analyzer.py compare --pytorch ../models_py/moonshine-tiny --espm model.espm
  
  # Report storage and declared PSRAM requirements
  python model_analyzer.py memory --model model.espm
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis command')
    
    # PyTorch command
    pytorch_parser = subparsers.add_parser('pytorch', help='Analyze PyTorch model')
    pytorch_parser.add_argument('--model-dir', required=True, help='PyTorch model directory')
    pytorch_parser.add_argument('--output', '-o', help='Save analysis to JSON file')
    pytorch_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    espm_parser = subparsers.add_parser('espm', help='Validate and analyze an ESPM model')
    espm_parser.add_argument('--model', required=True, help='.espm model file')
    espm_parser.add_argument('--list-tensors', action='store_true', help='List tensor metadata')
    espm_parser.add_argument('--output', '-o', help='Save analysis to JSON file')
    
    compare_parser = subparsers.add_parser(
        'compare', help='Compare PyTorch tensors with an ESPM export')
    compare_parser.add_argument('--pytorch', required=True, help='PyTorch model directory')
    compare_parser.add_argument('--espm', required=True, help='.espm model file')
    
    memory_parser = subparsers.add_parser('memory', help='Report ESPM storage and PSRAM metadata')
    memory_parser.add_argument('--model', required=True, help='.espm model file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command handler
    if args.command == 'pytorch':
        return cmd_analyze_pytorch(args)
    elif args.command == 'espm':
        return cmd_analyze_espm(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'memory':
        return cmd_memory(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
