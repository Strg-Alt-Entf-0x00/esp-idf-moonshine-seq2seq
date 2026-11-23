#!/usr/bin/env python3
import json
import argparse
import sys
import os

def export_tokenizer(input_json, output_bin):
    print(f"Loading {input_json}...")
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
        
    vocab = data.get('model', {}).get('vocab', {})
    added_tokens = data.get('added_tokens', [])
    if not vocab:
        print("No vocab found in tokenizer.json!")
        return

    # Merge the base vocabulary with tokenizer entries outside model.vocab.
    tokens_by_id = {token_id: token_str for token_str, token_id in vocab.items()}
    for token in added_tokens:
        token_id = token.get('id')
        token_str = token.get('content')
        if token_id is not None and token_str is not None:
            tokens_by_id[token_id] = token_str

    sorted_vocab = sorted(tokens_by_id.items())
    
    max_id = sorted_vocab[-1][0]
    if max_id != len(sorted_vocab) - 1:
        print(f"WARNING: Vocab IDs are not contiguous! Max ID: {max_id}, Vocab Size: {len(sorted_vocab)}")
        
    print(f"Exporting {len(sorted_vocab)} tokens...")
    
    with open(output_bin, 'wb') as f:
        # We need to export them in ID order. If there are gaps, we'll write empty strings.
        current_id = 0
        for token_id, token_str in sorted_vocab:
            # Fill gaps
            while current_id < token_id:
                f.write(bytes([0]))
                current_id += 1
                
            # Replace magic tokens if necessary (e.g., space character ' ')
            # Moonshine uses ' ' (U+2581) for spaces
            token_bytes = token_str.encode('utf-8')
            length = len(token_bytes)
            
            if length < 128:
                f.write(bytes([length]))
            else:
                f.write(bytes([(length % 128) + 128, length // 128]))
                
            f.write(token_bytes)
            current_id += 1
            
    print(f"Done! Exported to {output_bin} ({os.path.getsize(output_bin)} bytes)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export Moonshine tokenizer to ESP32 binary")
    parser.add_argument('--input', required=True, help="Path to tokenizer.json")
    parser.add_argument('--output', required=True, help="Path to output vocab.bin")
    
    args = parser.parse_args()
    export_tokenizer(args.input, args.output)
