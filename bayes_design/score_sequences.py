import argparse
import torch
import os
import sys
import glob
import numpy as np
import pandas as pd

# Add the current directory to path so we can import bayes_design
sys.path.append(os.getcwd())

from bayes_design.utils import resolve_protein_input, build_aa_allowed_mask
from bayes_design.model import model_dict
from bayes_design.evaluate import metric_dict
from bayes_design.decode import decode_order_dict

# set export HF_HUB_OFFLINE=1
os.environ["HF_HUB_OFFLINE"] = "1"

def get_sequences_from_file(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            # Basic validation: check if mostly uppercase letters
            if all(c.isupper() or c == '-' for c in line):
                sequences.append(line)
    return sequences

def score_sequences(
    pdb_path,
    sequences_dir,
    model_name='bayes_design',
    decode_order='n_to_c',
    bayes_balance_factor=0.002,
    device=None,
    output_file=None
):
    """
    Score protein sequences against a structure using BayesDesign.
    
    Args:
        pdb_path (str): Path to local PDB file
        sequences_dir (str): Path to directory containing subdirectories with sequence files
        model_name (str): Name of the model to use (default: 'bayes_design')
        decode_order (str): Decode order strategy (default: 'n_to_c')
        bayes_balance_factor (float): Bayes balance factor (default: 0.002)
        device (str): Device to use, e.g. 'cuda:0' or 'cpu' (default: auto-detect)
        output_file (str): Optional path to write results to CSV file
    
    Returns:
        pd.DataFrame: DataFrame with columns [File_Path, Sequence_Index, Log_Prob]
                     and summary statistics (Mean, Std, Count per file)
    """
    # Device setup
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load protein structure
    try:
        # Create a mock args object for resolve_protein_input
        class Args:
            pass
        args = Args()
        args.pdb_path = pdb_path
        args.protein_id = None
        seq_struct, struct = resolve_protein_input(args)
    except Exception as e:
        print(f"Error loading PDB: {e}")
        raise

    # Initialize model
    print(f"Initializing {model_name}...")
    if model_name == 'bayes_design':
        prob_model = model_dict[model_name](device=device, bayes_balance_factor=bayes_balance_factor)
    elif model_name == 'protein_mpnn':
        prob_model = model_dict[model_name](device=device)
    else:
        prob_model = model_dict[model_name](device=device)

    # We want to score the entire sequence, so nothing is "fixed"
    fixed_position_mask = np.zeros(len(seq_struct))
    mask_type = 'bidirectional_autoregressive' 
    decode_order = decode_order_dict[decode_order](seq_struct)

    input_path = os.path.abspath(sequences_dir)
    input_pardirname = os.path.basename(input_path)
    seq_paths = glob.glob(os.path.join(input_path, '**', '*.fasta'), recursive=True) + \
                glob.glob(os.path.join(input_path, '**', '*.txt'), recursive=True)
    
    # Collect results
    results = []
    
    for seq_path in seq_paths:
        print(f"Processing {seq_path}...")
        try:
            sequences = get_sequences_from_file(seq_path)
            if not sequences:
                print(f"  No sequences found in {seq_path}. Skipping.")
                continue
                
            for i, input_seq in enumerate(sequences):
                if len(input_seq) != len(seq_struct):
                    print(f"  Seq {i}: Length mismatch ({len(input_seq)} vs {len(seq_struct)}). Skipping.")
                    continue
                    
                score = metric_dict['log_prob'](
                    seq=input_seq,
                    prob_model=prob_model,
                    decode_order=decode_order,
                    structure=struct,
                    fixed_position_mask=fixed_position_mask,
                    mask_type=mask_type,
                    aa_allowed_mask=None
                )

                seq_pardirname = os.path.basename(os.path.dirname(seq_path))
                seq_id = seq_pardirname if seq_pardirname != input_pardirname else os.path.basename(seq_path)
                
                results.append({
                    'File_Path': seq_id,
                    'Sequence_Index': i,
                    'Log_Prob': round(score, 4)
                })
                
        except Exception as e:
            print(f"  Error processing {seq_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add summary statistics per file
    if len(df) > 0:
        summary_stats = df.groupby('File_Path')['Log_Prob'].agg(['mean', 'std', 'count']).reset_index()
        summary_stats.columns = ['File_Path', 'Mean', 'Std', 'Count']
        print("\nSummary Statistics by File:")
        print(summary_stats.to_string(index=False))
    
    # Write to CSV if output_file is provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results written to {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Calculate probability of structure given sequence using BayesDesign.")
    parser.add_argument('--pdb_path', help="Path to local PDB file", required=True)
    parser.add_argument('--sequences_dir', help="Path to directory containing subdirectories with sequence files", required=True)
    parser.add_argument('--output_file', help="Path to output file for log probabilities (optional)", default=None)
    parser.add_argument('--model_name', default='bayes_design', choices=list(model_dict.keys()))
    parser.add_argument('--decode_order', default='n_to_c', choices=list(decode_order_dict.keys()))
    parser.add_argument('--bayes_balance_factor', default=0.002, type=float)
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()

    # Call score_sequences with parsed arguments
    df = score_sequences(
        pdb_path=args.pdb_path,
        sequences_dir=args.sequences_dir,
        model_name=args.model_name,
        decode_order=args.decode_order,
        bayes_balance_factor=args.bayes_balance_factor,
        device=args.device,
        output_file=args.output_file
    )
    
    return df


if __name__ == "__main__":
    main()
