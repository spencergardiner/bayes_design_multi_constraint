import argparse
import torch
import os
import sys
import glob
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description="Calculate probability of structure given sequence using to BayesDesign.")
    parser.add_argument('--pdb_path', help="Path to local PDB file", required=True)
    parser.add_argument('--sequences_dir', help="Path to directory containing subdirectories with sequence files", required=True)
    parser.add_argument('--output_file', help="Path to output file for log probabilities", required=True)
    parser.add_argument('--model_name', default='bayes_design', choices=list(model_dict.keys()))
    parser.add_argument('--decode_order', default='n_to_c', choices=list(decode_order_dict.keys()))
    parser.add_argument('--bayes_balance_factor', default=0.002, type=float)
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load protein structure
    try:
        # resolve_protein_input expects args.protein_id or args.pdb_path and sets args.protein_label
        if not hasattr(args, 'protein_id'):
            args.protein_id = None
        seq_struct, struct = resolve_protein_input(args)
    except Exception as e:
        print(f"Error loading PDB: {e}")
        return

    # Initialize model
    print(f"Initializing {args.model_name}...")
    if args.model_name == 'bayes_design':
        prob_model = model_dict[args.model_name](device=device, bayes_balance_factor=args.bayes_balance_factor)
    elif args.model_name == 'protein_mpnn':
        prob_model = model_dict[args.model_name](device=device)
    else:
        prob_model = model_dict[args.model_name](device=device)

    # We want to score the entire sequence, so nothing is "fixed"
    fixed_position_mask = np.zeros(len(seq_struct))
    mask_type = 'bidirectional_autoregressive' 
    decode_order = decode_order_dict[args.decode_order](seq_struct)

    input_path = os.path.abspath(args.sequences_dir)
    input_pardirname = os.path.basename(input_path)
    seq_paths = glob.glob(os.path.join(input_path, '**', '*.fasta'), recursive=True) + \
                glob.glob(os.path.join(input_path, '**', '*.txt'), recursive=True)
    
    
    with open(args.output_file, 'w') as out_f:
        out_f.write("File_Path,Sequence_Index,Log_Prob\n")
        
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
                    
                    out_f.write(f"{seq_id},{i},{score:.4f}\n")
                    # Flush to ensure data is written if script crashes
                    out_f.flush()
                    
            except Exception as e:
                print(f"  Error processing {seq_path}: {e}")

    print(f"Done. Results written to {args.output_file}")

if __name__ == "__main__":
    main()
