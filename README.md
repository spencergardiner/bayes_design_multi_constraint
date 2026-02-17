# BayesDesign
<img src="https://github.com/dellacortelab/bayes_design/blob/master/data/figs/bayes_design.png?raw=true" alt="drawing" width="700"/>

BayesDesign is an algorithm for designing proteins with high stability and conformational specificity. See [preprint here](https://www.biorxiv.org/content/10.1101/2022.12.28.521825v1?rss=1).

Try out the BayesDesign model here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/bayes_design/blob/master/examples/BayesDesign.ipynb)

Dependencies: `./dependencies/requirements.txt`.

## One-line sequence design
To design a protein sequence to fit a protein backbone:
```
python3 design.py --protein_id 6MRR --model_name bayes_design --decode_order n_to_c --decode_algorithm beam --n_beams 128 --design_regions '{"fixed_region": {"positions": "67-68"}}'
```

### Using a local PDB file
Instead of a PDB ID, you can pass a local PDB file directly:
```
python3 design.py --pdb_path ./data/structures/my_protein.pdb --model_name bayes_design --decode_algorithm beam --n_beams 128
```

### Using a JSON config file
All arguments can also be specified in a JSON config file. JSON values override any CLI arguments (with a warning when conflicts are detected):
```
python3 design.py --config examples/config_example.json
```

See `examples/config_example.json` for a full example.

## Design Regions
Design regions specify which residue positions to redesign and (optionally) which amino acids are allowed or excluded at those positions. Positions **not** in any design region are kept fixed (unchanged from the original PDB sequence).

### Position syntax
Positions are specified as comma-separated values and/or inclusive ranges (1-indexed):
- Single position: `17`
- Range: `63-96`
- Mixed: `3,5-7,17,88-91`
- Spaces are ignored: `3, 5 - 7, 17`

### Per-region amino acid constraints
Each region can optionally specify **one** of:
- `allowed_aas`: only these amino acids are considered (e.g. `"DEHKNQRST"`)
- `excluded_aas`: these amino acids are excluded, all others are allowed (e.g. `"C"`)

If neither is specified, all 20 standard amino acids are allowed.

### Example: multi-region design
```
python3 design.py --protein_id 6MRR --design_regions '{
    "loop1":       {"positions": "63-96", "excluded_aas": "C"},
    "active_site": {"positions": "12,14,17", "allowed_aas": "DEHKNQRST"},
    "helix_cap":   {"positions": "30-35,40-45"}
}'
```

## Detailed steps to run with Docker
- Clone repository
```
git clone https://github.com/dellacortelab/bayes_design.git
```
- Build docker image (should take ~5 minutes)
```
docker build -t bayes_design -f ./bayes_design/dependencies/Dockerfile ./bayes_design/dependencies
```
- Run container
```
docker run -dit --gpus all --name bayes_dev --rm -v $(pwd)/bayes_design:/code -v $(pwd)/bayes_design/data:/data bayes_design
docker exec -it bayes_dev /bin/bash
```
- Redesign a protein backbone
```
cd ./code && python3 design.py --protein_id 6MRR --model_name bayes_design --decode_order n_to_c --decode_algorithm beam --n_beams 128 --design_regions '{"fixed_region": {"positions": "67-68"}}'
```
On a V100 GPU, the greedy algorithm predicts ~10 residues/s and beam search with 128 beams predicts 1 residue every 2s.

## Citation
```
@Article{Stern2023,
author={Stern, Jacob A. and Free, Tyler J. and Stern, Kimberlee L. and Gardiner, Spencer and Dalley, Nicholas A. and Bundy, Bradley C. and Price, Joshua L. and Wingate, David and Della Corte, Dennis},
title={A probabilistic view of protein stability, conformational specificity, and design},
journal={Scientific Reports},
year={2023},
volume={13},
number={1},
pages={15493},
issn={2045-2322},
doi={10.1038/s41598-023-42032-1},
url={https://doi.org/10.1038/s41598-023-42032-1}
}
```


# Experiments on designed sequences
Evaluate the probability of a designed sequence under a probability model
```
python3 experiment.py compare_seq_metric --metric log_prob --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --design_regions '{"fixed_34": {"positions": "34"}}' --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG
```

### Example experiment commands

Make a PSSM from designed sequences:
```
python3 experiment.py make_pssm --sequences_path ./results/bayes_design_1PIN_sequences.txt --pssm_path ./results/bayes_design_1PIN_pssm.pkl
```

Make a histogram of sequence scores:
```
python3 experiment.py make_hist --protein_id 1PIN --model_name pssm --pssm_path ./results/bayes_design_1PIN_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "34"}}' --sequences_path ./results/bayes_design_1PIN_sequences.txt
```

Filter top sequences by score:
```
python3 experiment.py seq_filter --protein_id 1PIN --model_name pssm --pssm_path ./results/bayes_design_1PIN_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "34"}}' --sequences_path ./results/bayes_design_1PIN_sequences.txt
```

BayesDesign WW
python3 experiment.py seq_filter --protein_id 1PIN --model_name pssm --pssm_path ./results/bayes_design_1PIN_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "34"}}' --sequences_path ./results/bayes_design_1PIN_sequences.txt --n_seqs 3

Result: [(-60.534150819590586, 'MLPQGWQAKQDRDTNQWVYRNWITNKITFNKPRG'), (-62.07557004996536, 'KLPEGWIETKDVIHGKTQYHNVNLNETMEEQPVG'), (-62.343200271562175, 'ALIEVWQKQKDPETGQTKYLNVGKGERTPQRPKG')]

ProteinMPNN WW
python3 experiment.py seq_filter --protein_id 1PIN --model_name pssm --pssm_path ./results/protein_mpnn_1PIN_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "34"}}' --sequences_path ./results/protein_mpnn_1PIN_sequences.txt --n_seqs 3

Result: [(-43.503443719475136, 'ALPTGWEEKIDPVTNQLIYYNVKTKETTTEKPVG'), (-43.567347443909554, 'ELPEGWVEMVDIKTGEVVYYNDITKEITKEKPVG'), (-45.31501494556361, 'ALPAGWEEIIDPETGKVQYYNSQTKEVTTARPIG')]


ProteinMPNN NanoLuc Full Redesign
python3 design.py --model_name protein_mpnn --protein_id nanoluc --decode_order n_to_c --decode_algorithm sample --temperature 1. --n_designs 1000 --design_regions '{"full": {"positions": "10-179"}}'

python3 experiment.py make_pssm --pssm_path ./results/protein_mpnn_nanoluc_full_pssm.pkl --sequences_path ./results/protein_mpnn_nanoluc_full_sequences.txt

python3 experiment.py seq_filter --protein_id nanoluc --model_name pssm --pssm_path ./results/protein_mpnn_nanoluc_full_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"full": {"positions": "10-179"}}' --sequences_path ./results/protein_mpnn_nanoluc_full_sequences.txt --n_seqs 10



ProteinMPNN NanoLuc Partial Redesign
python3 design.py --model_name protein_mpnn --protein_id nanoluc --decode_order n_to_c --decode_algorithm sample --temperature 1. --n_designs 1000 --design_regions '{"region1": {"positions": "63-96"}}'

python3 experiment.py make_pssm --pssm_path ./results/protein_mpnn_nanoluc_partial_pssm.pkl --sequences_path ./results/protein_mpnn_nanoluc_partial_sequences.txt

python3 experiment.py seq_filter --protein_id nanoluc --model_name pssm --pssm_path ./results/protein_mpnn_nanoluc_partial_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "63-96"}}' --sequences_path ./results/protein_mpnn_nanoluc_partial_sequences.txt --n_seqs 10



BayesDesign NanoLuc Full Redesign
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order n_to_c --decode_algorithm sample --temperature 1. --n_designs 1000 --design_regions '{"full": {"positions": "10-179"}}'

python3 experiment.py make_pssm --pssm_path ./results/bayes_design_nanoluc_full_pssm.pkl --sequences_path ./results/bayes_design_nanoluc_full_sequences.txt

python3 experiment.py seq_filter --protein_id nanoluc --model_name pssm --pssm_path ./results/bayes_design_nanoluc_full_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"full": {"positions": "10-179"}}' --sequences_path ./results/bayes_design_nanoluc_full_sequences.txt --n_seqs 10



BayesDesign NanoLuc Partial Redesign
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order n_to_c --decode_algorithm sample --temperature 1. --n_designs 1000 --design_regions '{"region1": {"positions": "63-96"}}'

python3 experiment.py make_pssm --pssm_path ./results/bayes_design_nanoluc_partial_pssm.pkl --sequences_path ./results/bayes_design_nanoluc_partial_sequences.txt

python3 experiment.py seq_filter --protein_id nanoluc --model_name pssm --pssm_path ./results/bayes_design_nanoluc_partial_pssm.pkl --metric log_prob --decode_order n_to_c --design_regions '{"region1": {"positions": "63-96"}}' --sequences_path ./results/bayes_design_nanoluc_partial_sequences.txt --n_seqs 10


ProteinMPNN NanoLuc Partial Redesign
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order n_to_c --decode_algorithm max_prob_decode --n_designs 1 --design_regions '{"region1": {"positions": "48-53"}, "region2": {"positions": "84-92"}, "region3": {"positions": "168-171"}}'