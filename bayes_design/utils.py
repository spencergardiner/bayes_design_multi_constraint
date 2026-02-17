import os
import re
import json
import warnings
import torch
from .protein_mpnn.protein_mpnn_utils import parse_PDB, StructureDatasetPDB

from Bio.PDB import PDBParser
import numpy as np

AMINO_ACID_ORDER = 'ACDEFGHIKLMNPQRSTVWYX'
AMINO_ACID_SET = set(AMINO_ACID_ORDER[:-1])  # 20 standard amino acids (no 'X')

####################################################################################
# Protein Loading
####################################################################################

def get_protein(pdb_code=None, pdb_path=None, structures_dir='./data/structures'):
    """Get a sequence in string format and 4-atom protein structure in L x 4 x 3
    tensor format (with atoms in N CA CB C order).

    Args:
        pdb_code (str, optional): PDB ID to fetch from RCSB if no local file exists.
        pdb_path (str, optional): Path to a local PDB file to load directly.
        structures_dir (str): Directory to download PDB files into when using pdb_code.
    """
    if pdb_path is not None:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: '{pdb_path}'")
        resolved_path = pdb_path
    elif pdb_code is not None:
        resolved_path = os.path.join(structures_dir, pdb_code + '.pdb')
        if not os.path.exists(resolved_path):
            os.makedirs(structures_dir, exist_ok=True)
            os.system(f"cd {structures_dir} && wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    else:
        raise ValueError(
            "Either pdb_code or pdb_path must be provided. "
            "Use pdb_code to fetch a structure from the RCSB PDB (e.g. '6MRR'), "
            "or pdb_path to load a local PDB file."
        )
    chain_list = ['A']
    pdb_dict_list = parse_PDB(resolved_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    protein = dataset_valid[0]
    struct = torch.tensor([protein['coords_chain_A']['N_chain_A'], protein['coords_chain_A']['CA_chain_A'], protein['coords_chain_A']['C_chain_A'], protein['coords_chain_A']['O_chain_A']]).transpose(0, 1)
    return protein['seq'], struct


def resolve_protein_input(args):
    """Validate and resolve protein input from args, returning (seq, struct).

    Checks args.protein_id and args.pdb_path.  At least one must be provided.
    If both are provided, pdb_path takes priority (with a warning).
    Also sets args.protein_label for use in output filenames.
    """
    protein_id = getattr(args, 'protein_id', None)
    pdb_path = getattr(args, 'pdb_path', None)

    if protein_id is None and pdb_path is None:
        raise ValueError(
            "Either --protein_id or --pdb_path must be provided.\n"
            "  --protein_id  : a PDB ID to fetch from RCSB (e.g. '6MRR')\n"
            "  --pdb_path    : path to a local PDB file (e.g. './data/structures/my_protein.pdb')"
        )

    if protein_id is not None and pdb_path is not None:
        warnings.warn(
            f"Both --protein_id ('{protein_id}') and --pdb_path ('{pdb_path}') provided. "
            f"Using --pdb_path '{pdb_path}'."
        )

    if pdb_path is not None:
        args.protein_label = os.path.splitext(os.path.basename(pdb_path))[0]
        seq, struct = get_protein(pdb_path=pdb_path)
    else:
        args.protein_label = protein_id
        seq, struct = get_protein(pdb_code=protein_id)

    return seq, struct


####################################################################################
# Position Parsing
####################################################################################

def parse_position_string(position_str):
    """Parse a position selection string into a sorted list of unique 0-indexed positions.

    Accepts comma-separated individual positions and/or inclusive ranges.
    All values are 1-indexed in the input and converted to 0-indexed on output.
    Whitespace is stripped before parsing.

    Examples:
        '3,5-7,2,17,88-91'  -> [1, 2, 4, 5, 6, 16, 87, 88, 89, 90]
        '5 - 7, 3'          -> [2, 4, 5, 6]

    Args:
        position_str (str): Position selector string.

    Returns:
        list[int]: Sorted unique 0-indexed positions.

    Raises:
        ValueError: On invalid characters or malformed tokens.
    """
    EXPECTED_FORMAT = (
        "Expected format: comma-separated positions and/or inclusive ranges, "
        "e.g. '3,5-7,2,17,88-91'"
    )
    # Strip all whitespace
    cleaned = re.sub(r'\s+', '', position_str)
    if not cleaned:
        return []

    # Check for invalid characters (only digits, commas, hyphens allowed)
    invalid = re.findall(r'[^0-9,\-]', cleaned)
    if invalid:
        raise ValueError(
            f"Invalid character '{invalid[0]}' in position selector '{position_str}'. "
            f"{EXPECTED_FORMAT}"
        )

    positions = set()
    tokens = cleaned.split(',')
    for token in tokens:
        if not token:
            continue
        if '-' in token:
            parts = token.split('-')
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"Malformed range '{token}' in position selector '{position_str}'. "
                    f"Ranges must be 'start-stop' with both endpoints specified. "
                    f"{EXPECTED_FORMAT}"
                )
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(
                    f"Range start ({start}) > range end ({end}) in '{token}' "
                    f"in position selector '{position_str}'. "
                    f"{EXPECTED_FORMAT}"
                )
            if start < 1:
                raise ValueError(
                    f"Position values must be >= 1 (1-indexed), got {start} "
                    f"in position selector '{position_str}'. "
                    f"{EXPECTED_FORMAT}"
                )
            # Convert 1-indexed inclusive range to 0-indexed
            positions.update(range(start - 1, end))
        else:
            val = int(token)
            if val < 1:
                raise ValueError(
                    f"Position values must be >= 1 (1-indexed), got {val} "
                    f"in position selector '{position_str}'. "
                    f"{EXPECTED_FORMAT}"
                )
            positions.add(val - 1)  # Convert to 0-indexed

    return sorted(positions)


####################################################################################
# Design Region Constraints
####################################################################################

def build_aa_allowed_mask(design_regions, seq_len):
    """Build position-level masks from a design_regions specification.

    Positions NOT in any design region are treated as fixed (kept from the original sequence).
    Each design region specifies a set of positions and optionally allowed or excluded amino acids.

    Args:
        design_regions (dict): Named regions, e.g.::

            {
                "loop1": {"positions": "63-96", "excluded_aas": "C"},
                "active_site": {"positions": "12,14,17", "allowed_aas": "DEHKNQRST"},
                "helix_cap": {"positions": "30-35,40-45"}
            }

        seq_len (int): Total length of the protein sequence.

    Returns:
        tuple: (fixed_position_mask, aa_allowed_mask)
            - fixed_position_mask: np.ndarray of shape (seq_len,). 1 = fixed, 0 = design.
            - aa_allowed_mask: np.ndarray of shape (seq_len, 21). For design positions,
              1 = amino acid allowed, 0 = disallowed. For fixed positions, all zeros.
    """
    if not isinstance(design_regions, dict):
        raise TypeError(
            f"design_regions must be a dict, got {type(design_regions).__name__}. "
            f"Expected format: {{\"region_name\": {{\"positions\": \"1-10\", ...}}, ...}}"
        )

    # Parse all regions and collect their positions
    region_positions = {}  # region_name -> set of 0-indexed positions
    region_aa_specs = {}   # region_name -> (mode, aa_set) or None

    for name, spec in design_regions.items():
        if 'positions' not in spec:
            raise ValueError(
                f"Design region '{name}' is missing required key 'positions'. "
                f"Each region must specify positions, e.g. {{\"positions\": \"1-10,15,20-25\"}}"
            )

        parsed = parse_position_string(spec['positions'])
        # Validate positions are within sequence bounds
        out_of_range = [p for p in parsed if p >= seq_len]
        if out_of_range:
            raise ValueError(
                f"Design region '{name}' contains positions exceeding sequence length ({seq_len}): "
                f"{[p + 1 for p in out_of_range]} (1-indexed). "
                f"Valid range is 1-{seq_len}."
            )
        region_positions[name] = set(parsed)

        # Parse amino acid constraints
        has_allowed = 'allowed_aas' in spec
        has_excluded = 'excluded_aas' in spec
        if has_allowed and has_excluded:
            raise ValueError(
                f"Design region '{name}' specifies both 'allowed_aas' and 'excluded_aas'. "
                f"Only one may be provided per region."
            )

        if has_allowed:
            aa_str = spec['allowed_aas'].upper()
            invalid_aas = set(aa_str) - AMINO_ACID_SET
            if invalid_aas:
                raise ValueError(
                    f"Design region '{name}' has invalid amino acids in 'allowed_aas': "
                    f"{sorted(invalid_aas)}. Valid amino acids: {AMINO_ACID_ORDER[:-1]}"
                )
            region_aa_specs[name] = ('allowed', set(aa_str))
        elif has_excluded:
            aa_str = spec['excluded_aas'].upper()
            invalid_aas = set(aa_str) - AMINO_ACID_SET
            if invalid_aas:
                raise ValueError(
                    f"Design region '{name}' has invalid amino acids in 'excluded_aas': "
                    f"{sorted(invalid_aas)}. Valid amino acids: {AMINO_ACID_ORDER[:-1]}"
                )
            region_aa_specs[name] = ('excluded', set(aa_str))
        else:
            region_aa_specs[name] = None  # All 20 standard AAs allowed

    # Check for cross-region overlaps
    all_region_names = list(region_positions.keys())
    for i in range(len(all_region_names)):
        for j in range(i + 1, len(all_region_names)):
            name_i, name_j = all_region_names[i], all_region_names[j]
            overlap = region_positions[name_i] & region_positions[name_j]
            if overlap:
                overlap_1indexed = sorted([p + 1 for p in overlap])
                raise ValueError(
                    f"Position(s) {overlap_1indexed} found in multiple design regions: "
                    f"'{name_i}' and '{name_j}'. Each position may only belong to one design region."
                )

    # Build masks
    # Default: all positions fixed
    fixed_position_mask = np.ones(seq_len)
    # aa_allowed_mask: (seq_len, 21) — all zeros by default (fixed positions are irrelevant)
    aa_allowed_mask = np.zeros((seq_len, len(AMINO_ACID_ORDER)))

    for name, pos_set in region_positions.items():
        aa_spec = region_aa_specs[name]
        for pos in pos_set:
            fixed_position_mask[pos] = 0.  # Mark as design position

            if aa_spec is None:
                # All 20 standard AAs allowed (not X)
                aa_allowed_mask[pos, :20] = 1.
            elif aa_spec[0] == 'allowed':
                for aa in aa_spec[1]:
                    aa_allowed_mask[pos, AMINO_ACID_ORDER.index(aa)] = 1.
            elif aa_spec[0] == 'excluded':
                aa_allowed_mask[pos, :20] = 1.  # Start with all allowed
                for aa in aa_spec[1]:
                    aa_allowed_mask[pos, AMINO_ACID_ORDER.index(aa)] = 0.

    return fixed_position_mask, aa_allowed_mask


####################################################################################
# JSON Config Loading
####################################################################################

def load_config_and_merge(args, parser):
    """Load a JSON config file and merge its values into the argparse namespace.

    JSON values override CLI-provided values. When a CLI-provided value (not the parser
    default) is overridden by the JSON config, a warning is printed.

    The ``design_regions`` value in the JSON is expected to be a dict of named regions
    (not a JSON string).

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        parser (argparse.ArgumentParser): The parser used, for determining defaults.

    Returns:
        argparse.Namespace: The merged args namespace (modified in-place).
    """
    config_path = getattr(args, 'config', None)
    if config_path is None:
        return args

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Determine which args the user explicitly set on the CLI
    defaults = vars(parser.parse_args([]))

    for key, json_value in config.items():
        if not hasattr(args, key):
            warnings.warn(f"Config file key '{key}' is not a recognized argument — ignoring.")
            continue

        cli_value = getattr(args, key)
        default_value = defaults.get(key)

        # If the user explicitly set this on the CLI (value differs from default),
        # warn that we are overriding it
        if cli_value != default_value:
            warnings.warn(
                f"CLI argument '--{key}' (value: {cli_value!r}) is overridden by "
                f"config file value: {json_value!r}"
            )

        setattr(args, key, json_value)

    return args


def parse_design_regions_arg(args):
    """Parse the design_regions attribute from a JSON string to a dict if needed.

    If args.design_regions is already a dict (e.g. loaded from a JSON config file),
    it is left as-is. If it is a string, it is parsed as JSON.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        argparse.Namespace: args with design_regions as a dict.
    """
    dr = getattr(args, 'design_regions', None)
    if dr is None:
        args.design_regions = {}
        return args
    if isinstance(dr, str):
        try:
            args.design_regions = json.loads(dr)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse --design_regions as JSON: {e}\n"
                f"Expected format: '{{\"region_name\": {{\"positions\": \"1-10\", ...}}, ...}}'"
            ) from e
    if not isinstance(args.design_regions, dict):
        raise TypeError(
            f"design_regions must be a JSON object/dict, got {type(args.design_regions).__name__}"
        )
    return args


####################################################################################
# CB Coordinates
####################################################################################

def get_cb_coordinates(pdb_code=None, pdb_path=None, structures_dir='/data/structures'):
    """Gets the CB coordinates for each residue. For glycine, provides CA coordinates
    in place of CB coordinates.

    Args:
        pdb_code (str, optional): PDB ID. Used to build file path from structures_dir.
        pdb_path (str, optional): Direct path to a PDB file. Takes priority over pdb_code.
        structures_dir (str): Directory containing PDB files (used with pdb_code).

    Returns:
        ((L x 3) torch.Tensor): CB (or CA for Gly) coordinates per residue.
    """
    if pdb_path is not None:
        resolved_path = pdb_path
    elif pdb_code is not None:
        resolved_path = os.path.join(structures_dir, pdb_code + '.pdb')
    else:
        raise ValueError("Either pdb_code or pdb_path must be provided to get_cb_coordinates.")
    residues = list(PDBParser(PERMISSIVE=True, QUIET=True).get_structure(id=os.path.basename(resolved_path), file=resolved_path)[0].get_residues())

    L = len(residues)
    cb_coordinates = np.zeros((L, 3), dtype=np.float32)

    # Set the coordinates for every residue
    for i, residue in enumerate(residues):
        try:
            if residue.resname == 'GLY':
                cb_coordinates[i, :] = residue["CA"].get_coord()
            else:
                cb_coordinates[i, :] = residue["CB"].get_coord()
        except KeyError as e:
            cb_coordinates[i, :] = residue['Cb'.lower()].get_coord()

    return torch.tensor(cb_coordinates)

def compute_distance_matrix(coordinates, epsilon=0.):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 3) torch.Tensor): an array of all four major atom coordinates
            per residue
        epsilon (float): a term to stabilize the gradients (because backpropping through sqrt
            gives you NaN at 0)
    Returns:
        ((L x L) torch.Tensor): the distance matrix for the residues
    """
    # In reality, pred_coordinates is an output of the network, but we initialize it here for a minimal working example
    L = len(coordinates)
    gram_matrix = torch.mm(coordinates, torch.transpose(coordinates, 0, 1))
    gram_diag = torch.diagonal(gram_matrix, dim1=0, dim2=1)
    # gram_diag: L
    diag_1 = torch.matmul(gram_diag.unsqueeze(-1), torch.ones(1, L).to(coordinates.device))
    # diag_1: L x L
    diag_2 = torch.transpose(diag_1, dim0=0, dim1=1)
    # diag_2: L x L
    squared_distance_matrix = diag_1 + diag_2 - (2 * gram_matrix )
    distance_matrix = torch.sqrt( squared_distance_matrix + epsilon)
    return distance_matrix

def compute_bins(matrix, bins, include_less_than=False, include_greater_than=False):
    """Bin values based on the bins array. Works for distances and trRosetta features.
    Args:
        matrix ((L x n) torch.Tensor): the matrix to bin
        bins ((n_bins) array-like): the bin endpoints
        include_less_than (bool): whether to include a bin for less than the min value
        include_greater_than (bool): whether to include a bin for greater than the max value
    Returns:
        binned_matrix ((L x n x n_bins) torch.Tensor): the matrix, but binned
    """
    L, n = matrix.shape
    # Number of bins is based on whether we have a bin for less than the lowest and greater than the highest
    n_bins = len(bins) - 1 + include_less_than + include_greater_than
    
    # Populate distogram
    binned_matrix = torch.zeros((L, n, n_bins))

    if include_less_than:
        binned_matrix[:, :, 0] = matrix < bins[0]

    for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
        # Bins are shifted by one if we have a "less than" bin
        binned_matrix[:, :, include_less_than + i] = ( (matrix >= bin_min) * (matrix < bin_max) )
    
    if include_greater_than:
        binned_matrix[:, :, -1] = matrix >= bins[-1]

    return binned_matrix

def compute_distogram(coordinates):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 4 x 3) np.ndarray): an array of all four major atom coordinates
            per residue
    Returns:
        ((N x L x L) torch.Tensor): the binned distance matrix for the atoms
    """
    distance_matrix = compute_distance_matrix(coordinates)
    # Make sure all distance values are positive
    assert torch.all(distance_matrix >= 0)
    # The endpoints of the bins (n_bins + 1 endpoints)
    tr_rosetta_bins = np.arange(2.5, 20.5, .5)
    # Compute the distogram
    distogram = compute_bins(matrix=distance_matrix, bins=tr_rosetta_bins, include_less_than=True, include_greater_than=True)
    # No need to normalize probabilities to sum to 1, because there is just one one in each distogram
    
    return distogram