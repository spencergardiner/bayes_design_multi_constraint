from bayes_design.utils import build_aa_allowed_mask, AMINO_ACID_ORDER
import numpy as np

def test_consider_wt_aa():
    design_regions = {
        "region1": {"positions": "1", "allowed_aas": "AC"}
    }
    seq_len = 5
    seq = "DEFGH" 
    
    # Without consider_wt_aa
    fixed, allowed = build_aa_allowed_mask(design_regions, seq_len)
    # allowed[0] should allow A and C. 'D' is not allowed.
    assert allowed[0, AMINO_ACID_ORDER.index('A')] == 1
    assert allowed[0, AMINO_ACID_ORDER.index('C')] == 1
    assert allowed[0, AMINO_ACID_ORDER.index('D')] == 0
    
    # With consider_wt_aa, 'D' should be allowed now
    fixed, allowed = build_aa_allowed_mask(design_regions, seq_len, seq=seq, consider_wt_aa=True)
    assert allowed[0, AMINO_ACID_ORDER.index('A')] == 1
    assert allowed[0, AMINO_ACID_ORDER.index('C')] == 1
    assert allowed[0, AMINO_ACID_ORDER.index('D')] == 1
    
    print("PASS: test_consider_wt_aa (allowed_aas)")

def test_consider_wt_aa_excluded():
    design_regions = {
        "region1": {"positions": "1", "excluded_aas": "D"}
    }
    seq_len = 5
    seq = "DEFGH"
    
    # Without consider_wt_aa
    fixed, allowed = build_aa_allowed_mask(design_regions, seq_len)
    # excluded 'D', so 'D' should be 0.
    assert allowed[0, AMINO_ACID_ORDER.index('D')] == 0
    
    # With consider_wt_aa, 'D' should stay allowed even if excluded?
    # Logic:
    # 1. Start with all 1s (because excluded mode)
    # 2. Set 'D' to 0 because excluded.
    # 3. If consider_wt_aa: set 'D' back to 1.
    
    fixed, allowed = build_aa_allowed_mask(design_regions, seq_len, seq=seq, consider_wt_aa=True)
    assert allowed[0, AMINO_ACID_ORDER.index('D')] == 1
    
    print("PASS: test_consider_wt_aa (excluded_aas)")

if __name__ == "__main__":
    test_consider_wt_aa()
    test_consider_wt_aa_excluded()
