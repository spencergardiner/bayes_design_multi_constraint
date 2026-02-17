#!/usr/bin/env python3
"""Quick smoke tests for the new BayesDesign utilities."""
import sys
sys.path.insert(0, '.')

from design import parser
from bayes_design.utils import (
    load_config_and_merge, parse_design_regions_arg,
    build_aa_allowed_mask, parse_position_string, resolve_protein_input,
)
import numpy as np
import argparse
import tempfile
import json

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        failed += 1

# Test 1
def t1():
    r = parse_position_string('3,5-7,2,17,88-91')
    assert r == [1, 2, 4, 5, 6, 16, 87, 88, 89, 90], f"got {r}"
test("parse_position_string basic", t1)

# Test 2
def t2():
    r = parse_position_string('5 - 7, 3')
    assert r == [2, 4, 5, 6], f"got {r}"
test("parse_position_string spaces", t2)

# Test 3
def t3():
    try:
        parse_position_string('5;7')
        assert False, "should raise"
    except ValueError as e:
        assert ';' in str(e)
test("invalid character ValueError", t3)

# Test 4
def t4():
    try:
        parse_position_string('7-5')
        assert False, "should raise"
    except ValueError:
        pass
test("reversed range ValueError", t4)

# Test 5
def t5():
    regions = {
        'loop1': {'positions': '3-5', 'excluded_aas': 'C'},
        'site': {'positions': '8,10', 'allowed_aas': 'DE'}
    }
    fpm, aam = build_aa_allowed_mask(regions, 12)
    assert fpm[2] == 0 and fpm[3] == 0 and fpm[4] == 0
    assert fpm[7] == 0 and fpm[9] == 0
    assert fpm[0] == 1
    assert aam[2, 1] == 0 and aam[2, 0] == 1  # C excluded (idx 1), A allowed (idx 0)
    assert aam[7, 2] == 1 and aam[7, 3] == 1 and aam[7, 0] == 0  # D(idx2),E(idx3) allowed; A not
test("build_aa_allowed_mask multi-region", t5)

# Test 6
def t6():
    try:
        build_aa_allowed_mask({'r1': {'positions': '3-5'}, 'r2': {'positions': '4-6'}}, 10)
        assert False, "should raise"
    except ValueError as e:
        assert 'multiple design regions' in str(e)
test("overlapping regions ValueError", t6)

# Test 7
def t7():
    a = parser.parse_args(['--protein_id', '6MRR', '--design_regions', '{"loop": {"positions": "1-5"}}'])
    a = parse_design_regions_arg(a)
    assert isinstance(a.design_regions, dict) and 'loop' in a.design_regions
test("design_regions CLI JSON parsing", t7)

# Test 8
def t8():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'protein_id': 'ABCD', 'design_regions': {'r1': {'positions': '1-10'}}}, f)
        cp = f.name
    a = parser.parse_args(['--config', cp])
    a = load_config_and_merge(a, parser)
    a = parse_design_regions_arg(a)
    assert a.protein_id == 'ABCD'
    assert isinstance(a.design_regions, dict)
test("JSON config loading", t8)

# Test 9
def t9():
    ns = argparse.Namespace(protein_id=None, pdb_path=None)
    try:
        resolve_protein_input(ns)
        assert False, "should raise"
    except ValueError as e:
        assert '--protein_id' in str(e) and '--pdb_path' in str(e)
test("missing protein input ValueError", t9)

# Test 10: empty design_regions = all positions designable
def t10():
    fpm, aam = build_aa_allowed_mask({}, 5)
    # All positions should be fixed when no regions specified
    assert all(fpm[i] == 1 for i in range(5))
test("empty design_regions = all fixed", t10)

print(f"\n{passed}/{passed+failed} tests passed")
if failed:
    sys.exit(1)
