#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMON32/64 Data Generator for Quantum Cryptanalysis.

This script extends the evaluation framework to the SIMON block cipher (Feistel structure)
to demonstrate the generalization capability of the QuantumScouter.
It follows the same rigorous 'Real vs Independent Random' protocol as the SPECK experiments.

Usage:
    python generator_simon.py
"""

import numpy as np
import os
from os import urandom

# -------------------------------------------------------------------------
# 1. SIMON32/64 Implementation
# -------------------------------------------------------------------------

WORD_SIZE = 16
MASK_VAL = 2**WORD_SIZE - 1

# SIMON Constants (z0 sequence)
z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def rol(x, k):
    """Circular bitwise left rotate."""
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))

def f(x):
    """SIMON Round Function."""
    return (rol(x, 1) & rol(x, 8)) ^ rol(x, 2)

def enc_one_round(x, y, k):
    """
    Execute one round of SIMON encryption.
    Args:
        x, y: Input words (left, right)
        k: Round key
    Returns:
        new_x, new_y: Output words
    """
    new_x = y ^ f(x) ^ k
    new_y = x
    return new_x, new_y

def expand_key(k, t):
    """
    SIMON32/64 Key Schedule.
    Expands the master key into 't' round keys using the z0 sequence.
    """
    ks = [0] * t
    ks[0:4] = k[0:4] # Use first 4 words as initial keys
    
    for i in range(t - 4):
        tmp = ror(ks[i+3], 3)
        if 4 == 4: # m=4 for SIMON32/64
             tmp = tmp ^ ks[i+1]
        tmp = tmp ^ ror(tmp, 1)
        ks[i+4] = (~ks[i]) ^ tmp ^ z0[i % 62] ^ 3
        ks[i+4] &= MASK_VAL
        
    return ks

# Helper needed for Key Schedule (Rotate Right)
def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK_VAL)

def encrypt(p, ks):
    """
    Full SIMON encryption.
    Args:
        p (tuple): Plaintext (left, right)
        ks (list): Expanded round keys
    """
    x, y = p
    for k in ks:
        x, y = enc_one_round(x, y, k)
    return x, y

def convert_to_binary(arr):
    """
    Converts integer arrays to binary representation.
    Input shape: (4, N) -> Output shape: (64, N)
    """
    X = np.zeros((4 * WORD_SIZE, len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        index = i // WORD_SIZE
        offset = WORD_SIZE - (i % WORD_SIZE) - 1
        X[i] = (arr[index] >> offset) & 1
    return X.T

# -------------------------------------------------------------------------
# 2. Dataset Generation (Standard Protocol)
# -------------------------------------------------------------------------

def generate_simon_data(n, nr, diff=(0, 0x0040)):
    """
    Generate SIMON dataset using the 'Real vs Independent Random' protocol.
    
    Args:
        n (int): Number of samples.
        nr (int): Number of rounds.
        diff (tuple): Input difference (left_word, right_word).
                      Default (0, 0x0040) targets the Feistel structure.
    """
    assert n % 2 == 0, "Number of samples (n) must be even."
    
    # 1. Generate Labels
    Y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.uint8)
    np.random.shuffle(Y)

    # 2. Generate Keys and Expand
    # For SIMON32/64, master key is 4 words (64 bits)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    
    # Custom vectorized key schedule handling for simulation
    # (Simplified for batch processing logic)
    ks_batch = []
    for i in range(n):
        k_single = [keys[j][i] for j in range(4)]
        ks_batch.append(expand_key(k_single, nr))
    # Transpose for easier access: ks[round][sample]
    ks = list(map(list, zip(*ks_batch)))

    # 3. Generate Plaintext P0
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    
    # 4. Generate Plaintext P1
    
    # Case A: Real Pair (P1 = P0 ^ diff)
    plain1l_real = plain0l ^ diff[0]
    plain1r_real = plain0r ^ diff[1]
    
    # Case B: Random Pair (Independent Random)
    plain1l_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1r_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # 5. Encryption (Vectorized per round key)
    # Encrypt P0
    c0l, c0r = plain0l, plain0r
    for k in ks:
        c0l, c0r = enc_one_round(c0l, c0r, np.array(k))

    # Encrypt Real P1
    c1l_real, c1r_real = plain1l_real, plain1r_real
    for k in ks:
        c1l_real, c1r_real = enc_one_round(c1l_real, c1r_real, np.array(k))
        
    # Encrypt Random P1
    c1l_rand, c1r_rand = plain1l_rand, plain1r_rand
    for k in ks:
        c1l_rand, c1r_rand = enc_one_round(c1l_rand, c1r_rand, np.array(k))

    # 6. Construct Final Dataset
    ctdata1l = np.zeros_like(plain0l)
    ctdata1r = np.zeros_like(plain0r)
    
    mask_real = (Y == 1)
    ctdata1l[mask_real] = c1l_real[mask_real]
    ctdata1r[mask_real] = c1r_real[mask_real]
    
    mask_rand = (Y == 0)
    ctdata1l[mask_rand] = c1l_rand[mask_rand]
    ctdata1r[mask_rand] = c1r_rand[mask_rand]

    # 7. Convert to Binary
    X = convert_to_binary([c0l, c0r, ctdata1l, ctdata1r])
    
    return X, Y

def save_dataset(X, Y, path="simon_dataset"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "data_simon.npy"), X)
    np.save(os.path.join(path, "labels_simon.npy"), Y)
    print(f"[+] SIMON Dataset saved to {path}/")
    print(f"    - Data Shape: {X.shape}")
    print(f"    - Label Distribution: {np.bincount(Y)}")

# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration for SIMON Experiment
    N_SAMPLES = 500
    NUM_ROUNDS = 4
    DIFFERENCE = (0x0000, 0x0040) # Input difference for SIMON (Feistel structure)
    SAVE_PATH = "simon_dataset"

    print(f"[*] Generating Data: SIMON32/64, {NUM_ROUNDS} Rounds")
    print(f"    - Samples: {N_SAMPLES}")
    print(f"    - Input Diff: {DIFFERENCE}")
    
    X, Y = generate_simon_data(N_SAMPLES, NUM_ROUNDS, diff=DIFFERENCE)
    save_dataset(X, Y, path=SAVE_PATH)
    print("[*] Process Complete.")
