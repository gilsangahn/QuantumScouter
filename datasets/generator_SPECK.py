#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECK32/64 Data Generator for Quantum Cryptanalysis.

This script implements the SPECK32/64 block cipher and generates datasets
for machine learning-based distinguishers. It follows Gohr's standardized
protocol:
1. Real Pairs: (P0, P1) where P1 = P0 ^ InputDifference
2. Random Pairs: (P0, P1) where P1 is independently generated random data.

Usage:
    python generator_speck.py
"""

import numpy as np
import os
from os import urandom

# -------------------------------------------------------------------------
# 1. SPECK32/64 Implementation
# -------------------------------------------------------------------------

# SPECK32/64 Constants
WORD_SIZE = 16
ALPHA = 7
BETA = 2
MASK_VAL = 2**WORD_SIZE - 1

def rol(x, k):
    """Circular bitwise left rotate."""
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))

def ror(x, k):
    """Circular bitwise right rotate."""
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK_VAL)

def enc_one_round(p, k):
    """
    Execute one round of SPECK encryption.
    Args:
        p (tuple): (left_word, right_word)
        k (int): Round key
    Returns:
        tuple: (new_left, new_right)
    """
    c0, c1 = p
    c0 = ror(c0, ALPHA)
    c0 = (c0 + c1) & MASK_VAL
    c0 ^= k
    c1 = rol(c1, BETA)
    c1 ^= c0
    return c0, c1

def expand_key(k, t):
    """
    SPECK Key Schedule.
    Expands the master key into 't' round keys.
    """
    ks = [0 for _ in range(t)]
    ks[0] = k[-1]
    l = list(reversed(k[:-1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
    return ks

def encrypt(p, ks):
    """
    Full SPECK encryption.
    Args:
        p (tuple): Plaintext (left, right)
        ks (list): Expanded round keys
    """
    x, y = p
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return x, y

def convert_to_binary(arr):
    """
    Converts integer arrays to binary representation for ML/Quantum input.
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

def generate_speck_data(n, nr, diff=(0x0040, 0)):
    """
    Generate dataset using the 'Real vs Independent Random' protocol.
    
    Args:
        n (int): Number of samples (must be even).
        nr (int): Number of rounds.
        diff (tuple): Input difference (left_word, right_word) for Real pairs.
    
    Returns:
        X (np.array): Binary features (Ciphertext pairs).
        Y (np.array): Labels (1 for Real, 0 for Random).
    """
    assert n % 2 == 0, "Number of samples (n) must be even."
    
    # 1. Generate Labels (Balanced: 50% Real, 50% Random)
    Y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.uint8)
    np.random.shuffle(Y)

    # 2. Generate Master Keys and Expand
    # Using urandom for cryptographic security
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, nr)

    # 3. Generate First Plaintext (P0) - Common
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    
    # 4. Generate Second Plaintext (P1)
    
    # Case A: Real Pair (P1 = P0 ^ diff)
    plain1l_real = plain0l ^ diff[0]
    plain1r_real = plain0r ^ diff[1]
    
    # Case B: Random Pair (P1 = Independent Random)
    # Crucial for avoiding distinguishing artifacts
    plain1l_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1r_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # 5. Perform Encryption
    # Encrypt P0 -> C0
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    
    # Encrypt Real P1 -> Real C1
    c1l_real, c1r_real = encrypt((plain1l_real, plain1r_real), ks)
    
    # Encrypt Random P1 -> Random C1
    c1l_rand, c1r_rand = encrypt((plain1l_rand, plain1r_rand), ks)

    # 6. Construct Final Dataset based on Labels (Y)
    ctdata1l = np.zeros_like(plain0l)
    ctdata1r = np.zeros_like(plain0r)
    
    # Where Y=1, inject Real C1
    mask_real = (Y == 1)
    ctdata1l[mask_real] = c1l_real[mask_real]
    ctdata1r[mask_real] = c1r_real[mask_real]
    
    # Where Y=0, inject Random C1
    mask_rand = (Y == 0)
    ctdata1l[mask_rand] = c1l_rand[mask_rand]
    ctdata1r[mask_rand] = c1r_rand[mask_rand]

    # 7. Preprocess to Binary (64 bits total)
    # Structure: C0_L || C0_R || C1_L || C1_R
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    
    return X, Y

def save_dataset(X, Y, path="speck_dataset"):
    """Helper to save dataset as .npy files."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "data.npy"), X)
    np.save(os.path.join(path, "labels.npy"), Y)
    print(f"[+] Dataset saved to {path}/")
    print(f"    - Data Shape: {X.shape}")
    print(f"    - Label Distribution: {np.bincount(Y)}")

# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    N_SAMPLES = 500        # Dataset size (Validation phase)
    NUM_ROUNDS = 5         # Number of rounds to attack
    DIFFERENCE = (0x0040, 0x0000) # Optimal input difference for 5-round SPECK
    SAVE_PATH = "speck_dataset"

    print(f"[*] Generating Data: SPECK32/64, {NUM_ROUNDS} Rounds")
    print(f"    - Samples: {N_SAMPLES}")
    print(f"    - Input Diff: {DIFFERENCE}")
    
    X, Y = generate_speck_data(N_SAMPLES, NUM_ROUNDS, diff=DIFFERENCE)
    save_dataset(X, Y, path=SAVE_PATH)
    print("[*] Process Complete.")
