# -*- coding: utf-8 -*-
import numpy as np
import os
from os import urandom

# --- 1. SPECK32/64 Implementation ---

# Parameters for SPECK32/64
WORD_SIZE = 16
ALPHA = 7
BETA = 2
MASK_VAL = 2**WORD_SIZE - 1

def rol(x, k):
    """Rotate Left"""
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))

def ror(x, k):
    """Rotate Right"""
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK_VAL)

def enc_one_round(p, k):
    """Encrypt one round"""
    c0, c1 = p
    c0 = ror(c0, ALPHA)
    c0 = (c0 + c1) & MASK_VAL
    c0 ^= k
    c1 = rol(c1, BETA)
    c1 ^= c0
    return c0, c1

def expand_key(k, t):
    """Key Schedule: Expands the master key into round keys"""
    ks = [0 for _ in range(t)]
    ks[0] = k[-1]
    l = list(reversed(k[:-1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
    return ks

def encrypt(p, ks):
    """Encrypt a plaintext block with the expanded round keys"""
    x, y = p
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return x, y

def convert_to_binary(arr):
    """Converts integer arrays to binary representation for ML input"""
    X = np.zeros((4 * WORD_SIZE, len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        index = i // WORD_SIZE
        offset = WORD_SIZE - (i % WORD_SIZE) - 1
        X[i] = (arr[index] >> offset) & 1
    return X.T

# --- 2. Dataset Generation (Correction Applied) ---

def real_differences_data(n, nr, diff=(0x0040, 0)):
    """
    Generate dataset for differential cryptanalysis on SPECK32/64.
    
    Args:
        n (int): Number of samples (must be even).
        nr (int): Number of rounds.
        diff (tuple): Input difference (left_word, right_word).
    
    Returns:
        X (np.array): Binary input data (Ciphertext pairs).
        Y (np.array): Labels (1 for Real Pair, 0 for Random Pair).
    """
    assert n % 2 == 0, "n must be even"
    
    # 1. Generate Labels (50% Real, 50% Random)
    Y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.uint8)
    np.random.shuffle(Y)

    # 2. Generate Keys and Expand
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, nr)

    # 3. Generate First Plaintext (P0) - Common for all samples
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    
    # 4. Generate Second Plaintext (P1) - Based on Label Logic
    
    # (A) For Real Pair: P1 has input difference (P0 ^ diff)
    plain1l_real = plain0l ^ diff[0]
    plain1r_real = plain0r ^ diff[1]
    
    # (B) For Random Pair: P1 is completely random and independent
    plain1l_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1r_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # 5. Encryption
    # Encrypt P0 -> C0
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    
    # Encrypt Real P1 -> Real C1
    c1l_real, c1r_real = encrypt((plain1l_real, plain1r_real), ks)
    
    # Encrypt Random P1 -> Random C1
    c1l_rand, c1r_rand = encrypt((plain1l_rand, plain1r_rand), ks)

    # 6. Select C1 based on Label (Masking)
    # Initialize C1 arrays
    ctdata1l = np.zeros_like(plain0l)
    ctdata1r = np.zeros_like(plain0r)
    
    # If Y == 1 (Real), use Real C1
    mask_real = (Y == 1)
    ctdata1l[mask_real] = c1l_real[mask_real]
    ctdata1r[mask_real] = c1r_real[mask_real]
    
    # If Y == 0 (Random), use Random C1
    mask_rand = (Y == 0)
    ctdata1l[mask_rand] = c1l_rand[mask_rand]
    ctdata1r[mask_rand] = c1r_rand[mask_rand]

    # 7. Convert to Binary (64-bit format: C0_L || C0_R || C1_L || C1_R)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    
    return X, Y

def save_dataset(X, Y, path="dataset"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "data_speck.npy"), X)
    np.save(os.path.join(path, "labels_speck.npy"), Y)
    print(f"Saved dataset to {path}:")
    print(f"  - Data shape: {X.shape}")
    print(f"  - Label counts: {np.bincount(Y)}")

# --- 3. Main Execution ---

if __name__ == "__main__":
    # Experiment Settings
    N_SAMPLES = 500        # Number of samples
    NUM_ROUNDS = 5           # Target rounds
    DIFFERENCE = (0x0040, 0x0000) # Input difference
    SAVE_PATH = "dataset"

    print(f"Generating data for {NUM_ROUNDS}-round SPECK32/64...")
    print(f" - Input Difference: {DIFFERENCE}")
    print(f" - Samples: {N_SAMPLES}")
    
    X, Y = real_differences_data(N_SAMPLES, NUM_ROUNDS, diff=DIFFERENCE)
    save_dataset(X, Y, path=SAVE_PATH)
    print("Done!")