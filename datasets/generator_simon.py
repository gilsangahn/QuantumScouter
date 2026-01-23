# -*- coding: utf-8 -*-
import numpy as np
import os
from os import urandom

# --- 1. SIMON32/64 Implementation (Vectorized) ---

WORD_SIZE = 16
MASK_VAL = 2**WORD_SIZE - 1

# SIMON Constants (z0)
z0 = [1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0]

def rol(x, k):
    """Rotate Left (Supports NumPy Arrays)"""
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))

def f(x):
    """SIMON Round Function"""
    return (rol(x, 1) & rol(x, 8)) ^ rol(x, 2)

def enc_one_round(x, y, k):
    """Encrypt one round"""
    new_x = y ^ f(x) ^ k
    new_y = x
    return new_x, new_y

def expand_key(k, t):
    """
    SIMON32/64 Key Schedule (Vectorized)
    k: shape (4, n_samples)
    """
    ks = [None] * t
    # k is passed as [k3, k2, k1, k0] where k0 is LSB. 
    # Logic follows standard implementation handling arrays.
    
    # First m round keys are the key words themselves
    m = 4
    for i in range(m):
        ks[i] = k[i]
        
    # Generate remaining round keys
    for i in range(m, t):
        tmp = rol(ks[i-1], 3)
        if m == 4:
            tmp ^= ks[i-3]
        tmp ^= rol(tmp, 1)
        
        c = 0xfffc ^ z0[(i-m) % 62]
        ks[i] = (~ks[i-m] & MASK_VAL) ^ tmp ^ c
        
    return ks

def encrypt(p, ks):
    """Encrypt a plaintext block with expanded keys"""
    x, y = p 
    for k in ks:
        x, y = enc_one_round(x, y, k)
    return x, y

def convert_to_binary(arr):
    """Converts integer arrays to binary representation for ML input"""
    # arr shape: (4, n_samples) -> Output shape: (64, n_samples)
    X = np.zeros((4 * WORD_SIZE, len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        index = i // WORD_SIZE
        offset = WORD_SIZE - (i % WORD_SIZE) - 1
        X[i] = (arr[index] >> offset) & 1
    return X.T

# --- 2. Dataset Generation (Same Structure as SPECK) ---

def generate_simon_data_vectorized(n, nr, diff=(0x0000, 0x0040)):
    """
    Generate dataset for SIMON32/64 using Vectorized approach (Same as SPECK).
    """
    assert n % 2 == 0, "n must be even"
    
    # 1. Generate Labels (50% Real, 50% Random)
    Y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.uint8)
    np.random.shuffle(Y)

    # 2. Generate Keys and Expand (Vectorized)
    # keys shape: (4, n)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, nr)

    # 3. Generate First Plaintext (P0) - Common
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    
    # 4. Generate Second Plaintext (P1) - Based on Logic
    
    # (A) Real Pair Candidates: P1 = P0 ^ diff
    plain1l_real = plain0l ^ diff[0]
    plain1r_real = plain0r ^ diff[1]
    
    # (B) Random Pair Candidates: P1 = Independent Random
    plain1l_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1r_rand = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # 5. Encryption (Encrypt Everything)
    # Encrypt P0 -> C0
    c0l, c0r = encrypt((plain0l, plain0r), ks)
    
    # Encrypt Real P1 -> Real C1
    c1l_real, c1r_real = encrypt((plain1l_real, plain1r_real), ks)
    
    # Encrypt Random P1 -> Random C1
    c1l_rand, c1r_rand = encrypt((plain1l_rand, plain1r_rand), ks)

    # 6. Select C1 based on Label (Masking - Same as SPECK code)
    ctdata1l = np.zeros_like(plain0l)
    ctdata1r = np.zeros_like(plain0r)
    
    # If Y == 1 (Real), pick from Real C1
    mask_real = (Y == 1)
    ctdata1l[mask_real] = c1l_real[mask_real]
    ctdata1r[mask_real] = c1r_real[mask_real]
    
    # If Y == 0 (Random), pick from Random C1
    mask_rand = (Y == 0)
    ctdata1l[mask_rand] = c1l_rand[mask_rand]
    ctdata1r[mask_rand] = c1r_rand[mask_rand]

    # 7. Convert to Binary
    # Structure: C0_L || C0_R || C1_L || C1_R
    X = convert_to_binary([c0l, c0r, ctdata1l, ctdata1r])
    
    return X, Y

def save_dataset(X, Y, path="dataset"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "data_simon.npy"), X)
    np.save(os.path.join(path, "labels_simon.npy"), Y)
    print(f"Saved dataset to {path}:")
    print(f"  - Data shape: {X.shape}")
    print(f"  - Label counts: {np.bincount(Y)}")

if __name__ == "__main__":
    N_SAMPLES = 500       # Sample size for validation
    NUM_ROUNDS = 4        # 4 Rounds
    DIFFERENCE = (0x0000, 0x0040) # Input difference for Feistel property
    SAVE_PATH = "dataset"

    print(f"Generating data for {NUM_ROUNDS}-round SIMON32/64 (Vectorized)...")
    print(f" - Input Difference: {DIFFERENCE}")
    
    X, Y = generate_simon_data_vectorized(N_SAMPLES, NUM_ROUNDS, diff=DIFFERENCE)
    save_dataset(X, Y, path=SAVE_PATH)
    print("Done!")