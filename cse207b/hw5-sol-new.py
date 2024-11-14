#!/usr/bin/env sage

from sage.all import *
import struct
import re
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import hashlib
key_header = '-----BEGIN PRETTY BAD PUBLIC KEY BLOCK-----\n'
key_footer = '-----END PRETTY BAD PUBLIC KEY BLOCK-----\n'
encrypt_header = '-----BEGIN PRETTY BAD ENCRYPTED MESSAGE-----\n'
encrypt_footer = '-----END PRETTY BAD ENCRYPTED MESSAGE-----\n'

def b64_dec(s):
    return base64.b64decode(s)

def parse_mpi(s, index):
    length = struct.unpack('<I', s[index:index+4])[0]
    xbytes = s[index+4:index+4+length]
    z = int.from_bytes(xbytes, 'big')
    return z, index+4+length

def parse_public_key(s):
    data = re.search(key_header + "(.*)" + key_footer, s, flags=re.DOTALL).group(1)
    data = b64_dec(data)
    index = 0
    p, index = parse_mpi(data, index)
    g, index = parse_mpi(data, index)
    y, index = parse_mpi(data, index)
    return {'p': p, 'g': g, 'y': y}

def parse_encrypted_message(s):
    data = re.search(encrypt_header + "(.*)" + encrypt_footer, s, flags=re.DOTALL).group(1)
    data = b64_dec(data)
    index = 0
    c1, index = parse_mpi(data, index)
    c2, index = parse_mpi(data, index)
    iv = data[index:index+16]
    aes_ciphertext = data[index+16:]
    return c1, c2, iv, aes_ciphertext

def brute_force_dlog(g, p, h, order):
    for x in range(order):
        if pow(g, x, p) == h:
            return x
    return None

def baby_step_giant_step(g, p, h, order):
    m = ceil(sqrt(order))
    baby_steps = {}
    for j in range(m):
        baby_steps[pow(g, j, p)] = j
    c = pow(g, -m, p)
    gamma = h
    for i in range(m):
        if gamma in baby_steps:
            return i * m + baby_steps[gamma]
        gamma = (gamma * c) % p
    return None

def crt(remainders, moduli):
    x = 0
    M = 1
    for m_i in moduli:
        M *= m_i
    for r_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        inv = pow(M_i, -1, m_i)
        x += r_i * M_i * inv
    return x % M

def pohlig_hellman(g, h, p, factors):
    remainders = []
    moduli = []
    for q, e in factors:
        m = q ** e
        g0 = pow(g, (p - 1) // m, p)
        h0 = pow(h, (p - 1) // m, p)

        # solve for x modulo m
        x_qe = 0
        for k in range(e):
            # g0^{q^{k}}, h0^{q^{k}}
            gk = pow(g0, pow(q, k), p)
            hk = pow(h0 * pow(g0, -x_qe, p), pow(q, e - k - 1), p)

            if q < 1e6:
                dk = brute_force_dlog(gk, p, hk, q)
            else:
                dk = baby_step_giant_step(gk, p, hk, q)

            x_qe += dk * (q ** k)
        remainders.append(x_qe)
        moduli.append(m)
    x = crt(remainders, moduli)
    return x

def int_to_binary(z):
    z = int(z)
    return z.to_bytes((z.bit_length() + 7) // 8, 'big')

def main():
    # load public key
    pubkey_text = open('key.pub').read()
    print(pubkey_text)
    pubkey = parse_public_key(pubkey_text)
    p = pubkey['p']
    g = pubkey['g']
    y = pubkey['y']

    # Factor p - 1
    p_minus_1 = p - 1
    factors = factor(p_minus_1)
    fact_list = [(int(f[0]), int(f[1])) for f in factors]
    fact_list.sort()

    # Select factors to cover 2^128
    selected_factors = []
    product = 1
    for q, e in fact_list:
        selected_factors.append((q, e))
        product *= q ** e
        if product > 2 ** 128:
            break

    # Compute x using Pohlig-Hellman
    x = pohlig_hellman(g, y, p, selected_factors)
    print(f"x = {x}")

    # Load encrypted message
    encrypted_message = open('hw5.pdf.enc.asc').read()
    c1, c2, iv, aes_ciphertext = parse_encrypted_message(encrypted_message)

    # Convert variables to int if they are Sage Integers
    c1 = int(c1)
    c2 = int(c2)
    p = int(p)
    x = int(x)

    # Recover m (the AES key)
    k_inv = pow(c1, -x, p)
    m = (c2 * k_inv) % p
    # Ensure m is an int
    m = int(m)

    # Convert m to bytes using int_to_binary
    aes_key_bytes = int_to_binary(m)

    aes_key = aes_key_bytes[:16]

    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    plaintext_padded = cipher.decrypt(aes_ciphertext)
    plaintext = unpad(plaintext_padded, AES.block_size)
    print(f"Decryption successful")
    with open('hw5.pdf', 'wb') as f:
        f.write(plaintext)


if __name__ == '__main__':
    main()
