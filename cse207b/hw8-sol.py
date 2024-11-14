#!/usr/bin/env sage

from sage.all import *
from Crypto.PublicKey import RSA
import base64
import struct
import re
from Crypto.Cipher import AES
import itertools

def small_roots(f, bounds, m=1, d=None): # This is from Github
    if not d:
        d = f.degree()

    if isinstance(f, Polynomial):
        x, = polygens(f.base_ring(), f.variable_name(), 1)
        f = f(x)

    R = f.base_ring()
    N = R.cardinality()
    
    f /= f.coefficients().pop(0)
    f = f.change_ring(ZZ)

    G = Sequence([], f.parent())
    for i in range(m+1):
        base = N^(m-i) * f^i
        for shifts in itertools.product(range(d), repeat=f.nvariables()):
            g = base * prod(map(power, f.variables(), shifts))
            G.append(g)

    B, monomials = G.coefficient_matrix()
    monomials = vector(monomials)

    factors = [monomial(*bounds) for monomial in monomials]
    for i, factor in enumerate(factors):
        B.rescale_col(i, factor)

    B = B.dense_matrix().LLL()

    B = B.change_ring(QQ)
    for i, factor in enumerate(factors):
        B.rescale_col(i, 1/factor)

    H = Sequence([], f.parent().change_ring(QQ))
    for h in filter(None, B*monomials):
        H.append(h)
        I = H.ideal()
        if I.dimension() == -1:
            H.pop()
        elif I.dimension() == 0:
            roots = []
            for root in I.variety(ring=ZZ):
                root = tuple(R(root[var]) for var in f.variables())
                roots.append(root)
            return roots

    return []


# Read the public key
with open('key.pub', 'r') as f:
    key = RSA.import_key(f.read())

N = Integer(key.n)
e = Integer(key.e)

print("N.nbits() =", N.nbits())
print("e =", e)

# Read the encrypted data
with open('hw8.pdf.enc.asc', 'r') as f:
    encrypted_data = f.read()

# Parse the base64 content between the header and footer
pattern = r'-----BEGIN PRETTY BAD ENCRYPTED MESSAGE-----\n(.*?)\n-----END PRETTY BAD ENCRYPTED MESSAGE-----'
match = re.search(pattern, encrypted_data, re.DOTALL)
if not match:
    print("Encrypted message not found")
    exit(1)

data_b64 = match.group(1)
data_bytes = base64.b64decode(data_b64)

# Parse the data structure
offset = 0
rsa_ciphertext_length = struct.unpack('<I', data_bytes[0:4])[0]
offset += 4
rsa_ciphertext = data_bytes[offset:offset + rsa_ciphertext_length]
offset += rsa_ciphertext_length
iv = data_bytes[offset:offset + 16]
offset += 16
aes_encrypted = data_bytes[offset:]

c = Integer(int.from_bytes(rsa_ciphertext, 'big'))

# Reconstruct P
keysize_bits = N.nbits()
keysize_bytes = (keysize_bits + 7) // 8
k = keysize_bytes - 3 - 32  # As per the original encryption code

print("Key size (bytes):", keysize_bytes)
print("k (number of 0xFF bytes):", k)

EB_prefix = '0001' + 'ff' * k + '00'
P_int = Integer(EB_prefix, 16)
P = P_int

shift = 256  # Since s is 256 bits

# Define the polynomial f(s)
R = Integers(N)
PR.<s> = PolynomialRing(R)
f = (P * 2^shift + s)^e - c

print("Constructed polynomial f(s) mod N")

# Set the bounds for s
bounds = [2^256]

# Try different m and d values
for m in range(1, 6):
    for d in range(1, 4):
        print("---------------------------------------------")
        print("Trying m =", m, "d =", d)
        roots = small_roots(f, bounds, m, d)
        print(roots)
        if roots:
            print(f"Found roots with m={m}, d={d}")
            for root_tuple in roots:
                root = root_tuple[0]  # Since it's univariate
                s_int = Integer(root)
                if s_int == 0:
                    print("Invalid Root = 0")
                    continue
                else:
                    m_candidate = P * 2^shift + s_int
                    print("Found potential s =", s_int)
                    # Reconstruct AES key
                    aeskey = int(s_int).to_bytes(32, byteorder='big')

                    # Decrypt the data
                    cipher = AES.new(aeskey, AES.MODE_CBC, iv)
                    padded_plaintext = cipher.decrypt(aes_encrypted)

                    # Unpad (PKCS7)
                    pad_length = padded_plaintext[-1]
                    if 1 <= pad_length <= 16 and all(p == pad_length for p in padded_plaintext[-pad_length:]):
                        plaintext = padded_plaintext[:-pad_length]

                        # Save plaintext
                        with open(f'hw8_{m}_{d}.pdf', 'wb') as f_out:
                            f_out.write(plaintext)
                        print("Decryption complete")
                        break