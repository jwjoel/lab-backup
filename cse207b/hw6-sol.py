#!/usr/bin/env sage

from sage.all import *
import sys

def product_tree(ns):
    """
    Builds the product tree from the list of moduli.
    Each level is a list of products of adjacent pairs from the previous level.
    """
    tree = [ns]
    while len(ns) > 1:
        ns = [ns[i]*ns[i+1] for i in range(0, len(ns)-1, 2)] + \
             ([ns[-1]] if len(ns) % 2 == 1 else [])
        tree.append(ns)
    return tree

def remainder_tree(P, tree):
    """
    Computes the remainders P mod Ni^2 for each Ni efficiently using the remainder tree.
    """
    remainders = [P] * len(tree[-1])
    for level in reversed(range(len(tree))):
        ns = tree[level]
        new_remainders = []
        for i in range(len(ns)):
            remainder = remainders[i//2] % (ns[i]**2)
            new_remainders.append(remainder)
        remainders = new_remainders
    return remainders

def main():
    moduli_file = "moduli.sorted"

    # Read moduli from file
    moduli = []
    with open(moduli_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line and all(c in '0123456789abcdefABCDEF' for c in line):
                modulus = Integer('0x' + line)
                moduli.append(modulus)
            else:
                print(f"Invalid modulus at line {idx+1}: {line}")
    n = len(moduli)
    print(f"Loaded {n} moduli.")

    # Build product tree
    print("Building product tree...")
    tree = product_tree(moduli)

    # Compute P = product of all moduli
    P = tree[-1][0]

    print("Computing remainders modulo N_i^2...")
    # Compute remainders P mod N_i^2 for each N_i
    remainders = remainder_tree(P, tree)

    print("Computing GCDs...")
    found_pairs = []
    for i in range(n):
        N_i = moduli[i]
        z_i = remainders[i]
        # Compute z_i / N_i
        try:
            numerator = z_i // N_i
        except ZeroDivisionError:
            print(f"Error dividing by zero at index {i}")
            continue
        gcd_candidate = gcd(N_i, numerator)
        if 1 < gcd_candidate < N_i:
            print(f"Found common factor for modulus index {i}: GCD = {gcd_candidate}")
            # Now, find the other modulus that shares this factor
            for j in range(n):
                if i != j and moduli[j] % gcd_candidate == 0:
                    print(f"Moduli indices with common factor: {i}, {j}")
                    print(f"Moduli with common factor:\n{moduli[i]}\n{moduli[j]}")
                    found_pairs.append((i, j))
                    break  # Stop after finding one pair
            if found_pairs:
                break  # Stop after finding one pair

    if not found_pairs:
        print("No common factors found.")
    else:
        # Output the two moduli separated by a comma
        i, j = found_pairs[0]
        modulus_i_hex = format(moduli[i], 'x')
        modulus_j_hex = format(moduli[j], 'x')
        print(f"{modulus_i_hex},{modulus_j_hex}")

if __name__ == '__main__':
    main()



# #!/usr/bin/env sage

# from sage.all import *
# import sys
# import struct
# import base64
# import re

# from Crypto.PublicKey import RSA
# from Crypto.Cipher import AES, PKCS1_v1_5
# from Crypto import Random

# def main():
#     # Shared prime factor (GCD)
#     p = Integer('11811409921950843614552302871654017773160532181922872177818594357854934353501007911949499569442119830912083732355638978137030693379631614826227288991938499')
    
#     # Modulus N1 (hexadecimal string from your output)
#     N1_hex = '225a0510c14ae9747e20b0439ee8c5da49a17b489f38b67d7c820d5a81f7852ead2e401072d4e4556fb869427f605a75156996b7bc0e6a5d680eb0a3a39e45a6a5e31c05aef52ab8bef474c9b5bd174055de4401f02c71e3d6d90acb2128709dd06ecae02e1dae8e520a9b0b3e6903bca871ff78a2fafcad83de2ab5f6beb41'

#     # Modulus N2 (hexadecimal string from your output)
#     N2_hex = 'aa48c81c309c122f9b55516a45f64b9b22049aba3b64bba21f406704d2389925d870ace814957fc4d5e1942ee21b49ec36a6ee86eefd6270382bc696963ba39805ffb10f6ab468b4986a6ffb11f6eec4a409c243b8ed4784e8d9811b8533f5f0f58cfb5188e584340741597b748d5a8d64cea7af3fade4f2d84eaaf386dc8495'

#     # Convert moduli from hex strings to integers
#     N1 = Integer('0x' + N1_hex)
#     N2 = Integer('0x' + N2_hex)

#     # Public exponent e
#     e = Integer(65537)

#     # Compute q1 and q2
#     q1 = N1 // p
#     q2 = N2 // p

#     # Verify that N1 == p * q1
#     assert N1 == p * q1, "Failed to factor N1 correctly."
#     assert N2 == p * q2, "Failed to factor N2 correctly."

#     print("Successfully factored N1 and N2.")

#     # Compute phi(N1)
#     phi_N1 = (p - 1) * (q1 - 1)

#     # Compute d = e^{-1} mod phi_N1
#     d = inverse_mod(e, phi_N1)

#     # Ensure p > q for RSA key construction
#     if p > q1:
#         p_big = int(p)
#         q_small = int(q1)
#     else:
#         p_big = int(q1)
#         q_small = int(p)

#     # Compute additional RSA parameters
#     d_int = int(d)
#     e_int = int(e)
#     N1_int = int(N1)

#     dP = d_int % (p_big - 1)
#     dQ = d_int % (q_small - 1)
#     qInv = inverse_mod(q_small, p_big)
#     qInv_int = int(qInv)

#     # Construct the RSA private key
#     private_key = RSA.construct((N1_int, e_int, d_int, p_big, q_small))

#     print("RSA private key constructed.")

#     # Read the encrypted file
#     with open('hw6.pdf.enc.asc', 'r') as f:
#         enc_data = f.read()

#     # Remove headers and footers to extract base64-encoded data
#     match = re.search(r'-----BEGIN PRETTY BAD ENCRYPTED MESSAGE-----(.*?)-----END PRETTY BAD ENCRYPTED MESSAGE-----', enc_data, re.DOTALL)
#     if match:
#         b64_data = match.group(1)
#     else:
#         # If headers are not found, assume the entire file is the data
#         b64_data = enc_data

#     # Clean up the base64 data
#     b64_data = ''.join(b64_data.strip().split())

#     # Decode the base64 data
#     encrypted_data = base64.b64decode(b64_data)

#     # Parse the encrypted data
#     # Read the first 4 bytes to get the length of the MPI (little-endian)
#     l_bytes = encrypted_data[:4]
#     (l,) = struct.unpack('<I', l_bytes)
#     print(f"Length of encrypted AES key (MPI): {l} bytes")

#     # Extract the RSA-encrypted AES key
#     rsa_encrypted_aes_key = encrypted_data[4:4+l]

#     # The remaining data consists of the IV and the ciphertext
#     remaining = encrypted_data[4+l:]
#     iv = remaining[:AES.block_size]  # AES.block_size is typically 16 bytes
#     ciphertext = remaining[AES.block_size:]

#     # Decrypt the AES key using RSA private key
#     cipher_rsa = PKCS1_v1_5.new(private_key)
#     sentinel = Random.new().read(15)  # Random sentinel in case of failure
#     aes_key = cipher_rsa.decrypt(rsa_encrypted_aes_key, sentinel)
#     if aes_key == sentinel or len(aes_key) != 32:
#         print("Failed to decrypt AES key.")
#         return
#     else:
#         print("AES key decrypted successfully.")

#     # Decrypt the ciphertext using AES in CBC mode
#     cipher_aes = AES.new(aes_key, AES.MODE_CBC, iv)
#     padded_plaintext = cipher_aes.decrypt(ciphertext)

#     # Remove PKCS7 padding
#     pad_len = padded_plaintext[-1]
#     if isinstance(pad_len, str):  # For Python 2 compatibility
#         pad_len = ord(pad_len)
#     plaintext = padded_plaintext[:-pad_len]

#     # Save the decrypted file
#     with open('hw6.pdf', 'wb') as f:
#         f.write(plaintext)

#     print("Decryption complete. Plaintext saved to 'hw6.pdf'.")

# if __name__ == '__main__':
#     main()
