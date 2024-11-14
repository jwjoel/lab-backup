#!/usr/bin/env python3

import ecdsa
import hashlib
import binascii
from ecdsa.util import string_to_number, number_to_string
import sys
import re
from urllib.parse import urlparse, parse_qs

def extract_token_and_message(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    token = query_params['token'][0]
    message = parsed_url.query.split('&', 1)[1]
    return token, message.encode()

def extract_rs(signature_bytes):
    l = len(signature_bytes) // 2
    r = string_to_number(signature_bytes[:l])
    s = string_to_number(signature_bytes[l:])
    return r, s

url1 = sys.argv[1]
url2 = sys.argv[2]

# Extract tokens and messages from the URLs
token1, message1 = extract_token_and_message(url1)
token2, message2 = extract_token_and_message(url2)

# Convert tokens from hex to bytes
sig1_bytes = binascii.unhexlify(token1)
sig2_bytes = binascii.unhexlify(token2)

# Curve and domain parameters
curve = ecdsa.NIST256p
n = curve.order

# Given public key (from the problem statement)
pubkey_hex = 'aff7647dd6d4661dd3a05c76eeda1bd5fff0d43c2661427fd93ff3d0661e382d1512c32683351cd8e2f50d986cad3b2119b2bb1c461dfa73d6d678e7c98bebb6'
pubkey_bytes = binascii.unhexlify(pubkey_hex)
vk = ecdsa.VerifyingKey.from_string(pubkey_bytes, curve=curve)

# Verify that both signatures are valid
assert vk.verify(sig1_bytes, message1, hashfunc=hashlib.sha256)
assert vk.verify(sig2_bytes, message2, hashfunc=hashlib.sha256)

# Extract (r, s) from each signature
r1, s1 = extract_rs(sig1_bytes)
r2, s2 = extract_rs(sig2_bytes)

# Ensure that r values are the same
assert r1 == r2
r = r1

# Compute hashes of the messages
e1 = string_to_number(hashlib.sha256(message1).digest())
e2 = string_to_number(hashlib.sha256(message2).digest())

# Compute k
k = ((e1 - e2) * pow(s1 - s2, -1, n)) % n

# Compute private key d
d = ((s1 * k - e1) * pow(r, -1, n)) % n

# Create a signing key object with the private key
sk = ecdsa.SigningKey.from_secret_exponent(d, curve=curve)

# Desired message to sign
message3 = b'user=admin&get_file=hw7.pdf'

# Sign the message
signature3 = sk.sign(message3, hashfunc=hashlib.sha256)

# Verify the signature using the public key
assert vk.verify(signature3, message3, hashfunc=hashlib.sha256)

# Prepare the token (signature) and query string
token = binascii.hexlify(signature3).decode()
query_string = message3.decode()

# The forged URL
forged_url = f"https://cse207b.nh.cryptanalysis.fun/hw7/api?token={token}&{query_string}"
print(forged_url)
