import sys
import time
from datetime import datetime

def main():
    enc_filename = sys.argv[1]
    with open(enc_filename, 'rb') as f:
        ciphertext = f.read()
    
    N = 50 
    end_time = int(time.time())
    start_time = end_time - (20 * 24 * 60 * 60)  # 20 days

    found = False
    for seed in range(start_time, end_time + 1):
        plaintext_candidate = decrypt_seed(ciphertext, seed, N)
        if is_valid(plaintext_candidate):
            plaintext = decrypt_seed(ciphertext, seed, len(ciphertext))
            with open('hw2.tex', 'wb') as f_out:
                f_out.write(plaintext)
            print('Successful')
            found = True
            break
    if not found:
        print('Failed')

def decrypt_seed(ciphertext, seed, length):
    keystream = gen_keystream(seed, length)
    plaintext = bytes([c ^ k for c, k in zip(ciphertext[:length], keystream)])
    return plaintext

def gen_keystream(seed, length):
    state = seed & 0x7fffffff
    keystream = bytearray()
    i = 0
    while len(keystream) < length:
        state = (state * 1103515245 + 12345) & 0x7fffffff
        key_bytes = state.to_bytes(4, byteorder='little', signed=False)
        for b in key_bytes:
            keystream.append(b)
            if len(keystream) >= length:
                break
    return keystream[:length]

def is_valid(plaintext):
    try:
        text = plaintext.decode('ascii')
    except UnicodeDecodeError:
        return False

    if not all((32 <= ord(c) <= 126) or c in '\n\r\t' for c in text):
        return False

    return True
    
if __name__ == '__main__':
    main()
