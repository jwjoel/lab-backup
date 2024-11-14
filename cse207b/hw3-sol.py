#!/usr/bin/env python3

from Crypto.Cipher import AES
from Crypto import Random
import requests
import base64
import sys

def pad(msg,blocksize=16):
    n = blocksize-(len(msg)%16)
    return msg+bytes([n]*n)

def strip_padding(msg):
    padlen = msg[-1]
    assert 0 < padlen and padlen <= 16
    assert msg[-padlen:] == bytes([padlen]*padlen)
    return msg[:-padlen]

def enc(key,msg):
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(pad(msg))

def dec(key,msg):
    cipher = AES.new(key, AES.MODE_CBC, msg[:16])
    text = strip_padding(cipher.decrypt(msg[16:]))
    return text

def get_status(u: str, session: bytes):
    assert isinstance(u, str)
    assert isinstance(session, bytes)

    cookies = {'session': base64.b64encode(session).decode("ascii")}
    req = requests.get(u, cookies=cookies)
    
    print("HTTP Response:", req.status_code)

def solve(url: str, cookie_bytes: bytes) -> str:
    block_size = 16
    blocks = [cookie_bytes[i:i+block_size] for i in range(0, len(cookie_bytes), block_size)]
    num_blocks = len(blocks)
    plaintext = b''

    def check_padding(session_cipher):
        cookies = {'session': base64.b64encode(session_cipher).decode('ascii')}
        req = requests.get(url, cookies=cookies)
        # if req.status_code != 500:
        #     print(req, cookies)
        return req.status_code == 200

    for block_index in range(num_blocks-1, 0, -1):
        current_block = blocks[block_index]
        previous_block = blocks[block_index-1]
        intermediate = [0]*block_size
        decrypted_block = [0]*block_size

        for byte_index in range(block_size-1, -1, -1):
            padding_value = block_size - byte_index
            found = False
            for guess in range(256):
                modified_block = bytearray(previous_block)
                modified_block[byte_index] = guess

                for i in range(byte_index+1, block_size):
                    modified_block[i] = intermediate[i] ^ padding_value
                modified_cipher = bytes(modified_block) + current_block
                if check_padding(modified_cipher):
                    intermediate_byte = guess ^ padding_value
                    intermediate[byte_index] = intermediate_byte
                    decrypted_byte = intermediate_byte ^ previous_block[byte_index]
                    decrypted_block[byte_index] = decrypted_byte
                    # found = True
                    # print("Found")
                    break

            # if not found:
            #     print(f"Failed")

        plaintext = bytes(decrypted_block) + plaintext

    # remove padding
    padlen = plaintext[-1]
    plaintext = plaintext[:-padlen]
    return plaintext.decode('utf-8', errors='ignore')

if __name__=='__main__':
    url = sys.argv[1]
    encoded_cookie = sys.argv[2]
    cookie_bytes = base64.b64decode(encoded_cookie)

    print("Encrypted cookie bytes", cookie_bytes)
    print(solve(url, cookie_bytes))
    exit(0)
