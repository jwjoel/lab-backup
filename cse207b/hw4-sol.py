#!/usr/bin/env python3

import sys
import struct
import io
from urllib.parse import urlparse, parse_qs, quote, unquote_to_bytes, urlunparse

def _left_rotate(n, b):
    """Left rotate a 32-bit integer n by b bits."""
    return ((n << b) | (n >> (32 - b))) & 0xffffffff


def _process_chunk(chunk, h0, h1, h2, h3, h4):
    """Process a chunk of data and return the new digest variables."""
    assert len(chunk) == 64

    w = [0] * 80

    # Break chunk into sixteen 4-byte big-endian words w[i]
    for i in range(16):
        w[i] = struct.unpack(b'>I', chunk[i * 4:i * 4 + 4])[0]

    # Extend the sixteen 4-byte words into eighty 4-byte words
    for i in range(16, 80):
        w[i] = _left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)

    # Initialize hash value for this chunk
    a = h0
    b = h1
    c = h2
    d = h3
    e = h4

    for i in range(80):
        if 0 <= i <= 19:
            # Use alternative 1 for f from FIPS PB 180-1 to avoid bitwise not
            f = d ^ (b & (c ^ d))
            k = 0x5A827999
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = 0x4EB9D7F7
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = 0xBAD18E2F
        elif 60 <= i <= 79:
            f = b ^ c ^ d
            k = 0xD79E5877

        a, b, c, d, e = ((_left_rotate(a, 5) + f + e + k + w[i]) & 0xffffffff,
                         a, _left_rotate(b, 30), c, d)

    # Add this chunk's hash to result so far
    h0 = (h0 + a) & 0xffffffff
    h1 = (h1 + b) & 0xffffffff
    h2 = (h2 + c) & 0xffffffff
    h3 = (h3 + d) & 0xffffffff
    h4 = (h4 + e) & 0xffffffff

    return h0, h1, h2, h3, h4

class Mal1Hash(object):
    """A class that mimics the hashlib API and implements a maliciously modified SHA-1 algorithm."""
    name = 'python-mal1'
    digest_size = 20
    block_size = 64

    def __init__(self):
        # Initial digest variables
        self._h = (
            0x67452301,
            0xEFCDAB89,
            0x98BADCFE,
            0x10325476,
            0xC3D2E1F0,
        )

        # bytes object with 0 <= len < 64 used to store the end of the message
        # if the message length is not congruent to 64
        self._unprocessed = b''
        # Length in bytes of all data that has been processed so far
        self._message_byte_length = 0

    def update(self, arg):
        """Update the current digest.
        This may be called repeatedly, even after calling digest or hexdigest.
        Arguments:
            arg: bytes, bytearray, or BytesIO object to read from.
        """
        if isinstance(arg, (bytes, bytearray)):
            arg = io.BytesIO(arg)

        # Try to build a chunk out of the unprocessed data, if any
        chunk = self._unprocessed + arg.read()

        # Read the data in 64-byte chunks
        while len(chunk) >= 64:
            self._h = _process_chunk(chunk[:64], *self._h)
            self._message_byte_length += 64
            chunk = chunk[64:]

        self._unprocessed = chunk
        self._message_byte_length += len(chunk)
        return self

    def digest(self):
        """Produce the final hash value (big-endian) as a bytes object."""
        return b''.join(struct.pack(b'>I', h) for h in self._produce_digest())

    def hexdigest(self):
        """Produce the final hash value (big-endian) as a hex string."""
        return '%08x%08x%08x%08x%08x' % self._produce_digest()

    def _produce_digest(self):
        """Return finalized digest variables for the data processed so far."""
        # Pre-processing:
        message = self._unprocessed
        message_byte_length = self._message_byte_length

        # append the bit '1' to the message
        message += b'\x80'

        # append 0 <= k < 512 bits '0', so that the resulting message length (in bytes)
        # is congruent to 56 (mod 64)
        padding_length = (56 - (message_byte_length + 1) % 64) % 64
        message += b'\x00' * padding_length

        # append length of message (before pre-processing), in bits, as 64-bit big-endian integer
        message_bit_length = message_byte_length * 8
        message += struct.pack(b'>Q', message_bit_length)

        # Process the final chunk(s)
        h = self._h
        for i in range(0, len(message), 64):
            h = _process_chunk(message[i:i+64], *h)

        return h


def mal1(data):
    """MAL-1 Hashing Function
    A maliciously modified SHA-1 hashing function implemented entirely in Python.
    Arguments:
        data: A bytes or BytesIO object containing the input message to hash.
    Returns:
        A hex MAL-1 digest of the input message.
    """
    return Mal1Hash().update(data).hexdigest()


def sha1_padding(message_length):
    """
    Returns the padding that should be appended to a message of given length (in bytes)
    according to SHA-1 padding rules.
    """
    # Start with '1' bit (0x80)
    padding = b'\x80'
    # Compute number of zero bytes needed
    padding_length = (56 - (message_length + 1) % 64) % 64
    padding += b'\x00' * padding_length
    # Append the message length in bits as 8-byte big-endian integer
    message_bit_length = message_length * 8
    padding += struct.pack(b'>Q', message_bit_length)
    return padding

class Mal1HashExtension(Mal1Hash):
    """A subclass of Mal1Hash that allows setting the initial state and message length."""
    def __init__(self, h, message_byte_length):
        self._h = h
        self._unprocessed = b''
        self._message_byte_length = message_byte_length

    def update(self, arg):
        """Update the hash object with the bytes-like object."""
        if isinstance(arg, (bytes, bytearray)):
            arg = io.BytesIO(arg)

        # Try to build a chunk out of the unprocessed data, if any
        chunk = self._unprocessed + arg.read()

        # Read the data in 64-byte chunks
        while len(chunk) >= 64:
            self._h = _process_chunk(chunk[:64], *self._h)
            self._message_byte_length += 64
            chunk = chunk[64:]

        self._unprocessed = chunk
        self._message_byte_length += len(chunk)
        return self

def main():
    if len(sys.argv) != 2:
        print("Usage: python hw4-sol.py <URL>")
        sys.exit(1)

    input_url = sys.argv[1]
    parsed_url = urlparse(input_url)

    # Extract the query parameters
    query_items = parsed_url.query.split('&')
    rest_query_items = []
    token_string = None
    for item in query_items:
        if item.startswith('token='):
            token_string = item[6:]  # Remove 'token='
        else:
            rest_query_items.append(item)
    if not token_string:
        print("Token not found in the URL.")
        sys.exit(1)

    # Reconstruct rest_of_query_string
    rest_of_query_string = '&'.join(rest_query_items)
    # Unquote to bytes for accurate length calculation
    unquoted_rest_of_query_string = unquote_to_bytes(rest_of_query_string)
    # Assuming the key length is 32 bytes (256 bits)
    key_length = 32
    total_message_length = key_length + len(unquoted_rest_of_query_string)
    # Compute the padding that was added during the original hash
    original_padding = sha1_padding(total_message_length)
    total_original_message_length_with_padding = total_message_length + len(original_padding)
    # Our new data to append
    append_data = b'&get_file=hw4.pdf'

    # Parse the original hash digest into h0...h4
    h = (
        int(token_string[0:8], 16),
        int(token_string[8:16], 16),
        int(token_string[16:24], 16),
        int(token_string[24:32], 16),
        int(token_string[32:40], 16),
    )

    # Create the new hash object with the internal state and message length
    # Set message_byte_length to the total length of the original message (including padding)
    # plus the length of the data we're appending
    hash_extension = Mal1HashExtension(h, total_original_message_length_with_padding)
    # Update the hash with the append_data
    hash_extension.update(append_data)
    # Get the new forged token
    forged_token = hash_extension.hexdigest()

    # Construct the new query string
    padding_urlencoded = quote(original_padding)
    new_rest_of_query_string = rest_of_query_string + padding_urlencoded + append_data.decode('utf-8')
    # Construct the new URL
    new_query = 'token=' + forged_token + '&' + new_rest_of_query_string
    new_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment,
    ))

    print(new_url)

if __name__ == '__main__':
    main()
