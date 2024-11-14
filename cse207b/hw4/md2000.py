"""
This file provides the following two functions:

 md2000.hash_md2000(message)
	Hash a message with the MD2000 hash function. Output is 32 bytes long.
	MD2000 follows a Merkle-Damgard construction, with a 256-byte block size and 32-byte state size.

 md2000.find_collision(prefix1, prefix2)
	Compute a chosen-prefix collision: return two bytestrings (m1, m2) such that...
		hash(m1) == hash(m2),  
		m1.startswith(prefix1), and
		m2.startswith(prefix2).
	Additionally, len(m1) == len(m2) and is a multiple of the block size.

You don't need to worry about how these functions are implemented: you can safely ignore the rest of this file.
"""

#### If you're curious...
#
# It's actually a little tricky to come up with a hash function where chosen-prefix collisions can be computed in seconds but 2nd-preimage attacks are still hard.
# MD5 chosen-prefix collisions are too slow (many CPU-hours), and MD4 preimage attacks are too easy.
# For MD2000 we use a compression function based on a well-known lattice problem called SIS.
# SIS is the problem of solving an underdetermined linear system Ax = t (mod q), but subject to the constraint that the solution x be small.
# (i.e., ||x|| < B for some B, where the norm can be l2 or l_infty)
# For appropriately chosen parameters, SIS is believed to be hard even for quantum computers.
#
# Specifically, MD2000's compression function works like this: Let A be a fixed uniformly random n-by-m matrix mod 509, with m > n.
# Encode the hash input as an m-dimensional vector x with coefficients in the range [0, 255].
# The output is A*x (mod 509), which is an n-dimensional vector. We then hash this value with SHA3 to make it smaller.
#
# Finding a preimage (or second preimage) means finding a small (in l_infty norm) x such that A*x = target (mod 509), which is hard under the SIS assumption.
# Finding a collision means finding x1 and x2 such that A*x1 = A*x2, or equivalently, A*(x1-x2) = 0 (mod 509).
# But this is easy to do: first find any (not necessarily small) solution x to A*x = 0 (mod 509).
# Then choose x1 and x2 to have coefficients in [0, 255] such that x1 - x2 = x (mod 509).
# Now the hash inputs that encode to vectors x1 and x2 are a hash collision: A*x1 = A*x2 (mod 509).
#
# The only difficulty here is choosing concrete parameters (namely, the dimensions of the matrix A) such that SIS is believed to be hard.
# Our parameters might not be chosen correctly! It might (or might not) be possible to do a second-preimage attack against it.
# Extra credit if you can carry out a second-preimage attack.
###

from struct import pack, iter_unpack
from hashlib import shake_128 as shake, sha3_256 as sha3

def _vsmul(v,s,m):
	return [vi*s % m for vi in v]
def _vadd(v1,v2,m):
	return [(a+b)%m for (a,b) in zip(v1,v2)]
def _vsub(v1,v2,m):
	return [(a-b)%m for (a,b) in zip(v1,v2)]
def _vdot(v1,v2,m):
	return sum((a*b)%m for (a,b) in zip(v1,v2)) % m

#A_NROWS = N = 224
#A_NCOLS = M = 508
#Q = 509
#LG_Q = 9
## For these parameters
## * output length (= n lg q) is 252 bytes
## * input length (= m (lg q - 1) ) is 508 bytes
## * block size = input length - output length = 256 bytes
## * IV length = output length = 288 bytes
#BLOCK_SIZE = 256
#STATE_SIZE = 252
#OUTPUT_SIZE = 32
#

# For the attack to work, the dimension of an input block as a vector (256) must be at least as large as the state dimension (= n = 224)

A_NROWS = N = 224
A_NCOLS = M = 32 + 256
Q = 509
LG_Q = 9
BLOCK_SIZE = 256
STATE_SIZE = 32
OUTPUT_SIZE = 32

assert (STATE_SIZE + BLOCK_SIZE) * 8 == M * (LG_Q - 1)

IV = shake(b"MD2000 IV").digest(STATE_SIZE)

A = [
	shake(f"A matrix row {i}".encode('utf-8')).digest(2*A_NCOLS)
	for i in range(A_NROWS)
]
for i in range(A_NROWS):
	A[i] = [(a<<8 | b) % Q for (a,b) in zip(*(2*[iter(A[i])]))]
	# The above is a fancy one-liner to treat pairs of bytes as uint16s and reduce them mod Q (=509)
# A is now a ~uniform A_NROWS-by-A_NCOLS matrix mod Q

def _mmul(M, v):
	# Return M * v (mod Q)
	return [_vdot(Mi, v, Q) for Mi in M]

def _mulA(v):
	# Return A * v (mod Q)
	return _mmul(A, v)


def _input_to_vector(s):
	# Convert a bytestring of length A_NCOLS into an m-dimensional **short** vector mod Q
	if len(s) != M:
		raise ValueError(f"s should have length m (={A_NCOLS})")
	return s # (entries are already between 0 and 256)

def _output_to_bits(v):
	# Convert an n-dimensional vector mod Q into bits
	if len(v) != N:
		raise ValueError(f"v should have dimension n (={A_NROWS})")
	for elem in v:
		for i in range(LG_Q):
			yield (elem>>i)&1

def _bits_to_bytes(bits):
	char = 0
	for (i,bit) in enumerate(bits):
		i &= 7 
		char = char | (bit << i)
		if i == 7:
			yield char
			char = 0

def _output_to_bytes(v):
	# Convert an n-dimensional vector mod Q into bytes
	if len(v) != N:
		raise ValueError(f"y should have dimension n (={A_NROWS})")
	assert (N * LG_Q) % 8 == 0
	return bytes(_bits_to_bytes(_output_to_bits(v)))

def _compression_function(state, block):
	if len(state) != STATE_SIZE:
		raise ValueError(f"state should be {STATE_SIZE} bytes long")
	if len(block) != BLOCK_SIZE:
		raise ValueError(f"block should be {BLOCK_SIZE} bytes long")
	out = _output_to_bytes(_mulA(_input_to_vector(state + block))) # compute A * (state | block) mod q...
	return sha3(out).digest() # then use sha3 to compress to 32 bytes

def _pad_message(msg):
	# Padding is as follows:
	# Append a 0x01 byte
	# Then append zero or more 0x00 bytes to make the output 8 bytes less than a multiple of block length
	# Then append len(msg) as a little-endian uint64
	num_zerobytes = ((BLOCK_SIZE - 8) - (len(msg) + 1)) % BLOCK_SIZE
	assert 0 <= num_zerobytes < BLOCK_SIZE
	assert (len(msg) + 1 + num_zerobytes + 8) % BLOCK_SIZE == 0
	padding = b"\x01" + (b"\x00" * num_zerobytes) + pack("<Q", len(msg))
	assert (len(msg) + len(padding)) % BLOCK_SIZE == 0
	return msg + padding

def _hash_nopad(msg):
	state = IV
	assert len(msg) % BLOCK_SIZE == 0
	num_blocks = len(msg) // BLOCK_SIZE
	for i in range(num_blocks):
		state = _compression_function(state, msg[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
	return state

def hash_md2000(msg):
	"""
	Hash a message (which should be a bytes object) using the MD2000 hash function.
	Output is a bytes object of length 32
	"""
	return _hash_nopad(_pad_message(msg))

#### Now, the attack part ####

Astate = [Ai[:STATE_SIZE] for Ai in A] # columns of A that get multiplied by the state
Ablock = [Ai[STATE_SIZE:] for Ai in A] # columns of A that get multiplied by the message block

def _modQ_to_pair(x):
	# Return b1, b2 such that b1 - b2 == x (mod 509) and 0 <= bi <= 255
	assert Q == 509
	x %= 509
	if x <= 255: return x, 0
	if x >= 256: return 0, (-x) % 509

def _load_matrix(filename):
	# Load an N-by-N matrix over Zmod(509).
	# Format: each entry is 2 bytes little-endian; first 2*N bytes are the first row, next 2*N bytes are the second row, and so on.
	with open(filename, 'rb') as f:
		data = f.read()
	assert len(data) == 2 * N * N, f"Matrix file ({filename}) is wrong length"
	return list(iter_unpack(
		"<" + "H"*N, # little-endian, N unsigned shorts
		data
	))

def find_collision(prefix1, prefix2):
	prefix1 = bytes(prefix1)
	prefix2 = bytes(prefix2)
	if prefix1 == prefix2:
		prefix1 += b"\x00"
		prefix2 += b"\xff"
	m1 = prefix1 + b"\x00"*max(0,len(prefix2)-len(prefix1))
	m2 = prefix2 + b"\x00"*max(0,len(prefix1)-len(prefix2))
	m1 += b"\x00" * ((-len(m1)) % BLOCK_SIZE)
	m2 += b"\x00" * ((-len(m2)) % BLOCK_SIZE)
	state1 = _hash_nopad(m1)
	state2 = _hash_nopad(m2)
	# We now want to find blocks g1 and g2 such that compression_function(state1, g1) == compression_function(state2, g2)
	# We want Astate*state1 + Ablock * g1 == Astate*state2 + Ablock * g2
	# i.e., Ablock * (g1 - g2) == Astate*(state2-state1)
	target = _mmul(Astate, [(s2 - s1) % Q for (s1,s2) in zip(state1, state2)])
	
	# Solve the (underconstrained) linear system Ablock * x == target (mod Q)
	# Load the inverse of a square submatrix of Ablock from a file
	Abinv = _load_matrix("collision_auxdata")
	blk = _mmul(Abinv, target)
	blk += [0]*(BLOCK_SIZE - len(blk))

	assert _mmul(Ablock, blk) == target
	# Find g1, g2 s.t. g1 - g2 == blk
	g1, g2 = zip(*(_modQ_to_pair(x) for x in blk))
	g1 = bytes(g1)
	g2 = bytes(g2)
	assert [(a-b)%Q for (a,b) in zip(g1,g2)] == blk 
	assert _mmul(Ablock, _vsub(g1,g2,Q)) == _mmul(Astate, _vsub(state2, state1, Q))
	msg1 = m1 + g1
	msg2 = m2 + g2
	assert hash_md2000(msg1) == hash_md2000(msg2)
	assert msg1.startswith(prefix1) and msg2.startswith(prefix2)
	return msg1, msg2

def collision_length(len1, len2):
    return 256*((max(len1, len2)+255)//256 + 1)