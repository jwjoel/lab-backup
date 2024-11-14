import struct
import time
import md2000

# ulna packet: code, timestamp, attrs

ULNA_ATTRS_BY_NAME = {
"username": b"u",
"password": b"p",
"filename": b"f",
"expires": b"x",
"session-state": b"s",
}
ULNA_ATTRS_BY_BYTE = {byte: name for (name, byte) in ULNA_ATTRS_BY_NAME.items()}


def compute_mac(packet, secret):
	""" Compute the MAC for a response packet. Response packet + MAC = ULNA token. """
	return md2000.hash_md2000(packet + secret)

def verify_mac(token, secret):
	""" Verify a token (i.e., a response packet + MAC). Return the packet if the MAC verifies, raise ValueError otherwise. """
	packet = token[:-md2000.OUTPUT_SIZE]
	mac = token[-md2000.OUTPUT_SIZE:]
	if compute_mac(packet, secret) != mac:
		raise ValueError("Token has incorrect MAC")
	return packet

### Here are some convenience functions for parsing and constructing ULNA packets.

def parse_ulna_packet(packet):
	"""
	Parse an ULNA packet, returning a dictionary mapping human-friendly attribute names to their values.
	The dictionary also includes "is_request" (True for request packets, False for responses) and "timestamp"
	"""
	code = packet[0] # First byte indicates the packet type
	if code not in [1, 2]:
		raise ValueError("Packet is corrupt - bad packet type")
	out = {"is_request": code == 1} # 0x01 means ULNA request. 0x02 means response
	timestamp = int.from_bytes(packet[1:5], byteorder="little") # Next four bytes are a 32-bit timestamp
	out["timestamp"] = timestamp
	i = 5
	while i < len(packet): # Next come a list of attributes. For each...
		attrtype = packet[i:i+1] # one byte indicating what attribute this is
		attrlen = int.from_bytes(packet[i+1:i+3], byteorder="little") # two-byte length
		i += 3
		if attrlen > 1024:
			raise ValueError("Packet is corrupt - bad attribute length (>1024)")
		if i + attrlen > len(packet):
			raise ValueError("Packet appears truncated")
		attrval = packet[i:i+attrlen] # the actual value of this attribute
		i += attrlen
		if attrtype not in ULNA_ATTRS_BY_BYTE:
			print("Warning: unknown attribute:", attrtype)
		else:
			attrtype = ULNA_ATTRS_BY_BYTE[attrtype]
		if attrtype in out:
			raise ValueError(f"Packet has duplicate attribute: {attrtype}")
		out[attrtype] = attrval
	return out

def make_ulna_packet(is_request, timestamp, attrs):
	"""
	Construct an ULNA packet.
	attrs should be a list of (attribute name, attribute value) pairs.
	Supported attribute names are in the ULNA_ATTRS_BY_NAME dict.
	Attribute values should be bytestrings of length <=1024.
	For convenience, this function also accepts ints (between 0 and 2**32-1) which get serialized as little-endian uint32s.
	"""
	pkt = bytearray()
	pkt += b"\x01" if is_request else b"\x02"
	pkt += timestamp.to_bytes(length=4, byteorder="little")
	print("timestamp", timestamp.to_bytes(length=4, byteorder="little"))
	print(timestamp)
	new_timestamp = timestamp + 300
	print("new_timestamp", new_timestamp.to_bytes(length=4, byteorder="little"))
	for (attrname, attrvalue) in attrs:
		if type(attrvalue) == int and 0 <= attrvalue < 2**32:
			attrvalue = attrvalue.to_bytes(length=4, byteorder="little")
			print("int captuyred", attrvalue)
		if type(attrvalue) != bytes:
			raise TypeError("Attribute values must be bytestrings or uint32s")
		if len(attrvalue) > 1024:
			raise ValueError("Maximum length of any attribute is 1024 bytes")
		pkt += ULNA_ATTRS_BY_NAME[attrname] # one byte for which attribute
		pkt += len(attrvalue).to_bytes(length=2, byteorder="little") # two bytes for length of the value field
		print("length for", attrname, len(attrvalue).to_bytes(length=2, byteorder="little"))
		pkt += attrvalue # between 0 and 1024 bytes for the value field
	return bytes(pkt)

