import sys
import time
import requests
import md2000
import ulna

ULNA_ATTRS_BY_NAME = {
    "username": b"u",
    "password": b"p",
    "filename": b"f",
    "expires": b"x",
    "session-state": b"s",
}
ULNA_ATTRS_BY_BYTE = {byte: name for (name, byte) in ULNA_ATTRS_BY_NAME.items()}
ATTR_TYPE_CODES = {name: ord(code) for name, code in ULNA_ATTRS_BY_NAME.items()}

def send_to_ulna_server(ulna_server_url, ulna_request):
    ulna_request_hex = ulna_request.hex()
    params = {'ulna_request': ulna_request_hex}
    response = requests.get(ulna_server_url, params=params)
    response.raise_for_status()
    return bytes.fromhex(response.text.strip())

def download_file_with_token(file_server_url, filename, token):
    params = {'ulna_token': token.hex()}
    response = requests.get(f"{file_server_url}{filename}", params=params)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

def make_ulna_packet(is_request, timestamp, attrs):
    pkt = bytearray()
    pkt += b"\x01" if is_request else b"\x02"
    pkt += timestamp.to_bytes(length=4, byteorder="little")
    for (attrname, attrvalue) in attrs:
        if type(attrvalue) == int and 0 <= attrvalue < 2**32:
            attrvalue = attrvalue.to_bytes(length=4, byteorder="little")
        if type(attrvalue) != bytes:
            raise TypeError("Attribute values must be bytestrings or uint32s")
        if len(attrvalue) > 1024:
            raise ValueError("Maximum length of any attribute is 1024 bytes")
        pkt += ULNA_ATTRS_BY_NAME[attrname]
        pkt += len(attrvalue).to_bytes(length=2, byteorder="little") 
        pkt += attrvalue  
    return bytes(pkt)

def main():
    ULNA_SERVER_URL = sys.argv[1]
    ACCESSIBLE_FILENAME = 'kitten.jpg'
    FILE_SERVER_URL = 'https://cse207b.nh.cryptanalysis.fun/hw4/files/'
    FORGED_FILENAME = 'hw4.pdf'
    timestamp = int(time.time())
    expire = timestamp + 300
    username = 'testuser'.encode('utf-8')
    password = 'testpass'.encode('utf-8')
    
    # prepare attributes for all packets
    attrs_pkt1 = [
        ("filename", ACCESSIBLE_FILENAME.encode('utf-8')),
        ("username", username),
        ("password", password),
    ]
    attrs_pkt1_r = [
        ("expires", expire),
        ("filename", ACCESSIBLE_FILENAME.encode('utf-8')),
    ]
    attrs_pkt2 = [
        ("expires", expire),
        ("filename", FORGED_FILENAME.encode('utf-8')),
    ]

    # build packets with fake session-state
    # build pkt1 pkt1 real
    pkt1_prefix = ulna.make_ulna_packet(True, timestamp, attrs_pkt1)
    pkt1_r_prefix = ulna.make_ulna_packet(False, timestamp, attrs_pkt1_r)
    session_state_type = ULNA_ATTRS_BY_NAME['session-state']
    pkt1_prefix += session_state_type
    pkt1_r_prefix += session_state_type
    # length take 2 bytes, will set it after calculating collision length
    pkt1_prefix_length = len(pkt1_prefix) + 2  # with length field
    pkt1_r_prefix_length = len(pkt1_r_prefix) + 2
    # pkt2
    pkt2_prefix = ulna.make_ulna_packet(False, timestamp, attrs_pkt2)
    pkt2_prefix += session_state_type
    pkt2_prefix_length = len(pkt2_prefix) + 2  # with length field

    # total length for collision
    collision_total_length = md2000.collision_length(pkt1_r_prefix_length, pkt2_prefix_length)
    session_state_value_length = collision_total_length - pkt1_prefix_length
    session_state_r_value_length = collision_total_length - pkt1_r_prefix_length
    session_state_value_length_pkt2 = collision_total_length - pkt2_prefix_length

    # build full prefixes including length fields
    pkt1_prefix_full = pkt1_prefix 
    pkt1_prefix_full += session_state_r_value_length.to_bytes(2, 'little')
    pkt1_prefix_r_full = pkt1_r_prefix  
    pkt1_prefix_r_full += session_state_r_value_length.to_bytes(2, 'little')
    pkt2_prefix_full = pkt2_prefix
    pkt2_prefix_full += session_state_value_length_pkt2.to_bytes(2, 'little')

    # collision blocks
    m1, m2 = md2000.find_collision(pkt1_prefix_r_full, pkt2_prefix_full)
    m1_without_pkt1_prefix_r_full = m1[len(pkt1_prefix_r_full):]
    # print("m1_without_pkt1_prefix_r_full", m1_without_pkt1_prefix_r_full)
    # print("pkt1_prefix_full", pkt1_prefix_full)
    pkt1_prefix_full = pkt1_prefix_full + m1_without_pkt1_prefix_r_full
    # print("pkt1_prefix_full", pkt1_prefix_full)
    # send kitten one to ULNA server
    token = send_to_ulna_server(ULNA_SERVER_URL, pkt1_prefix_full)
    # print("token for m1", token)
    # print('token', token)
    # print('m1', m1)
    # extract MAC and forge token with m2
    mac_size = md2000.OUTPUT_SIZE
    mac = token[-mac_size:]
    # print(mac)
    forged_token = m2 + mac

    # print("token_r", token)
    # print("-" * 20)
    # print("token_f", forged_token)
    print(forged_token.hex())

    # download_file_with_token(FILE_SERVER_URL, FORGED_FILENAME, forged_token)

if __name__ == "__main__":
    main()
