import time
from urllib.request import urlopen
from urllib.error import HTTPError

import ulna

ULNA_SERVER_URL="https://cse207b.nh.cryptanalysis.fun/hw4/ulna/login"
FILE_SERVER_URL="http://cse207b.nh.cryptanalysis.fun/hw4/files/"

def get(url):
	try:
		with urlopen(url) as resp:
			return resp.read()
	except HTTPError as e:
		e.add_note("Server said: " + str(e.fp.read()))
		raise e
def download(url, filename):
	if "http://" in filename or "https://" in filename:
		raise ValueError("URL goes first, filename goes second")
	resp = get(url)
	with open(filename, "xb") as f:
		f.write(resp)

def send_to_ulna_server(ulna_request):
	return bytes.fromhex(get(ULNA_SERVER_URL + "?ulna_request=" + ulna_request.hex()).decode("utf-8"))

def download_file_with_token(filename, token):
	download(FILE_SERVER_URL + filename + "?ulna_token=" + token.hex(), filename)

if __name__ == "__main__":
	timestamp = int(time.time())
	username = input("Username? ").encode("utf-8")
	password = input("Password? ").encode("utf-8")
	filename = input("Filename? ")
	request = ulna.make_ulna_packet(is_request=True, timestamp=timestamp, attrs=[
		("filename", filename.encode("utf-8")),
		("username", username),
		("password", password),
	])
	print("Sending the following ULNA request to the ULNA server:", request)
	token = send_to_ulna_server(request)
	print("Got ULNA token:", token)
	print(f"Downloading {filename} from file server")
	download_file_with_token(filename, token)
