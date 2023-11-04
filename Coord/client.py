import jsonrpclib

coord = jsonrpclib.Server('http://localhost:4000')

# coord.add(site:int, op:str) -> False | int
# coord.remove(site:int, op:str) -> bool
