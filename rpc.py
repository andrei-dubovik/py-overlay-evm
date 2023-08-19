"""Python wrappers around Ethereum JSON-RPC."""

# Load standard packages
from typing import Any
import binascii
import json

# Load external packages
import requests

class Node:
    """A limited wrapper around the Ethereum JSON-RPC.

    This wrapper memorizes the current block number on initialization, and all
    subsequent queries are issued for that block number. This behaviour ensures
    data cosnistency across queries. However, with public nodes, all the
    queries need to be done within a short time interval, becuase public nodes
    allow queries to a past block number for a short time only.
    """

    node: str
    block: str
    timestamp: str
    query_id: int
    verbose: bool

    def __init__(self, node: str, block: int = -1, verbose=False) -> None:
        """Initialize an interface to a Ethereum node, freeze block number."""
        self.node = node
        self.verbose = verbose
        self.query_id = 0
        xblock = 'latest' if block == -1 else hex(block)
        block_data = self.query('eth_getBlockByNumber', [xblock, False])
        self.block = block_data['number']
        self.timestamp = block_data['timestamp']

    def query(self, method: str, params: list[str|bool]) -> Any:
        """Run an arbitrary query, deserialize the result."""
        if self.verbose:
            print(f"Fetching {method}({params})")
        self.query_id += 1
        payload = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
            'id': 5,
        }
        response = requests.post(self.node, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f'http query failed with status code {response.status_code}')
        content = json.loads(response.content)
        if 'error' in content:
            raise RuntimeError(f'JSON-RPC failed with error: {content["error"]["message"]}')
        return content['result']

    def eth_blockNumber(self) -> int:
        """Get the frozen block number."""
        return int(self.block, 16)

    def block_timestamp(self) -> int:
        """Get the timestamp associated with the frozen block number."""
        return int(self.timestamp, 16)

    def eth_getBalance(self, address: int) -> int:
        """Get the balance at the given address."""
        xaddr = f'0x{address:040x}'
        balance = self.query('eth_getBalance', [xaddr, self.block])
        return int(balance, 16)

    def eth_getCode(self, address: int) -> bytes:
        """Get the contract code at the given address."""
        xaddr = f'0x{address:040x}'
        code = self.query('eth_getCode', [xaddr, self.block])
        return binascii.a2b_hex(code[2:])

    def eth_getStorageAt(self, address: int, position: int) -> int:
        """Get contract storage at given address and position."""
        xaddr = f'0x{address:040x}'
        xpos = hex(position)
        value = self.query('eth_getStorageAt', [xaddr, xpos, self.block])
        return int(value, 16)
