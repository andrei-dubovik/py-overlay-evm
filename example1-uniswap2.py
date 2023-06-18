from evm import Chain, mkcall, save_trace
from solidity import solidity, address, string, uint256
from rpc import Node


@solidity
def name() -> string:
    pass

@solidity
def balanceOf(address: address) -> uint256:
    pass

@solidity
def transfer(dst: address, wad: uint256) -> bool:
    pass

@solidity
def swap(amount0Out: uint256, amount1Out: uint256, to: address, data: bytes):
    pass


# Sandbox
if __name__ == '__main__':
    # Addresses
    uniswap = int("0x58F876857a02D6762E0101bb5C46A8c1ED44Dc16", 16)
    holder = int("0xd7d069493685a581d27824fc46eda46b7efc0063", 16)
    wbnb = int("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", 16)
    busd = int("0xe9e7cea3dedca5984780bafc599bd69add087d56", 16)
    receiver = int("0xabababababababababababababababababababab", 16)

    # Initialize a link to a public node
    url = "https://bsc-dataseed1.binance.org/"
    chain = Chain(Node(url, verbose=True))

    # Contract name
    x, _, trace = name(mkcall(chain, holder, uniswap, 0, trace=True))
    print(x)

    # What's my initial balance?
    x, _, trace = balanceOf(mkcall(chain, holder, wbnb, 0, trace=True), holder)
    print(x)

    # Send WBNB to Uniswap V2
    x, ch2, trace = transfer(mkcall(chain, holder, wbnb, 0, trace=True), uniswap, 100_000000000000000000)

    # Swap
    x, ch3, trace = swap(mkcall(ch2, holder, uniswap, 0, trace=True), 0, 20_000_000000000000000000, receiver, b'')
    save_trace(trace, 'swap.trace')

    # Did it work? It did!
    x, _, trace = balanceOf(mkcall(ch3, holder, busd, 0, trace=True), receiver)
    print(x)
