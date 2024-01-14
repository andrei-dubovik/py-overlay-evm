# py-overlay-evm

Ethereum VM written in Python with minimal external dependencies and aimed at a mock execution of smart contracts.

## Rationale

This project grew from what was essentially an exercise to understand how smart contracts work on the Ethereum blockchain and its clones. One way to gain understanding is to program an EVM from scratch, without referencing existing code, and that is precisely what I did. The codebase includes an implementation for the Ethereum bytecodes, including the corresponding gas calculations, own Keccak-256 implementation, as well as a Solidity wrapper for conveniently calling functions in Solidity contracts.

As of now, the implementation of the EVM is incomplete, and the API is still likely to change. I reckon the code is good enough to play around with, but it is certainly not ready for production (for instance, there is zero test coverage). The examples below demonstrate what can be done now.

## Features
- *Minimal external dependencies:* numpy, requests.
- *Functional style state changes:* each call to a smart contract returns a new view of the blockchain with the respective changes implemented, the initial view from which the call started remains available and unchanged.
- *Overlay architecture:* data is read from the (public) node if these data have not been previously accessed, further reads are cached, and any changes are kept in an overlay in memory.
- *Gas calculations:* gas usage is calculated exactly (work in progress).

## Examples
The following examples all use the Ethereum blockchain.

### Example I (manual contract calls)

```python
from py_overlay_evm.rpc import Node
from py_overlay_evm.evm import Chain, execute
from py_overlay_evm.keccak import keccak

# Initialize a link to a public node
# 
# During initialization, the current block number is noted, and all the
# subsequent requests to the chain are done for the same block number. This
# behaviour ensures consistency but it also means that all the subsequent
# requests must be done within a short time interval, because public nodes do
# not return data for somewhat older blocks.
#
# For testing purposes, we can use the public nodes from the Flashbots project.
url = "https://rpc.flashbots.net"
chain = Chain(Node(url, verbose=True))

# We consider a Wrapped Ether contract for this example
#
# Addresses are always encoded as integers.
weth = int("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 16)

# We will be calling the contract from a mock address
caller = int("0xabababababababababababababababababababab", 16)

# We are going to call the `name()` method on the contract. The method accepts
# zero arguments. According to the Solidity ABI, we need to compute the Keccak
# signature of "name()", take its first 4 bytes and pass it as the byte input
# to the contract.
data = keccak(b'name()')[:4]
rslt = execute(
    chain = chain,
    caller = caller,
    address = weth,
    value = 0,
    data = data,
    trace = True,
)

# `rslt.data` contains the string "Wrapped Ether" encoded according to the
# Solidity ABI
print(rslt.data)

# prints an equivalent of:
# bytearray(b'
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x20
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0d
# Wrapped Ether\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# ')

# `rslt.chain` contains the new state of the blockchain after a successful
# execution of a contract. In this case, given that the method `name()` is
# read-only, `rslt.chain` will be equivalent to `chain`.

# `rslt.gas` reports the exact gas used by the transaction. Please note that
# gas calculations have not been extensively tested and could well be
# erroneous.
print(rslt.gas)  # prints "24174"

# Finally, `rslt.trace` contains the trace of the contract execution.
print(len(rslt.trace))  # prints "228"

for op in rslt.trace:
    print(op)

# prints:
# 0x60 push1() -> 0x60, gas: 3
# 0x60 push1() -> 0x40, gas: 3
# 0x52 mstore(0x40, 0x60), gas: 12
# 0x60 push1() -> 0x4, gas: 3
# 0x36 calldatasize() -> 0x4, gas: 2
# ...
# 0x80 dup1() -> 0xa0, gas: 3
# 0x91 swap2(), gas: 3
# 0x03 sub(0x100, 0xa0) -> 0x60, gas: 3
# 0x90 swap1(), gas: 3
# 0xf3 op_return(0xa0, 0x60), gas: 0

# In this case there are no external calls to other contracts, but if there are
# such calls, their traces also get recorded.
```

### Example II (Solidity wrapper)

For conveniently calling Solidity contracts, there is a wrapper that handles the encoding of the call signature as well as the translation of the data from Python types to Solidity binary format and back. Here, we repeat the first example but using the Solidity wrapper.

```python
from py_overlay_evm.rpc import Node
from py_overlay_evm.evm import Chain, execute, mkcall
from py_overlay_evm.solidity import solidity, string

# Initialize a link to a public node
url = "https://rpc.flashbots.net"
chain = Chain(Node(url, verbose=True))

# We consider a Wrapped Ether contract for this example
weth = int("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 16)

# We will be calling the contract from a mock address
caller = int("0xabababababababababababababababababababab", 16)

# To use the Solidity wrapper, we simply define a Python function with the same
# name as the Solidity function and we type annotate it using Solidity type
# names. The body of the function does not matter, so we'll just use `pass`.
@solidity
def name() -> string:
    pass

# A wrapped function accepts a partial call to `execute()` as the first
# parameter, and the remaining parameters are simply the parameters for the
# Solidity function, in this case none. 
rslt = name(
    lambda data: execute(chain, caller, weth, 0, data, trace=True),
)
print(rslt.value)  # prints "Wrapped Ether"

# To simplify making a partial call to `execute()` there is also `mkcall()`:
rslt = name(
    mkcall(chain, caller, weth, 0, trace=True),
)
```

### Example III (Uniswap V3 router)

This is a longer example that walks through a non-trivial operation, namely depositing mock WETH coins to a mock account, and then changing them into USDT coins via a Uniswap V3 router. This example also demonstrates more extensive usage of the Solidity wrapper.

```python
from dataclasses import dataclass

from py_overlay_evm.evm import Chain, mkcall, save_trace
from py_overlay_evm.rpc import Node
from py_overlay_evm.solidity import solidity, address
from py_overlay_evm.solidity import uint24, uint160, uint256


## Firstly, we define all the Solidity functions that we will need

@solidity
def deposit():
    pass

@solidity
def balanceOf(address: address) -> uint256:
    pass

@solidity
def approve(
    guy: address,
    wad: uint256,
) -> None:
    pass

# A dataclass can be used if a Solidity function wants a structure as a
# parameter. The dataclass can have an arbitrary name, because the name does
# not form a part of the function's signature.
@dataclass
class SwapParams:
    tokenIn:           address
    tokenOut:          address
    fee:               uint24
    recipient:         address
    deadline:          uint256
    amountIn:          uint256
    amountOutMinimum:  uint256
    sqrtPriceLimitX96: uint160

@solidity
def exactInputSingle(params: SwapParams) -> uint256:
    pass


## Secondly, we define all the required addresses

# Tokens
weth = int("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 16)
usdt = int("0xdac17f958d2ee523a2206206994597c13d831ec7", 16)

# Uniswap V2 router
router = int("0xe592427a0aece92de3edee1f18e0157c05861564", 16)

# Mock holder address
holder = int("0xabababababababababababababababababababab", 16)

## Finally, we prepare mock WETH and change them into USDT

# Initialize a link to a public node
url = "https://rpc.flashbots.net"
chain = Chain(Node(url, verbose=True))

# Create an unlimited supply of ETH at the mock address :3
chain[holder].balance = (1 << 256) - 1

# Deposit 100 ETH to WETH
amount = 100*10**18
rslt1 = deposit(
    mkcall(chain, holder, weth, amount),
)

# Check that the deposit operation was successful
x = balanceOf(
    mkcall(rslt1.chain, holder, weth, 0),
    address = holder,
).value
print(f'WETH balance: {x/10**18}')

# Approve withdrawal of 100 WETH
rslt2 = approve(
    mkcall(rslt1.chain, holder, weth, 0),
    guy = router,
    wad = amount,
)

# We use the block's timestamp plus 60 seconds as a deadline
deadline = chain.node.block_timestamp() + 60

# Exchange 100 WETH to USDT
rslt3 = exactInputSingle(
    mkcall(rslt2.chain, holder, router, 0, trace=True),
    params = SwapParams(
        tokenIn           = weth,
        tokenOut          = usdt,
        fee               = 500,  # 0.05%
        recipient         = holder,
        deadline          = deadline,
        amountIn          = amount,
        amountOutMinimum  = 0,
        sqrtPriceLimitX96 = 0,
    ),
)
print(f'USDT amount out: {rslt3.value/10**6:.2f}')
# prints "USDT amount out: 256474.98" at the time or writing

# The trace is large (8,402 operations at the time of writing, but that will
# differ depending on how many ticks get traversed), so we save it to a file for
# viewing in an external editor
save_trace(rslt3.trace, 'swap.trace')

# Check final WETH balance
x = balanceOf(
    mkcall(rslt3.chain, holder, weth, 0),
    address = holder,
).value
print(f'WETH balance: {x/10**18:.2f}')  # prints "0.00"

# Check final USDT balance
x = balanceOf(
    mkcall(rslt3.chain, holder, usdt, 0),
    address = holder,
).value
print(f'USDT balance: {x/10**6:.2f}')
# prints "USDT balance: 256474.98" at the time or writing
```

## Roadmap

In case I happen to have time to further work on this project, the overall roadmap is as follows:
- [x] Gas calculations
- [x] Python package
- [ ] Full coverage of EVM bytecodes
- [ ] Tests for EVM bytecodes
- [ ] Full coverage of Solidity datatypes
- [ ] Tests for Solidity datatypes
- [ ] Documentation
