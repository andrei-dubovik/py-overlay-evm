# py-overlay-evm

Ethereum VM written in Python with minimal external dependencies and aimed at a mock execution of smart contracts.

## Rationale

This project grew from what was essentially an exercise to understand how smart contracts work on the Ethereum blockchain and its clowns. One way to gain understanding is to program an EVM from scratch, without referencing existing code, and that is precisely what I did. The codebase includes an implementation for the Ethereum bytecodes, own Keccak-256 implementation, as well as a Solidity wrapper for conveniently calling functions in Solidity contracts.

As of now, the implementation of the EVM is incomplete, and the API is still likely to change. I reckon the code is good enough to play around, but it is certainly not ready for production. (For one, there is zero test coverage.) See the examples below for what can be done now.

## Features
- *Minimal external dependencies:* numpy, requests.
- *Functional style state changes:* each call to a smart contract returns a new view of the blockchain with the respective changes implemented, the initial view from which the call started remains available and unchanged.
- *Overlay architecture:* data is read from the (public) node if these data has not been previously accessed, further reads are cached, and any changes are kept in an overlay in memory.

## Examples
The following examples all use the Binance chain.

### Example I (manual contract calls)

```python
from rpc import Node
from evm import Chain, execute
from keccak import keccak

# Initialize a link to a public node
# 
# During initialization, the current block number is noted, and all the
# subsequent requests to the chain are done for the same block number. This
# behaviour ensures consistency but it also means that all the subsequent
# requests must be done within a short time interval, because public nodes do
# not return data for somewhat older blocks.
url = "https://bsc-dataseed1.binance.org/"
chain = Chain(Node(url, verbose=True))

# We consider a Wrapped BNB contract for this example
#
# Addresses are always encoded as integers.
wbnb = int("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", 16)

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
    address = wbnb,
    value = 0,
    data = data,
    trace = True,
)

# `rslt.data` contains the string "Wrapped BNB" encoded according to the
# Solidity ABI
print(rslt.data)

# prints:
# bytearray(b'
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x20
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b
# Wrapped BNB\x00\x00\x00\x00\x00
# \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
# ')

# `rslt.chain` contains the new state of the blockchain after a successful
# execution of a contract. In this case, given that the method `name()` is
# read-only, `rslt.chain` will be equivalent to `chain`.

# Finally, `rslt.trace` contains the trace of the contract execution.
print(len(rslt.trace))  # prints "228"

for op in rslt.trace:
    print(op)

# prints:
# 0x60 push1() -> 0x60
# 0x60 push1() -> 0x40
# 0x52 mstore(0x40, 0x60)
# 0x60 push1() -> 0x4
# 0x36 calldatasize() -> 0x4
# ...
# 0x80 dup1() -> 0xa0
# 0x91 swap2()
# 0x03 sub(0x100, 0xa0) -> 0x60
# 0x90 swap1()
# 0xf3 op_return(0xa0, 0x60)

# In this case there are no external calls to other contracts, but if there are
# such calls, their traces also get recorded.
```

### Example II (Solidity wrapper)

For conveniently calling Solidity contracts, there is a wrapper that handles the encoding of the call signature as well as the translation of the data from Python types to Solidity binary format and back. Here, we repeat the first example but using the Solidity wrapper.

```python
from rpc import Node
from evm import Chain, execute, mkcall
from solidity import solidity, string

# Initialize a link to a public node
url = "https://bsc-dataseed1.binance.org/"
chain = Chain(Node(url, verbose=True))

# We consider a Wrapped BNB contract for this example
wbnb = int("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", 16)

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
    lambda data: execute(chain, caller, wbnb, 0, data, trace=True),
)
print(rslt.value)  # prints "Wrapped BNB"

# To simplify making a partiall call to `execute()` there is also `mkcall()`:
rslt = name(
    mkcall(chain, caller, wbnb, 0, trace=True),
)
```

### Example III (Uniswap V2 router)

This is a longer example that walks through a non-trivial operation, namely depositing mock WBNB coins to a mock account, and then changing them into BUSD coins via a Uniswap V2 router. This example also demonstrates more extensive usage of the Solidity wrapper.

```python
from evm import Chain, mkcall, save_trace
from rpc import Node
from solidity import solidity, address
from solidity import uint32, uint112, uint256


# Firstly, we define all the Solidity function that we will need

@solidity
def deposit():
    pass


@solidity
def balanceOf(address: address) -> uint256:
    pass


@solidity
def getReserves() -> tuple[uint112, uint112, uint32]:
    pass


@solidity
def getAmountOut(
    amountIn: uint256,
    reserveIn: uint256,
    reserveOut: uint256,
) -> uint256:
    pass


@solidity
def approve(
    guy: address,
    wad: uint256,
) -> None:
    pass


@solidity
def swapExactTokensForTokens(
    amountIn: uint256,
    amountOutMin: uint256,
    path: list[address],
    to: address,
    deadline: uint256,
) -> None:
    pass


# Secondly, we define all the required addresses

## Tokens
wbnb = int("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", 16)
busd = int("0xe9e7cea3dedca5984780bafc599bd69add087d56", 16)

## Pancake Uniswap V2
uniswap = int("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16", 16)
router = int("0x10ed43c718714eb63d5aa57b78b54704e256024e", 16)

## Mock holder address
holder = int("0xabababababababababababababababababababab", 16)

# Finally, we prepare mock WBNB and change them into BUSD

## Initialize a link to a public node
url = "https://bsc-dataseed1.binance.org/"
chain = Chain(Node(url, verbose=True))

## Create an unlimited supply of BNB at the mock address
chain[holder].balance = (1 << 256) - 1

## Deposit 100 BNB to WBNB
amount = 100_000000000000000000
rslt1 = deposit(
    mkcall(chain, holder, wbnb, amount),
)

## Check that the deposit operation was successful
x = balanceOf(
    mkcall(rslt1.chain, holder, wbnb, 0),
    address = holder,
).value
print(f'WBNB balance: {x/10**18}')

## Get reserves
reserve0, reserve1, timestamp = getReserves(
    mkcall(rslt1.chain, holder, uniswap, 0)
).value

## Estimate amount to be received
x = getAmountOut(
    mkcall(rslt1.chain, holder, router, 0),
    amountIn = amount,
    reserveIn = reserve0,
    reserveOut = reserve1,
).value
print(f'BUSD amount out: {x/10**18}')

## Approve withdrawl of 100 WBNB
rslt2 = approve(
    mkcall(rslt1.chain, holder, wbnb, 0),
    guy = router,
    wad = amount,
)

## Exchange 100 WBNB to BUSD
rslt3 = swapExactTokensForTokens(
    mkcall(rslt2.chain, holder, router, 0, trace=True),
    amountIn = amount,
    amountOutMin = 0,
    path = [wbnb, busd],
    to = holder,
    deadline = timestamp + 60,
)

## The trace is large (4,974 operations) so we save it to a file for viewing in
## an external editor
save_trace(rslt3.trace, 'swap.trace')

## Check final WBNB balance
x = balanceOf(
    mkcall(rslt3.chain, holder, wbnb, 0),
    address = holder,
).value
print(f'WBNB balance: {x/10**18}')

## Check final BUSD balance
x = balanceOf(
    mkcall(rslt3.chain, holder, busd, 0),
    address = holder,
).value
print(f'BUSD balance: {x/10**18}')
```

## Roadmap

In case I happen to have time to further work on this project, the overall roadmap is as follows:
- [ ] Gas calculations
- [ ] Full coverage of EVM bytecodes (possibly using stab implementations)
- [ ] Tests for EVM bytecodes
- [ ] Full coverage of Solidity datatypes
- [ ] Tests for Solidity datatypes
- [ ] Python package
- [ ] Documentation
