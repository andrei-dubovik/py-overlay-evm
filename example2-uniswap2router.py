from evm import Chain, mkcall, save_trace
from rpc import Node
from solidity import solidity, address
from solidity import uint32, uint112, uint256


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


# Example
if __name__ == '__main__':
    # Define addresses

    ## Tokens
    wbnb = int("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", 16)
    busd = int("0xe9e7cea3dedca5984780bafc599bd69add087d56", 16)

    ## Pancake Uniswap V2
    uniswap = int("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16", 16)
    router = int("0x10ed43c718714eb63d5aa57b78b54704e256024e", 16)

    ## Mock holder address
    holder = int("0xabababababababababababababababababababab", 16)

    # Define amounts
    amount = 100_000000000000000000

    # Initialize a link to a public node
    url = "https://bsc-dataseed1.binance.org/"
    chain = Chain(Node(url, verbose=True))

    # Deposit 100 BNB to a mock address
    _, ch2, _ = deposit(
        mkcall(chain, holder, wbnb, amount),
    )

    # Check that the deposit operation was successful
    x, _, _ = balanceOf(
        mkcall(ch2, holder, wbnb, 0),
        address = holder,
    )
    print(f'WBNB balance: {x/10**18}')

    # Get reserves
    (reserve0, reserve1, timestamp), _, _ = getReserves(
        mkcall(ch2, holder, uniswap, 0)
    )

    # Estimate amount to be received
    x, _, _ = getAmountOut(
        mkcall(ch2, holder, router, 0),
        amountIn = amount,
        reserveIn = reserve0,
        reserveOut = reserve1,
    )
    print(f'BUSD amount out: {x/10**18}')

    # Approve withdrawl of 100 WBNB
    _, ch3, _ = approve(
        mkcall(ch2, holder, wbnb, 0),
        guy = router,
        wad = amount,
    )

    # Exchange 100 WBNB to BUSD; save the trace
    _, ch4, trace = swapExactTokensForTokens(
        mkcall(ch3, holder, router, 0, trace=True),
        amountIn = amount,
        amountOutMin = 0,
        path = [wbnb, busd],
        to = holder,
        deadline = timestamp + 60,
    )
    save_trace(trace, 'swap.trace')

    # Check final WBNB balance
    x, _, _ = balanceOf(
        mkcall(ch4, holder, wbnb, 0),
        address = holder,
    )
    print(f'WBNB balance: {x/10**18}')
    
    # Check final BUSD balance
    x, _, _ = balanceOf(
        mkcall(ch4, holder, busd, 0),
        address = holder,
    )
    print(f'BUSD balance: {x/10**18}')
