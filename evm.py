"""An overlay Ethereum virtual machine."""

# Load standard packages
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Union

# Load local packages
from keccak import keccak
from rpc import Node


OPCODES: dict[int, Callable[..., int]] = {}
MOD = 1 << 256

Pc = list[int]


class Result(ABC, BaseException):
    """A prototype for the result of a contract execution."""

    def __repr__(self) -> str:
        return type(self).__name__ + '()'

    @abstractmethod
    def __int__(self) -> int:
        pass

    def data(self) -> bytes:
        """Return associated data."""
        return b''

    def __str__(self) -> str:
        return self.__repr__()


class InsufficientFunds(Result):
    """A funds transfer failed due to insufficient funds."""

    def __int__(self) -> int:
        return 0


class Stop(Result):
    """A contract stopped normally without returning any data."""

    def __int__(self) -> int:
        return 1


class Return(Result):
    """A contract stopped normally and returned some data."""

    _data: bytes

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f'Return({repr(self._data)})'

    def data(self) -> bytes:
        """Return associated data."""
        return self._data

    def __int__(self) -> int:
        return 1


class Revert(Result):
    """A contract encountered an error and the call stack was reverted."""

    _data: bytes

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __repr__(self) -> str:
        return f'Revert({repr(self._data)})'

    def data(self) -> bytes:
        """Return associated data."""
        return self._data

    def __int__(self) -> int:
        return 0


class Storage:
    """An overlay storage for smart contracts."""

    node: Node 
    address: str
    cache: dict[int, int]
    storage: dict[int, int]

    def __init__(self, node: Node, address: str) -> None:
        self.node = node
        self.address = address
        self.cache = {}
        self.storage = {}

    def __getitem__(self, key: int) -> int:
        if key in self.storage:
            return self.storage[key]
        if key not in self.cache:
            self.cache[key] = self.node.eth_getStorageAt(self.address, key)
        return self.cache[key]

    def __setitem__(self, key: int, value: int) -> None:
        self.storage[key] = value

    def clone(self) -> "Storage":
        """Make a clone with own storage."""
        s = Storage.__new__(Storage)
        s.node = self.node
        s.address = self.address
        s.cache = self.cache
        s.storage = self.storage.copy()
        return s


class Balance:
    """A funds balance on a smart contract."""

    def __get__(self, obj: Any, _: Any) -> int:
        return 0 if obj is None else obj._balance

    def __set__(self, obj: Any, value: int) -> None:
        if value < 0:
            raise InsufficientFunds()
        obj._balance = value


@dataclass
class Contract:
    """An Ethereum smart contract."""

    address: int
    code: bytes
    storage: Storage
    balance: Balance = Balance()

    def clone(self) -> "Contract":
        """Make a clone with own storage."""
        return Contract(
            self.address,
            self.code,
            self.storage.clone(),
            self.balance,
        )


class Chain:
    """A overlay storage for all contracts on a blockchain."""

    node: Node
    cache: dict[int, Contract]
    storage: dict[int, Contract]

    def __init__(self, node: Node) -> None:
        self.node = node
        self.cache = {}
        self.storage = {}

    def __getitem__(self, address: int) -> Contract:
        """Get contract code and contract storage."""
        if address not in self.cache:
            code = self.node.eth_getCode(hex(address))
            storage = Storage(self.node, hex(address))
            balance = self.node.eth_getBalance(hex(address))
            contract = Contract(address, code, storage, balance)
            self.cache[address] = contract
        if address not in self.storage:
            self.storage[address] = self.cache[address].clone()
        return self.storage[address]

    def clone(self) -> "Chain":
        """Make a clone with own storage."""
        chain = Chain.__new__(Chain)
        chain.node = self.node
        chain.cache = self.cache
        chain.storage = {a: c.clone() for a,c in self.storage.items()}
        return chain

    def transfer(self, src: int, dst: int, value: int) -> None:
        """Transfer funds between addresses."""
        self[src].balance -= value
        self[dst].balance += value


@dataclass
class Operation:
    """An opcode trace."""

    opcode: int
    name: str
    args: list[int]
    rslt: None | int

    def __repr__(self) -> str:
        args = ', '.join(f'0x{a:x}' for a in self.args)
        rslt = '' if self.rslt is None else f' -> 0x{self.rslt:x}'
        return f'0x{self.opcode:02x} {self.name}({args}){rslt}'


Trace = None | list[Union[Operation, 'Trace']]

@dataclass
class Space:
    """Runtime environment for a smart contract."""

    chain: Chain
    address: int
    code: bytes
    msg: dict
    stack: list[int]
    memory: bytearray  # big endian
    returndata: bytes
    trace: Trace


def sint(x: int) -> int:
    """Convert unsigned integer to signed integer."""
    return x - (x >> 255)*MOD


def uint(x: int) -> int:
    """Convert signed integer to unsigned integer."""
    return x + MOD if x < 0 else x


def cmpl(size: int) -> int:
    """Compute a 256-bit integer with a given number of leading 1s."""
    return (1 << size) - 1 << 256 - size


def memcpy(
        dst: bytearray,
        src: bytes,
        dst_offset: int,
        src_offset: int,
        size: int,
    ) -> None:
    """Copy bytes from src to dst."""
    # Pad short data on the right
    data = src[src_offset:src_offset+size]
    if (n := size - len(data)) > 0:
        data += bytes(n)

    # Expand src if necessary
    if (n := dst_offset + size - len(dst)) > 0:
        dst.extend(bytes(n))

    dst[dst_offset:dst_offset+size] = data


def execute(
        chain: Chain,
        caller: int,
        address: int,
        value: int,
        data: bytes,
        static = False,
        trace = False,
    ) -> tuple[Result, Chain, Trace]:
    """Run an Ethereum contract in a virtual machine."""
    contract = chain[address]
    space = Space(
        chain = chain.clone(),  # make a local clone for state reversal
        address = address,
        code = contract.code,
        msg = {
            'value': value,
            'data': data,
            'caller': caller,
            'static': static,
        },
        stack = [],
        memory = bytearray(),
        returndata = bytes(),
        trace = [] if trace else None
    )
    pc = [0]
    try:
        space.chain.transfer(caller, address, value)
        while True:
            opcode = contract.code[pc[0]]
            pc[0] += 1
            OPCODES[opcode](space, pc)
    except Result as rslt:
        chain = space.chain if int(rslt) == 1 else chain
        return rslt, chain, space.trace


def mkcall(
        chain: Chain,
        caller: int,
        address: int,
        value: int,
        static=False,
        trace=False,
    ) -> Callable[[bytes], tuple[Result, Chain, Trace]]:
    """Prepare a call that accepts arbibrary data."""
    return lambda data: execute(chain, caller, address, value, data, static, trace)


def register(opcode: int) -> Any:
    """Register an opcode handler."""
    def wrapper(func):
        no_args = func.__code__.co_argcount - 2
        name = func.__name__
        def wrapped(s, pc):
            args = [s.stack.pop() for _ in range(no_args)]
            if s.trace is not None:
                trace = Operation(opcode, name, args, None)
                s.trace.append(trace)
            rslt = func(s, pc, *args)
            if rslt is not None:
                s.stack.append(rslt)
            if s.trace is not None:
                trace.rslt = rslt
        OPCODES[opcode] = wrapped
        return wrapped
    return wrapper


@register(0x0)
def stop(s: Space, pc: Pc) -> None:
    raise Stop()


@register(0x1)
def add(s: Space, pc: Pc, a: int, b: int) -> int:
    return (a + b)%MOD


@register(0x2)
def mul(s: Space, pc: Pc, a: int, b: int) -> int:
    return (a*b)%MOD


@register(0x3)
def sub(s: Space, pc: Pc, a: int, b: int) -> int:
    return (a - b)%MOD


@register(0x4)
def div(s: Space, pc: Pc, a: int, b: int) -> int:
    return a//b


@register(0x5)
def sdiv(s: Space, pc: Pc, a: int, b: int) -> int:
    a = sint(a)
    b = sint(b)
    if a >= 0:
        if b >= 0:
            c = a//b
        else:
            c = -(a//(-b))
    else:
        if b >= 0:
            c = -((-a)//b)
        else:
            c = (-a)//(-b)
    return uint(c)


@register(0x6)
def mod(s: Space, pc: Pc, a: int, b: int) -> int:
    return a%b


@register(0x7)
def smod(s: Space, pc: Pc, a: int, b: int) -> int:
    a = sint(a)
    b = sint(b)
    return uint(abs(a)%abs(b)*(1 if a >= 0 else -1))


@register(0x8)
def addmod(s: Space, pc: Pc, a: int, b: int, N: int) -> int:
    return (a + b)%N


@register(0x9)
def mulmod(s: Space, pc: Pc, a: int, b: int, N: int) -> int:
    return (a*b)%N


@register(0xa)
def exp(s: Space, pc: Pc, a: int, b: int) -> int:
    return pow(a, b, MOD)


@register(0xb)
def signextend(s: Space, pc: Pc, b: int, x: int) -> int:
    size = 8*(b + 1)
    x &= (1 << size) - 1
    if x >> size - 1 == 0:
        return x
    else:
        return x | cmpl(256 - size)


@register(0x10)
def lt(s: Space, pc: Pc, a: int, b: int) -> int:
    return int(a < b)


@register(0x11)
def gt(s: Space, pc: Pc, a: int, b: int) -> int:
    return int(a > b)


@register(0x12)
def slt(s: Space, pc: Pc, a: int, b: int) -> int:
    return int(sint(a) < sint(b))


@register(0x13)
def sgt(s: Space, pc: Pc, a: int, b: int) -> int:
    return int(sint(a) > sint(b))


@register(0x14)
def equal(s: Space, pc: Pc, a: int, b: int) -> int:
    return int(a == b)


@register(0x15)
def iszero(s: Space, pc: Pc, a: int) -> int:
    return int(a == 0)


@register(0x16)
def bitwise_and(s: Space, pc: Pc, a: int, b: int) -> int:
    return a & b


@register(0x17)
def bitwise_or(s: Space, pc: Pc, a: int, b: int) -> int:
    return a | b


@register(0x18)
def bitwise_xor(s: Space, pc: Pc, a: int, b: int) -> int:
    return a ^ b


@register(0x19)
def bitwise_not(s: Space, pc: Pc, a: int) -> int:
    return a ^ (MOD - 1)


@register(0x20)
def sha3(s: Space, pc: Pc, offset: int, length: int) -> int:
    hash = keccak(s.memory[offset:offset+length])
    return int.from_bytes(hash, 'big')


@register(0x1b)
def shl(s: Space, pc: Pc, shift: int, value: int) -> int:
    return (value << shift)%MOD


@register(0x1c)
def shr(s: Space, pc: Pc, shift: int, value: int) -> int:
    return value >> shift


@register(0x1d)
def sar(s: Space, pc: Pc, shift: int, value: int) -> int:
    if value >> 255 == 0:
        return value >> shift
    else:
        return value >> shift | cmpl(shift)


@register(0x30)
def address(s: Space, pc: Pc) -> int:
    return s.address


@register(0x31)
def balance(s: Space, pc: Pc, address: int) -> int:
    return s.chain[address].balance


@register(0x33)
def caller(s: Space, pc: Pc) -> int:
    return s.msg['caller']


@register(0x34)
def callvalue(s: Space, pc: Pc) -> int:
    return s.msg['value']


@register(0x35)
def calldataload(s: Space, pc: Pc, offset: int) -> int:
    word = s.msg['data'][offset:offset+32]
    word += bytes(32 - len(word))  # Short data are padded on the right
    return int.from_bytes(word, 'big')


@register(0x36)
def calldatasize(s: Space, pc: Pc) -> int:
    return len(s.msg['data'])


@register(0x37)
def calldatacopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    memcpy(s.memory, s.msg['data'], dest_offset, offset, length)


@register(0x39)
def codecopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    memcpy(s.memory, s.code, dest_offset, offset, length)


@register(0x3b)
def extcodesize(s: Space, pc: Pc, addr: int) -> int:
    contract = s.chain[addr]
    return len(contract.code)


@register(0x3d)
def returndatasize(s: Space, pc: Pc) -> int:
    return len(s.returndata)


@register(0x3e)
def returndatacopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    memcpy(s.memory, s.returndata, dest_offset, offset, length)


@register(0x42)
def timestamp(s: Space, pc: Pc) -> int:
    return s.chain.node.block_timestamp()


@register(0x47)
def selfbalance(s: Space, pc: Pc) -> int:
    return s.chain[s.address].balance


@register(0x50)
def pop(s: Space, pc: Pc, _: int) -> None:
    pass


@register(0x51)
def mload(s: Space, pc: Pc, offset: int) -> int:
    word = s.memory[offset:offset+32]
    return int.from_bytes(word, 'big')


@register(0x52)
def mstore(s: Space, pc: Pc, offset: int, value: int) -> None:
    if (n := offset + 32 - len(s.memory)) > 0:
        s.memory.extend(bytearray(n))
    s.memory[offset:offset+32] = value.to_bytes(32, 'big')


@register(0x54)
def sload(s: Space, pc: Pc, key: int) -> int:
    storage = s.chain[s.address].storage
    return storage[key]


@register(0x55)
def sstore(s: Space, pc: Pc, key: int, value: int) -> None:
    if s.msg['static']:
        raise RuntimeError('Non-static method called from a static call')
    storage = s.chain[s.address].storage
    storage[key] = value


@register(0x56)
def jump(s: Space, pc: Pc, dest: int) -> None:
    pc[0] = dest


@register(0x57)
def jumpi(s: Space, pc: Pc, dest: int, cond: int) -> None:
    if cond != 0:
        pc[0] = dest


@register(0x5a)
def gas(s: Space, pc: Pc) -> int:
    return MOD - 1  # Dummy number (max u256)


@register(0x5b)
def jumpdest(s: Space, pc: Pc) -> None:
    pass


def push(k: int) -> Any:
    def push_k(s: Space, pc: Pc) -> None:
        value = int.from_bytes(s.code[pc[0]:pc[0]+k], 'big')
        if s.trace is not None:
            s.trace.append(Operation(0x5f + k, f'push{k}', [], value))
        s.stack.append(value)
        pc[0] += k
    return push_k

# Register all push functions
# 0x60 - 0x7F
for i in range(32):
    OPCODES[0x60+i] = push(i + 1)


def dup(k: int) -> Any:
    def dup_k(s: Space, pc: Pc) -> None:
        value = s.stack[-k]
        if s.trace is not None:
            s.trace.append(Operation(0x7f + k, f'dup{k}', [], value))
        s.stack.append(value)
    return dup_k

# Register all dup functions
# 0x80 - 0x8F
for i in range(16):
    OPCODES[0x80+i] = dup(i + 1)


def swap(k: int) -> Any:
    def swap_k(s: Space, pc: Pc) -> None:
        if s.trace is not None:
            s.trace.append(Operation(0x8f + k, f'swap{k}', [], None))
        s.stack[-1], s.stack[-k-1] = s.stack[-k-1], s.stack[-1]
    return swap_k

# Register all swap functions
# 0x90 - 0x9F
for i in range(16):
    OPCODES[0x90+i] = swap(i + 1)


@register(0xa0)
def log0(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
    ) -> None:
    # TODO: event handling
    pass


@register(0xa1)
def log1(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
    ) -> None:
    # TODO: event handling
    pass


@register(0xa2)
def log2(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
        topic1: int,
    ) -> None:
    # TODO: event handling
    pass


@register(0xa3)
def log3(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
        topic1: int,
        topic2: int,
    ) -> None:
    # TODO: event handling
    pass


@register(0xa4)
def log4(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
        topic1: int,
        topic2: int,
        topic3: int,
    ) -> None:
    # TODO: event handling
    pass


def generic_call(
        static: bool,
        s: Space,
        pc: Pc,
        gas: int,
        addr: int,
        value: int,
        args_offset: int,
        args_length: int,
        ret_offset: int,
        ret_length: int,
    ) -> int:
    data = s.memory[args_offset:args_offset+args_length]
    result, chain, trace = execute(
        s.chain, s.address, addr, value, data, static, s.trace is not None,
    )
    memcpy(s.memory, result.data(), ret_offset, 0, ret_length)
    s.returndata = result.data()
    s.chain = chain
    if s.trace is not None:
        s.trace.append(trace)
    return int(result)


@register(0xf1)
def call(
        s: Space,
        pc: Pc,
        gas: int,
        addr: int,
        value: int,
        args_offset: int,
        args_length: int,
        ret_offset: int,
        ret_length: int,
    ) -> int:
    if s.msg['static'] and value != 0:
        raise RuntimeError('Non-static call from a static call')
    return generic_call(
        s.msg['static'], s, pc,
        gas, addr, value, args_offset, args_length, ret_offset, ret_length,
    )


@register(0xf3)
def op_return(s: Space, pc: Pc, offset: int, length: int) -> None:
    raise Return(s.memory[offset:offset+length])


@register(0xfa)
def staticcall(
        s: Space,
        pc: Pc,
        gas: int,
        addr: int,
        args_offset: int,
        args_length: int,
        ret_offset: int,
        ret_length: int,
    ) -> int:
    return generic_call(
        True, s, pc,
        gas, addr, 0, args_offset, args_length, ret_offset, ret_length,
    )


@register(0xfd)
def revert(s: Space, pc: Pc, offset: int, length: int) -> None:
    raise Revert(s.memory[offset:offset+length])


def save_trace(trace: Trace, path: str) -> None:
    """Write a trace to a file."""
    def write(trace, indent):
        for op in trace:
            if type(op) == list:
                write(op, indent + 1)
            else:
                f.write('  '*indent)
                f.write(op.__repr__())
                f.write('\n')

    with open(path, 'w') as f:
        write(trace, 0)
