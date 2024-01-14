# Copyright (c) 2023-2024, Andrey Dubovik <andrei@dubovik.eu>

"""An overlay Ethereum virtual machine."""

# Load standard packages
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Union

# Load local packages
from .keccak import keccak
from .rpc import Node


OPCODES: dict[int, Callable[..., int]] = {}
MOD = 1 << 256

Pc = list[int]


@dataclass
class Signal(ABC, BaseException):
    """A prototype for the base result of a contract call."""

    data: bytes = b''

    @abstractmethod
    def __int__(self) -> int:
        pass


class InsufficientFunds(Signal):
    """A funds transfer failed due to insufficient funds."""

    def __int__(self) -> int:
        return 0


class InsufficientGas(Signal):
    """An execution failed due to exhausted gas."""

    def __int__(self) -> int:
        return 0


class Stop(Signal):
    """A contract stopped normally without returning any data."""

    def __int__(self) -> int:
        return 1


class Return(Signal):
    """A contract stopped normally and returned some data."""

    def __int__(self) -> int:
        return 1


class Revert(Signal):
    """A contract encountered an error and the call stack was reverted."""

    def __int__(self) -> int:
        return 0


class Storage:
    """An overlay storage for smart contracts."""

    node: Node 
    address: int
    cache: dict[int, int]
    storage: dict[int, int]

    def __init__(self, node: Node, address: int) -> None:
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


class FundsBalance:
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
    balance: FundsBalance = FundsBalance()

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
            code = self.node.eth_getCode(address)
            storage = Storage(self.node, address)
            balance = self.node.eth_getBalance(address)
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
    rslt: None | int = None
    gas: int = 0

    def __repr__(self) -> str:
        args = ', '.join(f'0x{a:x}' for a in self.args)
        rslt = '' if self.rslt is None else f' -> 0x{self.rslt:x}'
        gas = f', gas: {self.gas}'
        return f'0x{self.opcode:02x} {self.name}({args}){rslt}{gas}'


Trace = None | list[Union[Operation, 'Trace']]


class GasBalance:
    """An ongoing gas balance."""

    def __get__(self, obj: Any, _: Any) -> int:
        return 0 if obj is None else obj._gas

    def __set__(self, obj: Any, value: int) -> None:
        obj._gas = value
        if value < 0:
            raise InsufficientGas()


def memory_cost(size: int) -> int:
    """Compute the gas costs for a given memory usage."""
    words = (size + 31)//32
    return words**2//512 + 3*words


class Memory:
    """An interface to manage the runtime memory of a smart contract."""

    def __init__(self, space: 'Space') -> None:
        """Allocate an interface and link it to a specific runtime."""
        self.space = space

    def __getitem__(self, key: slice) -> bytes:
        """Get memory for a given address range."""
        key = self.specialize(key)
        if key.start == key.stop:
            return b''
        self.resize(key.stop)
        return self.space._memory[key]

    def __setitem__(self, key: slice, value: bytes) -> None:
        """Set memory at a given address range."""
        key = self.specialize(key)
        size = key.stop - key.start
        if size == 0:
            return
        if size < len(value):
            raise RuntimeError('Data exceeds the destination memory range')
        if (n := size - len(value)) > 0:
            # Pad short data on the right
            value = value + bytes(n)
        self.resize(key.stop)
        self.space._memory[key] = value

    def specialize(self, key: slice) -> slice:
        """Specialize a partial slice."""
        key = slice(
            key.start or 0,
            key.stop or len(self.space._memory),
            key.step or 1,
        )
        # Flexible slices are not needed and are likely an error indication
        if key.step != 1:
            raise RuntimeError('Non-contiguous slices are not supported')
        if key.start < 0 or key.stop < 0:
            raise RuntimeError('Negative slice bounds are not supported')
        if key.start > key.stop:
            raise RuntimeError('Right-to-left slices are not supported')
        return key

    def resize(self, size: int) -> None:
        """Resize the underlying buffer when necessary, do gas accounting."""
        size0 = len(self.space._memory)
        if size > size0:
            self.space.gas -= memory_cost(size) - memory_cost(size0)
            self.space._memory.extend(bytearray(size - size0))


# AccessSet contains either
# - contract address: 0 or
# - (contract address, storage key): initial value
# records.
AccessSet = dict[int | tuple[int, int], int]


@dataclass
class Space:
    """Runtime environment for a smart contract."""

    chain: Chain
    address: int
    code: bytes
    msg: dict
    stack: list[int]
    returndata: bytes
    trace: Trace
    access_set: AccessSet
    gas: GasBalance = GasBalance()

    _memory: bytearray = field(default_factory=bytearray)  # big endian

    @property
    def memory(self) -> Memory:
        return Memory(self)

    def get_contract(self, address: int) -> Contract:
        if address in self.access_set:
            self.gas -= 100
        else:
            self.gas -= 2600
            self.access_set[address] = 0
        return self.chain[address]


@dataclass
class CallResult(BaseException):
    """A full result of a contract call, including the new chain state."""

    signal: Signal
    chain: Chain
    access_set: AccessSet
    gas: int
    trace: Trace

    @property
    def data(self) -> bytes:
        return self.signal.data

    def __int__(self) -> int:
        return int(self.signal)

    def __repr__(self) -> str:
        """Pretty-print the object for interactive debugging."""
        return 'CallResult(%s, %s, %s, %s, %s, %s)' % (
            'signal=%s(...)' % type(self.signal).__name__,
            'data=%s' % repr(self.data),
            'chain=...',
            'access_set=...',
            'gas=%i' % self.gas,
            'trace=%s' % ('None' if self.trace is None else '[...]'),
        )

    def __str__(self) -> str:
        """Pretty-print the object for exception messages."""
        return 'signal=%s' % repr(self.signal)


def sint(x: int) -> int:
    """Convert unsigned integer to signed integer."""
    return x - (x >> 255)*MOD


def uint(x: int) -> int:
    """Convert signed integer to unsigned integer."""
    return x + MOD if x < 0 else x


def cmpl(size: int) -> int:
    """Compute a 256-bit integer with a given number of leading 1s."""
    return (1 << size) - 1 << 256 - size


def intrinsic_gas(data: bytes) -> int:
    """Calculate fixed gas costs per transaction."""
    nb = sum(b == 0 for b in data)
    return 21_000 + 4*nb + 16*(len(data) - nb)


# TODO
# - [ ] split the top-level functionality into a separate function
def execute(
        chain: Chain,
        caller: int,
        address: int,
        value: int,
        data: bytes,
        gas: int = MOD - 1,  # largest u256
        static = False,
        trace = False,
        init = True,
        access_set: AccessSet = {},
        address_code: int = -1,
        value_env: int = -1,
    ) -> CallResult:
    """Run an Ethereum contract in a virtual machine."""
    if address_code == -1:
        address_code = address
    if value_env == -1:
        value_env = value
    code = chain[address_code].code  # access costs are incured in generic_call()
    space = Space(
        chain = chain.clone(),  # make a local clone for state reversal
        address = address,
        code = code,
        msg = {
            'value': value_env,
            'data': data,
            'caller': caller,
            'static': static,
        },
        stack = [],
        returndata = bytes(),
        trace = [] if trace else None,
        access_set = access_set.copy(),
        gas = gas,
    )
    pc = [0]
    try:
        space.chain.transfer(caller, address, value)
        if init:
            space.access_set.update({caller: 0, address: 0})
            space.gas -= intrinsic_gas(data)
        while True:
            opcode = code[pc[0]]
            pc[0] += 1
            OPCODES[opcode](space, pc)
    except Signal as signal:
        gas -= space.gas
        if int(signal) == 0:
            raise CallResult(signal, chain, access_set, gas, space.trace) from None
        else:
            return CallResult(signal, space.chain, space.access_set, gas, space.trace)


def mkcall(
        chain: Chain,
        caller: int,
        address: int,
        value: int,
        gas: int = MOD - 1,  # largest u256
        static=False,
        trace=False,
    ) -> Callable[[bytes], CallResult]:
    """Prepare a call that accepts arbibrary data."""
    return lambda data: execute(
        chain, caller, address, value, data, gas, static, trace
    )


def register(opcode: int) -> Any:
    """Register an opcode handler."""
    def wrapper(func):
        no_args = func.__code__.co_argcount - 2
        name = func.__name__
        def wrapped(s, pc):
            args = [s.stack.pop() for _ in range(no_args)]
            if s.trace is not None:
                # Record the trace before any external calls
                op = Operation(opcode, name, args)
                s.trace.append(op)
            rslt = None
            gas = s.gas
            try:
                rslt = func(s, pc, *args)
                if rslt is not None:
                    s.stack.append(rslt)
            finally:
                if s.trace is not None:
                    op.rslt = rslt
                    op.gas = gas - s.gas
        OPCODES[opcode] = wrapped
        return wrapped
    return wrapper


@register(0x0)
def stop(s: Space, pc: Pc) -> None:
    raise Stop()


@register(0x1)
def add(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return (a + b)%MOD


@register(0x2)
def mul(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 5
    return (a*b)%MOD


@register(0x3)
def sub(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return (a - b)%MOD


@register(0x4)
def div(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 5
    return a//b


@register(0x5)
def sdiv(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 5
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
    s.gas -= 5
    return a%b


@register(0x7)
def smod(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 5
    a = sint(a)
    b = sint(b)
    return uint(abs(a)%abs(b)*(1 if a >= 0 else -1))


@register(0x8)
def addmod(s: Space, pc: Pc, a: int, b: int, N: int) -> int:
    s.gas -= 8
    return (a + b)%N


@register(0x9)
def mulmod(s: Space, pc: Pc, a: int, b: int, N: int) -> int:
    s.gas -= 8
    return (a*b)%N


@register(0xa)
def exp(s: Space, pc: Pc, a: int, b: int) -> int:
    no_bytes = (b.bit_length() + 7)//8
    s.gas -= 10 + 50*no_bytes
    return pow(a, b, MOD)


@register(0xb)
def signextend(s: Space, pc: Pc, b: int, x: int) -> int:
    s.gas -= 5
    size = 8*(b + 1)
    x &= (1 << size) - 1
    if x >> size - 1 == 0:
        return x
    else:
        return x | cmpl(256 - size)


@register(0x10)
def lt(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return int(a < b)


@register(0x11)
def gt(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return int(a > b)


@register(0x12)
def slt(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return int(sint(a) < sint(b))


@register(0x13)
def sgt(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return int(sint(a) > sint(b))


@register(0x14)
def equal(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return int(a == b)


@register(0x15)
def iszero(s: Space, pc: Pc, a: int) -> int:
    s.gas -= 3
    return int(a == 0)


@register(0x16)
def bitwise_and(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return a & b


@register(0x17)
def bitwise_or(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return a | b


@register(0x18)
def bitwise_xor(s: Space, pc: Pc, a: int, b: int) -> int:
    s.gas -= 3
    return a ^ b


@register(0x19)
def bitwise_not(s: Space, pc: Pc, a: int) -> int:
    s.gas -= 3
    return a ^ (MOD - 1)


@register(0x20)
def sha3(s: Space, pc: Pc, offset: int, length: int) -> int:
    no_words = (length + 31)//32
    s.gas -= 30 + 6*no_words
    hash = keccak(s.memory[offset:offset+length])
    return int.from_bytes(hash, 'big')


@register(0x1a)
def byte(s: Space, pc: Pc, i: int, x: int) -> int:
    s.gas -= 3
    return (x >> (248 - i*8)) & 0xff


@register(0x1b)
def shl(s: Space, pc: Pc, shift: int, value: int) -> int:
    s.gas -= 3
    return (value << shift)%MOD


@register(0x1c)
def shr(s: Space, pc: Pc, shift: int, value: int) -> int:
    s.gas -= 3
    return value >> shift


@register(0x1d)
def sar(s: Space, pc: Pc, shift: int, value: int) -> int:
    s.gas -= 3
    if value >> 255 == 0:
        return value >> shift
    else:
        return value >> shift | cmpl(shift)


@register(0x30)
def address(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.address


@register(0x31)
def balance(s: Space, pc: Pc, address: int) -> int:
    return s.get_contract(address).balance


@register(0x33)
def caller(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.msg['caller']


@register(0x34)
def callvalue(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.msg['value']


@register(0x35)
def calldataload(s: Space, pc: Pc, offset: int) -> int:
    s.gas -= 3
    word = s.msg['data'][offset:offset+32]
    word += bytes(32 - len(word))  # Short data are padded on the right
    return int.from_bytes(word, 'big')


@register(0x36)
def calldatasize(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return len(s.msg['data'])


@register(0x37)
def calldatacopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    no_words = (length + 31)//32
    s.gas -= 3 + 3*no_words
    s.memory[dest_offset:dest_offset+length] = s.msg['data'][offset:offset+length]


@register(0x38)
def codesize(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return len(s.code)


@register(0x39)
def codecopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    no_words = (length + 31)//32
    s.gas -= 3 + 3*no_words
    s.memory[dest_offset:dest_offset+length] = s.code[offset:offset+length]


@register(0x3b)
def extcodesize(s: Space, pc: Pc, addr: int) -> int:
    return len(s.get_contract(addr).code)


@register(0x3d)
def returndatasize(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return len(s.returndata)


@register(0x3e)
def returndatacopy(s: Space, pc: Pc, dest_offset: int, offset: int, length: int) -> None:
    no_words = (length + 31)//32
    s.gas -= 3 + 3*no_words
    s.memory[dest_offset:dest_offset+length] = s.returndata[offset:offset+length]


@register(0x42)
def timestamp(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.chain.node.block_timestamp()


@register(0x43)
def number(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.chain.node.eth_blockNumber()


@register(0x47)
def selfbalance(s: Space, pc: Pc) -> int:
    s.gas -= 5
    return s.chain[s.address].balance  # skips warm access costs


@register(0x50)
def pop(s: Space, pc: Pc, _: int) -> None:
    s.gas -= 2
    pass


@register(0x51)
def mload(s: Space, pc: Pc, offset: int) -> int:
    s.gas -= 3
    word = s.memory[offset:offset+32]
    return int.from_bytes(word, 'big')


@register(0x52)
def mstore(s: Space, pc: Pc, offset: int, value: int) -> None:
    s.gas -= 3
    s.memory[offset:offset+32] = value.to_bytes(32, 'big')


@register(0x54)
def sload(s: Space, pc: Pc, key: int) -> int:
    value = s.chain[s.address].storage[key]
    if (s.address, key) in s.access_set:
        s.gas -= 100
    else:
        s.gas -= 2_100
        s.access_set[(s.address, key)] = value
    return value


# TODO:
# - [ ] proper gas refunds
# - [ ] restructure `if` clauses for better clarity
@register(0x55)
def sstore(s: Space, pc: Pc, key: int, value: int) -> None:
    if s.msg['static']:
        raise RuntimeError('Non-static method called from a static call')
    cval = s.chain[s.address].storage[key]
    if (s.address, key) not in s.access_set:
        s.gas -= 2_100
        s.access_set[(s.address, key)] = cval
    ival = s.access_set[(s.address, key)]
    if value == cval:
        s.gas -= 100
        return

    # value != cval
    if cval == ival:  # clean slot
        if ival == 0:
            s.gas -= 20_000
        else:
            s.gas -= 2_900
            if value == 0:
                s.gas += 4_800  # gas refund
    else:  # dirty slot
        s.gas -= 100
        if ival != 0:
            if cval == 0:
                s.gas -= 4_800  # negative gas refund
            elif value == 0:
                s.gas += 4_800  # gas refund
        if value == ival:
            if ival == 0:
                s.gas += 19_900  # gas refund
            else:
                s.gas += 2_800  # gas refund

    s.chain[s.address].storage[key] = value


@register(0x56)
def jump(s: Space, pc: Pc, dest: int) -> None:
    s.gas -= 8
    pc[0] = dest


@register(0x57)
def jumpi(s: Space, pc: Pc, dest: int, cond: int) -> None:
    s.gas -= 10
    if cond != 0:
        pc[0] = dest


@register(0x5a)
def gas(s: Space, pc: Pc) -> int:
    s.gas -= 2
    return s.gas


@register(0x5b)
def jumpdest(s: Space, pc: Pc) -> None:
    s.gas -= 1
    pass


# Register all push functions
# 0x60 - 0x7F
def push(k: int) -> Any:
    def push_k(s: Space, pc: Pc) -> int:
        s.gas -= 3
        value = int.from_bytes(s.code[pc[0]:pc[0]+k], 'big')
        pc[0] += k
        return value
    push_k.__name__ = f'push{k}'
    return push_k

for i in range(32):
    register(0x60 + i)(push(i + 1))


# Register all dup functions
# 0x80 - 0x8F
def dup(k: int) -> Any:
    def dup_k(s: Space, pc: Pc) -> int:
        s.gas -= 3
        return s.stack[-k]
    dup_k.__name__ = f'dup{k}'
    return dup_k

for i in range(16):
    register(0x80 + i)(dup(i + 1))


# Register all swap functions
# 0x90 - 0x9F
def swap(k: int) -> Any:
    def swap_k(s: Space, pc: Pc) -> None:
        s.gas -= 3
        s.stack[-1], s.stack[-k-1] = s.stack[-k-1], s.stack[-1]
    swap_k.__name__ = f'swap{k}'
    return swap_k

for i in range(16):
    register(0x90 + i)(swap(i + 1))


def generic_log(s: Space, pc: Pc, offset: int, length: int, *topics: int) -> None:
    s.gas -= 375*(len(topics) + 1) + 8*length
    _ = s.memory[offset:offset+length]  # trigger memory expansion costs
    # TODO: event handling


@register(0xa0)
def log0(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
    ) -> None:
    generic_log(s, pc, offset, length)


@register(0xa1)
def log1(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
    ) -> None:
    generic_log(s, pc, offset, length, topic0)


@register(0xa2)
def log2(
        s: Space,
        pc: Pc,
        offset: int,
        length: int,
        topic0: int,
        topic1: int,
    ) -> None:
    generic_log(s, pc, offset, length, topic0, topic1)


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
    generic_log(s, pc, offset, length, topic0, topic1, topic2)


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
    generic_log(s, pc, offset, length, topic0, topic1, topic2, topic3)


def generic_call(
        static: bool,
        s: Space,
        pc: Pc,
        caller: int,
        gas: int,
        addr_account: int,
        addr_code: int,
        value_trf: int,
        value_env: int,
        args_offset: int,
        args_length: int,
        ret_offset: int,
        ret_length: int,
    ) -> int:
    _ = s.get_contract(addr_code)  # incure access costs
    if value_trf > 0:
        s.gas -= 9000
    data = s.memory[args_offset:args_offset+args_length]
    try:
        rslt = execute(
            chain = s.chain,
            caller = caller,
            address = addr_account,
            value = value_trf,
            data = data,
            gas = gas,
            static = static,
            trace = s.trace is not None,
            init = False,
            access_set = s.access_set,
            address_code = addr_code,
            value_env = value_env,
        )
    except CallResult as err:
        rslt = err
    s.memory[ret_offset:ret_offset+ret_length] = rslt.data[:ret_length]
    s.returndata = rslt.data
    s.chain = rslt.chain
    s.access_set = rslt.access_set
    s.gas -= rslt.gas
    if s.trace is not None:
        s.trace.append(rslt.trace)
    return int(rslt)


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
        static = s.msg['static'],
        s = s,
        pc = pc,
        caller = s.address,
        gas = gas,
        addr_account = addr,
        addr_code = addr,
        value_trf = value,
        value_env = value,
        args_offset = args_offset,
        args_length = args_length,
        ret_offset = ret_offset,
        ret_length = ret_length,
    )


@register(0xf3)
def op_return(s: Space, pc: Pc, offset: int, length: int) -> None:
    raise Return(s.memory[offset:offset+length])


@register(0xf4)
def delegatecall(
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
        static = s.msg['static'],
        s = s,
        pc = pc,
        caller = s.msg['caller'],
        gas = gas,
        addr_account = s.address,
        addr_code = addr,
        value_trf = 0,
        value_env = s.msg['value'],
        args_offset = args_offset,
        args_length = args_length,
        ret_offset = ret_offset,
        ret_length = ret_length,
    )


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
        static = True,
        s = s,
        pc = pc,
        caller = s.address,
        gas = gas,
        addr_account = addr,
        addr_code = addr,
        value_trf = 0,
        value_env = 0,
        args_offset = args_offset,
        args_length = args_length,
        ret_offset = ret_offset,
        ret_length = ret_length,
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
