"""Encoding and decoding functionality for Solidity ABI."""

# Load standard packages
from collections.abc import Callable
from types import FunctionType
from typing import Any, Generic, TypeVar
import inspect
import types
import typing

# Load local packages
from keccak import keccak
from evm import sint, uint

# Define fixed generic arrays. This is done in a roundabout way as Python does
# not support generic containers of fixed length out of the box.

T = TypeVar('T')

class array:
    """A fixed-size generic array."""

    def __new__(cls, size: int) -> Any:
        def set_size(cls):
            cls['size'] = size

        return types.new_class(
            name = f'array({size})',
            bases = (array, Generic[T]),
            exec_body = set_size,
        )


# Single dispatch is used to effectively be able to add methods to built-in
# types such as a tuple

def fo_type(cls: Any) -> Any:
    """Determine the first-order type for a given type."""
    if type(cls) == type:
        return cls
    else:
        t = typing.get_origin(cls)
        if t is None:
            return type(cls)
        elif issubclass(t, array):
            return array
        else:
            return t


C = TypeVar('C', bound=Callable)

class dispatch(Generic[C]):
    """A dispatch mechanism that uses the value of the first argument."""

    methods: dict[Any, C]
    default: C

    def __init__(self, default: C) -> None:
        self.methods = {}
        self.default = default

    def register(self, t: Any) -> Any:
        def wrapper(func: C):
            self.methods[t] = func
        return wrapper

    def __call__(self, t: Any, *args, **kwargs) -> Any:
        return self.methods.get(fo_type(t), self.default)(t, *args, **kwargs)


@dispatch
def signature(cls: Any) -> bytes:
    """Compute function or type solidity signature."""
    return cls.__name__.encode('utf8')


@dispatch
def size(cls: Any) -> int:
    """Compute the size of a fixed type (return 0 for dynamic types)."""
    return 32  # Most common default


@dispatch
def encode(cls: Any, value: Any) -> bytes:
    """Encode a Python value to bytes using Solidity ABI."""
    raise RuntimeError('Method not implemented')


@dispatch
def decode(cls: Any, value: bytes) -> Any:
    """Decode bytes to a Python value using Solidity ABI."""
    raise RuntimeError('Method not implemented')


# bool
@encode.register(bool)
def _(_cls: Any, value: bool) -> bytes:
    return value.to_bytes(32, 'big')


@decode.register(bool)
def _(_cls: Any, value: bytes) -> bool:
    return bool.from_bytes(value[:32], 'big')


# uint<M> (no overflow checks)
for M in range(8, 257, 8):
    code = f"""
class uint{M}:
    pass


@encode.register(uint{M})
def _(_cls: Any, value: int) -> bytes:
    return value.to_bytes(32, 'big')


@decode.register(uint{M})
def _(_cls: Any, value: bytes) -> int:
    return int.from_bytes(value[:32], 'big')
"""
    exec(code)


# int<M> (no overflow checks)
for M in range(8, 257, 8):
    code = f"""
class int{M}:
    pass


@encode.register(int{M})
def _(_cls: Any, value: int) -> bytes:
    return uint(value).to_bytes(32, 'big')


@decode.register(int{M})
def _(_cls: Any, value: bytes) -> int:
    return sint(int.from_bytes(value[:32], 'big'))
"""
    exec(code)


# address
class address:
    """Solidity address type."""

    pass


@encode.register(address)
def _(_cls: Any, value: int) -> bytes:
    return value.to_bytes(32, 'big')


@decode.register(address)
def _(_cls: Any, value: bytes) -> int:
    return int.from_bytes(value[:32], 'big')


# bytes<M>
for M in range(1, 33):
    code = f"""
class bytes{M}:
    pass


@encode.register(bytes{M})
def _(_cls: Any, value: bytes) -> bytes:
    value = value[:{M}]
    return value + bytes(32 - len(value))


@decode.register(bytes{M})
def _(_cls: Any, value: bytes) -> bytes:
    return value[:{M}]
"""
    exec(code)


# bytes
@size.register(bytes)
def _(_cls: Any) -> int:
    return 0


@encode.register(bytes)
def _(_cls: Any, value: bytes) -> bytes:
    size = len(value).to_bytes(32, 'big')
    data = value + bytes((-len(value))%32)
    return size + data


@decode.register(bytes)
def _(_cls: Any, value: bytes) -> bytes:
    size = int.from_bytes(value[:32], 'big')
    return value[32:32+size]


# string
class string:
    """Solidity string type."""

    pass


@size.register(string)
def _(_cls: Any) -> int:
    return 0


@encode.register(string)
def _(_cls: Any, value: str) -> bytes:
    return encode(bytes, value.encode('utf8'))


@decode.register(string)
def _(_cls: Any, value: bytes) -> str:
    return decode(bytes, value).decode('utf8')


# (T1, ..., Tk) ( tuple[T1, ..., Tk] )
@signature.register(tuple)
def _(cls: Any) -> bytes:
    args = typing.get_args(cls)
    return b'(' + b','.join(signature(t) for t in args) + b')'


@size.register(tuple)
def _(cls: Any) -> int:
    s = 0
    for t in typing.get_args(cls):
        if (st := size(t)) > 0:
            s += st
        else:
            return 0
    return s


@encode.register(tuple)
def _(cls: Any, value: tuple) -> bytes:
    head = []
    head_size = 0
    tail_data = b''
    for t, v in zip(typing.get_args(cls), value):
        data = encode(t, v)
        if (s := size(t)) > 0:
            head.append(data)
            head_size += s
        else:
            head.append(len(tail_data))
            head_size += 32
            tail_data += data
    head_data = b''.join(
        (h + head_size).to_bytes(32, 'big') if type(h) == int else h
        for h in head
    )
    return head_data + tail_data


@decode.register(tuple)
def _(cls: Any, value: bytes) -> tuple:
    rslt = []
    cursor = 0
    for t in typing.get_args(cls):
        if (s := size(t)) > 0:
            rslt.append(decode(t, value[cursor:]))
            cursor += s
        else:
            offset = int.from_bytes(value[cursor:cursor+32], 'big')
            rslt.append(decode(t, value[offset:]))
            cursor += 32
    return tuple(rslt)


# <type>[M] ( array(M)[type] )
@signature.register(array)
def _(cls: Any) -> str:
    s = signature(typing.get_args(cls)[0])
    return s + f'[{cls.size}]'.encode('utf8')


@size.register(array)
def _(cls: Any) -> int:
    return size(typing.get_args(cls)[0])*cls.size


@encode.register(array)
def _(cls: Any, value: list) -> bytes:
    return encode(typing.get_args(cls)*cls.size, value)


@decode.register(array)
def _(cls: Any, value: bytes) -> list:
    return list(decode(typing.get_args(cls)*cls.size, value))


# <type>[] ( list[type] )
@signature.register(list)
def _(cls: Any) -> str:
    s = signature(typing.get_args(cls)[0])
    return s + b'[]'


@size.register(list)
def _(_cls: Any) -> int:
    return 0


@encode.register(list)
def _(cls: Any, value: list) -> bytes:
    s = len(value).to_bytes(32, 'big')
    argtypes = tuple[typing.get_args(cls)*len(value)]  # type: ignore
    return s + encode(argtypes, value)


@decode.register(list)
def _(cls: Any, value: bytes) -> list:
    s = int.from_bytes(value[:32], 'big')
    argtypes = tuple[typing.get_args(cls)*s]  # type: ignore
    return list(decode(argtypes, value[32:]))


# function (signature only)
@signature.register(FunctionType)
def _(func: Callable) -> bytes:
    name = func.__name__.encode('utf8')
    sig = inspect.signature(func)
    argtypes = tuple(p.annotation for p in sig.parameters.values())
    solsig = b'%s(%s)' % (name, b','.join(signature(t) for t in argtypes))
    return keccak(solsig)[:4]


# Solidity wrapper
def solidity(func: Callable) -> Callable:
    """Wrap a call to a Solidity contract into a Python function."""
    solsig = signature(func)
    sig = inspect.signature(func)
    argnames = tuple(sig.parameters.keys())
    argtypes = tuple(p.annotation for p in sig.parameters.values())
    rtype = sig.return_annotation
    def wrapped(f, *args, **kwargs):
        f_kwargs = dict(zip(argnames, args), **kwargs)
        f_args = [f_kwargs[k] for k in argnames]
        data = solsig + encode(tuple[argtypes], f_args)
        rslt, chain, trace = f(data)
        if int(rslt) == 0:
            raise rslt
        if rtype is None or rtype == inspect.Signature.empty:
            rslt = None
        elif fo_type(rtype) == tuple:
            rslt = decode(rtype, rslt.data())
        else:
            # Unwrap single-valued tuples
            rslt = decode(tuple[rtype], rslt.data())[0]
        return rslt, chain, trace
    return wrapped
