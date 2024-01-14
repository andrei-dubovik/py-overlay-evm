# Copyright (c) 2023-2024, Andrey Dubovik <andrei@dubovik.eu>

"""A NumPy-based implementation of Keccak-256."""

import math
import numpy as np
import numpy.typing as ntp

# Define Keccak-256 constants
RATE = 1088
CAPACITY = 512
DIGEST_SIZE = 256
WIDTH = (RATE + CAPACITY)//25
LOG_WIDTH = int(math.log2(WIDTH))

# Define types
Array = ntp.NDArray[np.uint8]


def keccak(msg: bytes, variant='legacy') -> bytes:
    """Compute a Keccak message digest."""
    msg = pad(msg, variant)
    n = len(msg)*8//RATE
    s = np.zeros(RATE + CAPACITY, dtype=np.uint8)
    for i in range(n):
        p8 = msg[i*RATE//8:(i+1)*RATE//8] + bytes(CAPACITY//8)
        p = np.unpackbits(np.frombuffer(p8, dtype=np.uint8), bitorder='little')
        s = keccak_p(s ^ p)
    return np.packbits(s[:DIGEST_SIZE], bitorder='little').tobytes()


def pad(msg: bytes, variant: str) -> bytes:
    """Apply 10*1 padding."""
    if variant == 'legacy':
        start = 0x01
    elif variant == 'nist':
        start = 0x06
    else:
        raise RuntimeError('Uknown padding variant')

    j = (-len(msg) - 1)%(RATE//8) + 1
    if j == 1:
        return msg + bytes([start | 0x80])
    else:
        return msg + bytes([start]) + bytes(j - 2) + b'\x80'


def keccak_p(s: Array) -> Array:
    """Compute a Kessak permutation."""
    a = s.reshape(5, 5, WIDTH).swapaxes(0, 1)
    for i in range(12 + 2*LOG_WIDTH):
        a = iota(chi(pi(rho(theta(a)))), i)
    return a.swapaxes(0, 1).reshape(-1)


def theta(a: Array) -> Array:
    x, z = np.indices((5, WIDTH))
    c = a[:,0,:] ^ a[:,1,:] ^ a[:,2,:] ^ a[:,3,:] ^ a[:,4,:]
    d = c[(x-1)%5,z] ^ c[(x+1)%5,(z-1)%WIDTH]
    return a ^ d[:,None,:]


def rho(a: Array) -> Array:
    a = a.copy()
    x, y = 1, 0
    z, = np.indices((WIDTH,))
    for t in range(24):
        a[x,y] = a[x,y,(z-(t+1)*(t+2)//2)%WIDTH]
        x, y = y, (2*x + 3*y)%5
    return a


def pi(a: Array) -> Array:
    x, y = np.indices((5, 5))
    return a[(x+3*y)%5,x]


def chi(a: Array) -> Array:
    x, = np.indices((5,))
    return a[x] ^ ((a[(x+1)%5] ^ 1) & a[(x+2)%5])


def iota(a: Array, i: int) -> Array:
    RC = np.zeros(WIDTH, dtype=np.uint8)
    for j in range(LOG_WIDTH + 1):
        RC[2**j-1] = rc(j + 7*i)
    a = a.copy()
    a[0,0] ^= RC
    return a


def rc(t: int) -> int:
    r = [1,0,0,0,0,0,0,0]
    for _ in range(t%255):
        r = [0] + r
        r[0] ^= r[8]
        r[4] ^= r[8]
        r[5] ^= r[8]
        r[6] ^= r[8]
        r = r[:8]
    return r[0]
