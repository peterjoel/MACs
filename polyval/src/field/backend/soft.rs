//! Software emulation support for CLMUL hardware intrinsics.
//!
//! WARNING: Not constant time! Should be made constant-time or disabled by default.

use super::Xmm;
use byteorder::{ByteOrder, LE};
use core::{
    mem,
    ops::{BitXor, BitXorAssign},
};
use field::clmul::{self, Clmul};
use Block;

/// 2 x `u64` values emulating an XMM register
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct U64x2([u64; 2]);

impl From<Block> for U64x2 {
    fn from(bytes: Block) -> U64x2 {
        let mut u64x2 = [0u64; 2];
        LE::read_u64_into(&bytes, &mut u64x2);
        U64x2(u64x2)
    }
}

impl From<U64x2> for Block {
    fn from(u64x2: U64x2) -> Block {
        let x: u128 = u64x2.into();
        let mut result = Block::default();
        LE::write_u128(&mut result, x);
        result
    }
}

impl From<u128> for U64x2 {
    fn from(x: u128) -> U64x2 {
        let lo = (x & 0xFFFF_FFFFF) as u64;
        let hi = (x >> 64) as u64;
        U64x2([lo, hi])
    }
}

impl From<U64x2> for u128 {
    fn from(u64x2: U64x2) -> u128 {
        u128::from(u64x2.0[0]) | (u128::from(u64x2.0[1]) << 64)
    }
}

impl BitXor for U64x2 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        U64x2([self.0[0] ^ rhs.0[0], self.0[1] ^ rhs.0[1]])
    }
}

impl BitXorAssign for U64x2 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0[0] ^= rhs.0[0];
        self.0[1] ^= rhs.0[1];
    }
}

impl Clmul for U64x2 {
    fn clmul<I>(self, other: Self, imm: I) -> Self
    where
        I: Into<clmul::PseudoOp>,
    {
        let (a, b) = match imm.into() {
            clmul::PseudoOp::PCLMULLQLQDQ => (self.0[0], other.0[0]),
            clmul::PseudoOp::PCLMULHQLQDQ => (self.0[1], other.0[0]),
            clmul::PseudoOp::PCLMULLQHQDQ => (self.0[0], other.0[1]),
            clmul::PseudoOp::PCLMULHQHQDQ => (self.0[1], other.0[1]),
        };

        let mut result = [0u64; 2];

        for i in 0..64 {
            if b & (1 << i) != 0 {
                result[1] ^= a;
            }

            result[0] >>= 1;

            if result[1] & 1 != 0 {
                result[0] ^= 1 << 63;
            }

            result[1] >>= 1;
        }

        U64x2(result)
    }
}

impl Xmm for U64x2 {
    /// Rotate the contents of the register left by 64-bits
    fn rotate_left(self) -> Self {
        let t2: [u32; 4] = unsafe { mem::transmute(self.0) };
        let t3 = [t2[2], t2[3], t2[0], t2[1]];
        let t4: [u64; 2] = unsafe { mem::transmute(t3) };
        U64x2(t4)
    }

    /// Shift the contents of the register right by 64-bits
    fn shift_right(self) -> Self {
        let mut u64x2 = self.0;
        u64x2[1] = u64x2[0];
        u64x2[0] = 0;
        U64x2(u64x2)
    }

    /// Shift the contents of the register left by 64-bits
    fn shift_left(self) -> Self {
        let mut u64x2 = self.0;
        u64x2[0] = u64x2[1];
        u64x2[1] = 0;
        U64x2(u64x2)
    }
}
