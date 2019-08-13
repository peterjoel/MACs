//! Support for the VPCLMULQDQ CPU intrinsic on `x86` and `x86_64` target
//! architectures.

// The code below uses `loadu`/`storeu` to support unaligned loads/stores
#![allow(clippy::cast_ptr_alignment)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::Xmm;
use core::{
    mem,
    ops::{BitXor, BitXorAssign},
};
use field::clmul::{self, Clmul};
use Block;

/// Wrapper for `__m128i` - a 128-bit XMM register (SSE2)
#[repr(align(16))]
#[derive(Copy, Clone)]
pub struct M128i(__m128i);

impl From<Block> for M128i {
    fn from(bytes: Block) -> M128i {
        M128i(unsafe { _mm_loadu_si128(bytes.as_ptr() as *const __m128i) })
    }
}

impl From<M128i> for Block {
    fn from(xmm: M128i) -> Block {
        let mut result = Block::default();

        unsafe {
            _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, xmm.0);
        }

        result
    }
}

impl From<u128> for M128i {
    fn from(x: u128) -> M128i {
        M128i(unsafe { _mm_loadu_si128(&x as *const u128 as *const __m128i) })
    }
}

impl BitXor for M128i {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        M128i(unsafe { xor(self.0, rhs.0) })
    }
}

impl BitXorAssign for M128i {
    fn bitxor_assign(&mut self, rhs: Self) {
        // TODO(tarcieri): optimize
        self.0 = unsafe { xor(self.0, rhs.0) };
    }
}

impl Clmul for M128i {
    fn clmul<I>(self, rhs: Self, imm: I) -> Self
    where
        I: Into<clmul::PseudoOp>,
    {
        M128i(unsafe { vpclmulqdq(self.0, rhs.0, imm.into()) })
    }
}

impl Xmm for M128i {
    /// Rotate the contents of the register left by 64-bits
    // TODO(tarcieri): better optimize; eliminate transmute
    fn rotate_left(self) -> Self {
        let mut t1 = [0u64; 2];

        unsafe {
            _mm_storeu_si128(t1.as_mut_ptr() as *mut __m128i, self.0);
        }

        let t2: [u32; 4] = unsafe { mem::transmute(t1) };
        let t3 = [t2[2], t2[3], t2[0], t2[1]];
        let t4: [u64; 2] = unsafe { mem::transmute(t3) };

        M128i(unsafe { _mm_loadu_si128(t4.as_ptr() as *const __m128i) })
    }

    /// Shift the contents of the register right by 64-bits
    // TODO(tarcieri): better optimize
    fn shift_right(self) -> Self {
        let mut u64x2 = [0u64; 2];

        unsafe {
            _mm_storeu_si128(u64x2.as_mut_ptr() as *mut __m128i, self.0);
        }

        u64x2[1] = u64x2[0];
        u64x2[0] = 0;

        M128i(unsafe { _mm_loadu_si128(u64x2.as_ptr() as *const __m128i) })
    }

    /// Shift the contents of the register left by 64-bits
    // TODO(tarcieri): better optimize
    fn shift_left(self) -> Self {
        let mut u64x2 = [0u64; 2];

        unsafe {
            _mm_storeu_si128(u64x2.as_mut_ptr() as *mut __m128i, self.0);
        }

        u64x2[0] = u64x2[1];
        u64x2[1] = 0;

        M128i(unsafe { _mm_loadu_si128(u64x2.as_ptr() as *const __m128i) })
    }
}

#[target_feature(enable = "sse2", enable = "sse4.1")]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a, b)
}

#[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
unsafe fn vpclmulqdq(a: __m128i, b: __m128i, op: clmul::PseudoOp) -> __m128i {
    match op {
        clmul::PseudoOp::PCLMULLQLQDQ => _mm_clmulepi64_si128(a, b, 0x00),
        clmul::PseudoOp::PCLMULHQLQDQ => _mm_clmulepi64_si128(a, b, 0x01),
        clmul::PseudoOp::PCLMULLQHQDQ => _mm_clmulepi64_si128(a, b, 0x10),
        clmul::PseudoOp::PCLMULHQHQDQ => _mm_clmulepi64_si128(a, b, 0x11),
    }
}

#[cfg(test)]
mod tests {
    use super::M128i;
    use field::{
        backend::soft::U64x2,
        clmul::{self, Clmul},
    };
    use Block;

    #[test]
    fn vclmul_emulation() {
        let a: u128 = 0xada5f29b;
        let b: u128 = 0x2d978a49;
        let op = clmul::PseudoOp::from(0x00);

        let hard_result: Block = M128i::from(a).clmul(b.into(), op).into();
        let soft_result: Block = U64x2::from(a).clmul(b.into(), op).into();

        assert_eq!(&hard_result, &soft_result);
    }
}
