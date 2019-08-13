//! Field arithmetic backends

#[cfg(all(
    target_feature = "pclmulqdq",
    target_feature = "sse2",
    target_feature = "sse4.1",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod pclmulqdq;
mod soft;

use super::clmul::Clmul;
use core::ops::{BitXor, BitXorAssign};
use Block;

#[cfg(all(
    target_feature = "pclmulqdq",
    target_feature = "sse2",
    target_feature = "sse4.1",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(crate) use self::pclmulqdq::M128i;

#[cfg(not(all(
    target_feature = "pclmulqdq",
    target_feature = "sse2",
    target_feature = "sse4.1",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
pub(crate) use self::soft::U64x2 as M128i;

/// Trait representing the arithmetic operations we expect on the XMM registers
pub trait Xmm:
    BitXor<Output = Self> + BitXorAssign + Clmul + Copy + From<Block> + Into<Block> + From<u128>
{
    /// Rotate the contents of the register left by 64-bits
    fn rotate_left(self) -> Self;

    /// Shift the contents of the register right by 64-bits
    fn shift_right(self) -> Self;

    /// Shift the contents of the register left by 64-bits
    fn shift_left(self) -> Self;
}
