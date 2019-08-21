//! Implementation of POLYVAL's finite field.
//!
//! From [RFC 8452 Section 3] which defines POLYVAL for use in AES-GCM_SIV:
//!
//! > "POLYVAL, like GHASH (the authenticator in AES-GCM; ...), operates in a
//! > binary field of size 2^128.  The field is defined by the irreducible
//! > polynomial x^128 + x^127 + x^126 + x^121 + 1."
//!
//! This implementation provides multiplication over GF(2^128) optimized using
//! Shay Gueron's PCLMULQDQ-based techniques.
//!
//! For more information on how these techniques work, see:
//! <https://blog.quarkslab.com/reversing-a-finite-field-multiplication-optimization.html>
//!
//! [RFC 8452 Section 3]: https://tools.ietf.org/html/rfc8452#section-3

pub mod backend;
pub mod clmul;

use self::backend::Xmm;
use super::Block;
use core::ops::{BitXor, Mul};

/// Size of GF(2^128) in bytes (16-bytes).
pub const FIELD_SIZE: usize = 16;

/// Mask value used when performing Montgomery fast reduction.
/// This corresponds to POLYVAL's polynomial with the highest bit unset.
///
/// See: <https://crypto.stanford.edu/RealWorldCrypto/slides/gueron.pdf>
const MASK: u128 = 1 << 127 | 1 << 126 | 1 << 121 | 1;

/// POLYVAL field element.
#[derive(Copy, Clone)]
pub(crate) struct FieldElement<X: Xmm>(X);

impl<X: Xmm> FieldElement<X> {
    /// Load a `FieldElement` from its bytestring representation.
    pub fn from_bytes(bytes: Block) -> Self {
        FieldElement(bytes.into())
    }

    /// Serialize this `FieldElement` as a bytestring.
    pub fn to_bytes(self) -> Block {
        self.0.into()
    }

    /// Fast reduction modulo x^128 + x^127 + x^126 +x^121 + 1 (Gueron 2012)
    /// Algorithm 4: "Montgomery reduction"
    fn reduce(self) -> Self {
        let mask = X::from(MASK);
        let a = mask.clmul(self.0, 0x01);
        let b = self.0.rotate_left() ^ a;
        let c = mask.clmul(b, 0x01);
        FieldElement(b.rotate_left() ^ c)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<X: Xmm> Mul for FieldElement<X> {
    type Output = Self;

    /// Computes POLYVAL multiplication over GF(2^128).
    ///
    /// From [RFC 8452 Section 3]:
    ///
    /// > "The product of any two elements is calculated using standard
    /// > (binary) polynomial multiplication followed by reduction modulo the
    /// > irreducible polynomial."
    ///
    /// [RFC 8452 Section 3]: https://tools.ietf.org/html/rfc8452#section-3
    fn mul(self, rhs: Self) -> Self {
        let mut t1 = self.0.clmul(rhs.0, 0x00);
        let mut t2 = self.0.clmul(rhs.0, 0x01);
        let mut t3 = self.0.clmul(rhs.0, 0x10);
        let mut t4 = self.0.clmul(rhs.0, 0x11);

        t2 ^= t3;
        t3 = t2.shift_right();
        t2 = t2.shift_left();
        t1 ^= t3;
        t4 ^= t2;

        FieldElement(t4) ^ FieldElement(t1).reduce()
    }
}

impl<X: Xmm> From<X> for FieldElement<X> {
    fn from(element: X) -> FieldElement<X> {
        FieldElement(element)
    }
}

impl<X: Xmm> BitXor for FieldElement<X> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        FieldElement(self.0 ^ rhs.0)
    }
}
