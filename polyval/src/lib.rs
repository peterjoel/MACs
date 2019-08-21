//! **POLYVAL** is a GHASH-like universal hash over GF(2^128) useful for
//! implementing [AES-GCM-SIV] or [AES-GCM/GMAC].
//!
//! From [RFC 8452 Section 3] which defines POLYVAL for use in AES-GCM_SIV:
//!
//! > "POLYVAL, like GHASH (the authenticator in AES-GCM; ...), operates in a
//! > binary field of size 2^128.  The field is defined by the irreducible
//! > polynomial x^128 + x^127 + x^126 + x^121 + 1."
//!
//! By multiplying (in the finite field sense) a sequence of 128-bit blocks of
//! input data data by a field element `H`, POLYVAL can be used to authenticate
//! the message sequence as powers (in a finite field sense) of `H`.
//!
//! ## Relationship to GHASH
//!
//! POLYVAL can be thought of as the little endian equivalent of GHASH, which
//! affords it a small performance advantage over GHASH when used on little
//! endian architectures.
//!
//! It has also been designed so it can also be used to compute GHASH and with
//! it GMAC, the Message Authentication Code (MAC) used by AES-GCM.
//!
//! From [RFC 8452 Appendix A]:
//!
//! > "GHASH and POLYVAL both operate in GF(2^128), although with different
//! > irreducible polynomials: POLYVAL works modulo x^128 + x^127 + x^126 +
//! > x^121 + 1 and GHASH works modulo x^128 + x^7 + x^2 + x + 1.  Note
//! > that these irreducible polynomials are the 'reverse' of each other."
//!
//! [AES-GCM-SIV]: https://en.wikipedia.org/wiki/AES-GCM-SIV
//! [AES-GCM/GMAC]: https://en.wikipedia.org/wiki/Galois/Counter_Mode
//! [RFC 8452 Section 3]: https://tools.ietf.org/html/rfc8452#section-3
//! [RFC 8452 Appendix A]: https://tools.ietf.org/html/rfc8452#appendix-A

#![no_std]
#![doc(html_logo_url = "https://raw.githubusercontent.com/RustCrypto/meta/master/logo_small.png")]
#![deny(missing_docs)]

extern crate byteorder;
#[cfg(feature = "zeroize")]
extern crate zeroize;

pub mod field;

use self::field::FieldElement;
#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

// TODO(tarcieri): runtime selection of CLMUL vs soft backend when both are available
use self::field::backend::M128i;

/// Size of a POLYVAL block (128-bits)
pub const BLOCK_SIZE: usize = 16;

/// POLYVAL blocks (16-bytes)
pub type Block = [u8; BLOCK_SIZE];

/// **POLYVAL**: GHASH-like universal hash over GF(2^128).
#[allow(non_snake_case)]
#[derive(Clone)]
#[repr(align(16))]
pub struct Polyval {
    /// GF(2^128) field element input blocks are multiplied by
    H: FieldElement<M128i>,

    /// Field element representing the computed universal hash
    S: FieldElement<M128i>,
}

impl Polyval {
    /// Initialize POLYVAL with the given `H` field element
    pub fn new(h: Block) -> Self {
        Self {
            H: FieldElement::from_bytes(h),
            S: FieldElement::from_bytes(Block::default()),
        }
    }

    /// Input a field element `X` to be authenticated into POLYVAL.
    pub fn input(&mut self, x: Block) {
        // "The sum of any two elements in the field is the result of XORing them."
        // -- RFC 8452 Section 3
        let sum = self.S ^ FieldElement::from_bytes(x);
        self.S = sum * self.H;
    }

    /// Process input blocks in a chained manner
    pub fn chain(mut self, x: Block) -> Self {
        self.input(x);
        self
    }

    /// Get POLYVAL result (i.e. computed `S` field element)
    pub fn result(self) -> Block {
        self.S.to_bytes()
    }
}

#[cfg(feature = "zeroize")]
impl Drop for Polyval {
    fn drop(&mut self) {
        self.H.zeroize();
        self.S.zeroize();
    }
}
