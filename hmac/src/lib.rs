//! Generic implementation of Hash-based Message Authentication Code (HMAC).
//!
//! To use it you'll need a cryptographic hash function implementation from
//! RustCrypto project. You can either import specific crate (e.g. `sha2`), or
//! meta-crate `crypto-hashes` which reexport all related crates.
//!
//! # Usage
//! Let us demonstrate how to use HMAC using SHA256 as an example.
//!
//! To get the authentication code:
//!
//! ```rust
//! extern crate hmac;
//! extern crate sha2;
//!
//! use sha2::Sha256;
//! use hmac::{Hmac, Mac};
//!
//! // Create alias for HMAC-SHA256
//! type HmacSha256 = Hmac<Sha256>;
//!
//! # fn main() {
//! // Create HMAC-SHA256 instance which implements `Mac` trait
//! let mut mac = HmacSha256::new_varkey(b"my secret and secure key")
//!     .expect("HMAC can take key of any size");
//! mac.input(b"input message");
//!
//! // `result` has type `MacResult` which is a thin wrapper around array of
//! // bytes for providing constant time equality check
//! let result = mac.result();
//! // To get underlying array use `code` method, but be carefull, since
//! // incorrect use of the code value may permit timing attacks which defeat
//! // the security provided by the `MacResult`
//! let code_bytes = result.code();
//! # }
//! ```
//!
//! To verify the message:
//!
//! ```rust
//! # extern crate hmac;
//! # extern crate sha2;
//! # use sha2::Sha256;
//! # use hmac::{Hmac, Mac};
//! # fn main() {
//! # type HmacSha256 = Hmac<Sha256>;
//! let mut mac = HmacSha256::new_varkey(b"my secret and secure key")
//!     .expect("HMAC can take key of any size");
//!
//! mac.input(b"input message");
//!
//! # let code_bytes = mac.clone().result().code();
//! // `verify` will return `Ok(())` if code is correct, `Err(MacError)` otherwise
//! mac.verify(&code_bytes).unwrap();
//! # }
//! ```
//!
//! # Block and input sizes
//! Usually it is assumed that block size is larger than output size, due to the
//! generic nature of the implementation this edge case must be handled as well
//! to remove potential panic scenario. This is done by truncating hash output
//! to the hash block size if needed.
#![no_std]
#![doc(html_logo_url = "https://raw.githubusercontent.com/RustCrypto/meta/master/logo_small.png")]
pub extern crate crypto_mac;
pub extern crate digest;

use core::cmp::min;
use core::fmt;
pub use crypto_mac::Mac;
use crypto_mac::{InvalidKeyLength, MacResult};
use digest::generic_array::sequence::GenericSequence;
pub use digest::generic_array::{ArrayLength, GenericArray};
use digest::{BlockInput, FixedOutput, Input, Reset};

const IPAD: u8 = 0x36;
const OPAD: u8 = 0x5C;

/// The `Hmac` struct represents an HMAC using a given hash function `D`.
pub struct Hmac<D>
where
    D: Input + BlockInput + FixedOutput + Reset + Default + Clone,
    D::BlockSize: ArrayLength<u8>,
{
    digest: D,
    i_key_pad: GenericArray<u8, D::BlockSize>,
    opad_digest: D,
}

impl<D> Clone for Hmac<D>
where
    D: Input + BlockInput + FixedOutput + Reset + Default + Clone,
    D::BlockSize: ArrayLength<u8>,
{
    fn clone(&self) -> Hmac<D> {
        Hmac {
            digest: self.digest.clone(),
            i_key_pad: self.i_key_pad.clone(),
            opad_digest: self.opad_digest.clone(),
        }
    }
}

impl<D> fmt::Debug for Hmac<D>
where
    D: Input + BlockInput + FixedOutput + Reset + Default + Clone + fmt::Debug,
    D::BlockSize: ArrayLength<u8>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Hmac")
            .field("digest", &self.digest)
            .field("i_key_pad", &self.i_key_pad)
            .field("opad_digest", &self.opad_digest)
            .finish()
    }
}

impl<D> Mac for Hmac<D>
where
    D: Input + BlockInput + FixedOutput + Reset + Default + Clone,
    D::BlockSize: ArrayLength<u8>,
    D::OutputSize: ArrayLength<u8>,
{
    type OutputSize = D::OutputSize;
    type KeySize = D::BlockSize;

    fn new(key: &GenericArray<u8, Self::KeySize>) -> Self {
        Self::new_varkey(key.as_slice()).unwrap()
    }

    #[inline]
    fn new_varkey(key: &[u8]) -> Result<Self, InvalidKeyLength> {
        let mut hmac = Self {
            digest: Default::default(),
            i_key_pad: GenericArray::generate(|_| IPAD),
            opad_digest: Default::default(),
        };

        let mut opad = GenericArray::<u8, D::BlockSize>::generate(|_| OPAD);
        debug_assert!(hmac.i_key_pad.len() == opad.len());

        // The key that Hmac processes must be the same as the block size of the
        // underlying Digest. If the provided key is smaller than that, we just
        // pad it with zeros. If its larger, we hash it and then pad it with
        // zeros.
        if key.len() <= hmac.i_key_pad.len() {
            for (k_idx, k_itm) in key.iter().enumerate() {
                hmac.i_key_pad[k_idx] ^= *k_itm;
                opad[k_idx] ^= *k_itm;
            }
        } else {
            let mut digest = D::default();
            digest.input(key);
            let output = digest.fixed_result();
            // `n` is calculated at compile time and will equal
            // D::OutputSize. This is used to ensure panic-free code
            let n = min(output.len(), hmac.i_key_pad.len());
            for idx in 0..n {
                hmac.i_key_pad[idx] ^= output[idx];
                opad[idx] ^= output[idx];
            }
        }

        hmac.digest.input(&hmac.i_key_pad);
        hmac.opad_digest.input(&opad);

        Ok(hmac)
    }

    #[inline]
    fn input(&mut self, data: &[u8]) {
        self.digest.input(data);
    }

    #[inline]
    fn result(self) -> MacResult<D::OutputSize> {
        let mut opad_digest = self.opad_digest.clone();
        let hash = self.digest.fixed_result();
        opad_digest.input(&hash);
        MacResult::new(opad_digest.fixed_result())
    }

    #[inline]
    fn reset(&mut self) {
        self.digest.reset();
        self.digest.input(&self.i_key_pad);
    }
}
