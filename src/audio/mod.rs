//! Audio Processing Module
//!
//! Provides audio file reading, writing, format conversion, and basic processing functions.
//! Currently focuses on WAV format support.

pub mod wav;
pub mod converter;

pub use wav::{WavAudio, AudioFormat, AudioHeader, AudioData};
pub use converter::AudioConverter;