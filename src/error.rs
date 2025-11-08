//! Error Type Definition Module
//!
//! Defines all possible error types in ZipEnhancer Rust version,
//! providing clear error classification and handling mechanisms.

use std::fmt;

/// Main error type for ZipEnhancer application
#[derive(Debug, Clone)]
pub enum ZipEnhancerError {
    /// Audio processing related errors
    Audio { message: String },

    /// ONNX model related errors
    Onnx { message: String },

    /// Configuration related errors
    Config { message: String },

    /// File I/O related errors
    Io { message: String },

    /// Data validation related errors
    Validation { message: String },

    /// Performance related errors
    Performance { message: String },

    /// Processing logic related errors
    Processing { message: String },
}

impl fmt::Display for ZipEnhancerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZipEnhancerError::Audio { message } => write!(f, "Audio processing error: {message}"),
            ZipEnhancerError::Onnx { message } => write!(f, "ONNX model error: {message}"),
            ZipEnhancerError::Config { message } => write!(f, "Configuration error: {message}"),
            ZipEnhancerError::Io { message } => write!(f, "File I/O error: {message}"),
            ZipEnhancerError::Validation { message } => write!(f, "Data validation error: {message}"),
            ZipEnhancerError::Performance { message } => write!(f, "Performance error: {message}"),
            ZipEnhancerError::Processing { message } => write!(f, "Processing logic error: {message}"),
        }
    }
}

impl std::error::Error for ZipEnhancerError {}

impl ZipEnhancerError {
    /// Create new audio error
    pub fn audio<S: Into<String>>(message: S) -> Self {
        Self::Audio {
            message: message.into(),
        }
    }

    /// Create new config error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create new validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create new performance error
    pub fn performance<S: Into<String>>(message: S) -> Self {
        Self::Performance {
            message: message.into(),
        }
    }

    /// Create new processing error
    pub fn processing<S: Into<String>>(message: S) -> Self {
        Self::Processing {
            message: message.into(),
        }
    }

    /// Create new ONNX error
    pub fn onnx<S: Into<String>>(message: S) -> Self {
        Self::Onnx {
            message: message.into(),
        }
    }

    /// Create new IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        Self::Io {
            message: message.into(),
        }
    }
}

/// Application result type alias
pub type Result<T> = std::result::Result<T, ZipEnhancerError>;

impl From<std::io::Error> for ZipEnhancerError {
    fn from(err: std::io::Error) -> Self {
        ZipEnhancerError::io(err.to_string())
    }
}

// Implement From trait for onnxruntime::OrtError
impl From<onnxruntime::OrtError> for ZipEnhancerError {
    fn from(err: onnxruntime::OrtError) -> Self {
        ZipEnhancerError::onnx(format!("ONNX Runtime error: {}", err))
    }
}

/// Audio format error details
#[derive(Debug)]
pub enum AudioFormatError {
    /// Unsupported audio format
    UnsupportedFormat(String),
    /// Corrupted audio file
    CorruptedFile(String),
    /// Invalid audio parameters
    InvalidParameters(String),
    /// Sample rate mismatch
    SampleRateMismatch { expected: u32, actual: u32 },
    /// Channel count mismatch
    ChannelCountMismatch { expected: u16, actual: u16 },
    /// Bit depth mismatch
    BitDepthMismatch { expected: u16, actual: u16 },
}

impl fmt::Display for AudioFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioFormatError::UnsupportedFormat(format) => {
                write!(f, "Unsupported audio format: {format}")
            }
            AudioFormatError::CorruptedFile(file) => {
                write!(f, "Corrupted audio file: {file}")
            }
            AudioFormatError::InvalidParameters(params) => {
                write!(f, "Invalid audio parameters: {params}")
            }
            AudioFormatError::SampleRateMismatch { expected, actual } => {
                write!(f, "Sample rate mismatch: expected {expected}, actual {actual}")
            }
            AudioFormatError::ChannelCountMismatch { expected, actual } => {
                write!(f, "Channel count mismatch: expected {expected}, actual {actual}")
            }
            AudioFormatError::BitDepthMismatch { expected, actual } => {
                write!(f, "Bit depth mismatch: expected {expected}, actual {actual}")
            }
        }
    }
}

impl std::error::Error for AudioFormatError {}

/// ONNX model error details
#[derive(Debug)]
pub enum OnnxModelError {
    /// Model file not found
    ModelNotFound(String),
    /// Model loading failed
    LoadFailed(String),
    /// Model input validation failed
    InputValidationFailed(String),
    /// Model output validation failed
    OutputValidationFailed(String),
    /// Inference execution failed
    InferenceFailed(String),
    /// Model session creation failed
    SessionCreationFailed(String),
    /// Memory allocation failed
    MemoryAllocationFailed,
}

impl fmt::Display for OnnxModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxModelError::ModelNotFound(path) => {
                write!(f, "Model file not found: {}", path)
            }
            OnnxModelError::LoadFailed(msg) => {
                write!(f, "Model loading failed: {}", msg)
            }
            OnnxModelError::InputValidationFailed(msg) => {
                write!(f, "Model input validation failed: {}", msg)
            }
            OnnxModelError::OutputValidationFailed(msg) => {
                write!(f, "Model output validation failed: {}", msg)
            }
            OnnxModelError::InferenceFailed(msg) => {
                write!(f, "Inference execution failed: {}", msg)
            }
            OnnxModelError::SessionCreationFailed(msg) => {
                write!(f, "Model session creation failed: {}", msg)
            }
            OnnxModelError::MemoryAllocationFailed => {
                write!(f, "Memory allocation failed")
            }
        }
    }
}

impl std::error::Error for OnnxModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let audio_err = ZipEnhancerError::audio("Test audio error");
        assert!(matches!(audio_err, ZipEnhancerError::Audio { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = ZipEnhancerError::config("Invalid configuration parameters");
        let display = format!("{}", err);
        assert!(display.contains("Configuration error"));
    }

    #[test]
    fn test_audio_format_error() {
        let err = AudioFormatError::UnsupportedFormat("MP3".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Unsupported audio format"));
        assert!(display.contains("MP3"));
    }

    #[test]
    fn test_onnx_model_error() {
        let err = OnnxModelError::ModelNotFound("model.onnx".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Model file not found"));
        assert!(display.contains("model.onnx"));
    }

    #[test]
    fn test_error_clone() {
        let err = ZipEnhancerError::audio("Test error");
        let cloned_err = err.clone();
        assert!(matches!(cloned_err, ZipEnhancerError::Audio { .. }));
    }
}