//! ONNX Inference Module
//!
//! Provides ONNX model loading, validation, and inference functions.
//! Uses ort crate for ONNX Runtime integration.

pub mod model;
pub mod session;
pub mod inference;

// Re-export main public interfaces
pub use model::{OnnxModel, ModelMetadata, ModelValidationResult, ModelLoader};
pub use session::{OnnxEnvironment, OnnxSession, SessionConfig, SessionInfo, DynamicTensor};
pub use inference::{InferenceEngine, InferenceConfig, InferenceStats, InferenceResult, BatchInferenceEngine};