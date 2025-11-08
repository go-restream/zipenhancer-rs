//! Audio Processing Pipeline Module
//!
//! Provides audio preprocessing, postprocessing, and main processing pipeline functions.
//! Includes audio segmentation, streaming processing, and overlap-add algorithms.

pub mod preprocessor;  // Preprocessing module
pub mod postprocessor; // Postprocessing module
pub mod segment;       // Segmentation module
pub mod performance;   // Performance monitoring module
pub mod processor;     // Main processor

// Re-export main types
pub use preprocessor::{AudioPreprocessor, PreprocessingConfig, AudioSegment};
pub use postprocessor::{AudioPostprocessor, PostprocessingConfig, OverlapAddBuffer};
pub use segment::{AudioSegmenter, SegmentationConfig, SegmentationStrategy};
pub use performance::{
    PerformanceMonitor, PreprocessingProfiler, PerformanceReport,
    RealTimePerformanceStats, PreprocessingPerformanceMetrics,
    MemoryTracker
};
pub use postprocessor::{WindowType, OverlapAddValidationResult};
pub use processor::{
    AudioProcessor, ProcessedSegment, ProcessingResult,
    PerformanceMetrics, ProcessorInfo
};