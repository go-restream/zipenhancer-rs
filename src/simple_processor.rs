//! Simple audio processor implementation

use std::path::Path;
use crate::error::Result;
use crate::processing::{AudioProcessor, ProcessingResult};
use crate::config::{Config, ModelConfig, AudioConfig, ProcessingConfig};

pub struct RealAudioProcessor {
    sample_rate: u32,
    verbose: bool,
    #[allow(dead_code)]
    processor: Option<AudioProcessor>,
}

impl RealAudioProcessor {
    pub fn new(sample_rate: u32, verbose: bool) -> Self {
        Self {
            sample_rate,
            verbose,
            processor: None,
        }
    }

pub fn process_file(&mut self, input_path: &Path, output_path: &Path) -> Result<()> {
    if self.verbose {
        println!("Starting audio processing: {}", input_path.display());
    }

    let config = self.create_config(input_path, output_path)?;
    let mut processor = AudioProcessor::new(config.clone())?;
    processor.warm_up()?;

    if self.verbose {
        println!("Processing audio...");
    }

    let result = processor.process_file(input_path, output_path)?;

    if self.verbose {
        self.display_result(&result);
    }

    Ok(())
}

fn create_config(&self, input_path: &Path, output_path: &Path) -> Result<Config> {
    let model_path = Path::new("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");

    if !model_path.exists() {
        return Err(crate::error::ZipEnhancerError::config(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    Ok(Config {
        model: ModelConfig {
            path: model_path.to_path_buf(),
            max_retries: 3,
            inference_threads: num_cpus::get(),
        },
        audio: AudioConfig {
            sample_rate: self.sample_rate,
            overlap_ratio: 0.1,
            segment_size: 16000,
        },
        processing: ProcessingConfig {
            enable_agc: true,
            enable_performance_monitoring: true,
            verbose: self.verbose,
        },
        input_path: input_path.to_path_buf(),
        output_path: output_path.to_path_buf(),
    })
}

fn display_result(&self, result: &ProcessingResult) {
    println!("=== Processing Complete ===");
    println!("Input: {}", result.input_path.display());
    println!("Output: {}", result.output_path.display());
    println!("Time: {:.2}s", result.performance_metrics.processing_time_seconds);
    println!("RTF: {:.3}", result.performance_metrics.real_time_factor);
    println!("Duration: {:.2}s", result.performance_metrics.input_duration_seconds);
    println!("Segments: {}", result.performance_metrics.segment_count);

    if result.performance_metrics.average_inference_time_ms > 0.0 {
        println!("Avg inference: {:.2}ms", result.performance_metrics.average_inference_time_ms);
    }

    println!("✅ Complete!");
}
}

pub type SimpleAudioProcessor = RealAudioProcessor;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_real_processor_creation() {
        let processor = RealAudioProcessor::new(16000, true);
        assert_eq!(processor.sample_rate, 16000);
        assert!(processor.verbose);
        assert!(processor.processor.is_none());
    }

    #[test]
    fn test_simple_processor_alias() {
        let processor = SimpleAudioProcessor::new(16000, true);
        assert_eq!(processor.sample_rate, 16000);
        assert!(processor.verbose);
    }

    #[test]
    fn test_config_creation() {
        let processor = RealAudioProcessor::new(16000, false);

        let input_path = PathBuf::from("test_input.wav");
        let output_path = PathBuf::from("test_output.wav");

        let config_result = processor.create_config(&input_path, &output_path);

        let model_path = Path::new("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");
        if model_path.exists() {
            assert!(config_result.is_ok());
            let config = config_result.unwrap();
            assert_eq!(config.sample_rate, 16000);
            assert_eq!(config.input_path, input_path);
            assert_eq!(config.output_path, output_path);
            assert_eq!(config.segment_size, 16000);
            assert_eq!(config.overlap_ratio, 0.1);
        } else {
            assert!(config_result.is_err());
        }
    }
}