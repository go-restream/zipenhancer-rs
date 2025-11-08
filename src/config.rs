//! Configuration management for audio processing

use crate::error::{ZipEnhancerError, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub audio: AudioConfig,
    pub processing: ProcessingConfig,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub max_retries: u32,
    pub inference_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub overlap_ratio: f32,
    pub segment_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub enable_agc: bool,
    pub enable_performance_monitoring: bool,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            audio: AudioConfig::default(),
            processing: ProcessingConfig::default(),
            input_path: PathBuf::from("input.wav"),
            output_path: PathBuf::from("output.wav"),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./model/ZipEnhancer_ONNX/ZipEnhancer.onnx"),
            max_retries: 3,
            inference_threads: 4,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            overlap_ratio: 0.1,
            segment_size: 16000,
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_agc: true,
            enable_performance_monitoring: true,
            verbose: false,
        }
    }
}

impl Config {
    //! Get model path (convenience method)
        pub fn model_path(&self) -> &PathBuf {
            &self.model.path
        }
    
        /// Get sample rate (convenience method)
        pub fn sample_rate(&self) -> u32 {
            self.audio.sample_rate
        }
    
        /// Get overlap ratio (convenience method)
        pub fn overlap_ratio(&self) -> f32 {
            self.audio.overlap_ratio
        }
    
        /// Get segment size (convenience method)
        pub fn segment_size(&self) -> usize {
            self.audio.segment_size
        }
    
        /// Get max retries (convenience method)
        pub fn max_retries(&self) -> u32 {
            self.model.max_retries
        }
    
        /// Get inference thread count (convenience method)
        pub fn inference_threads(&self) -> usize {
            self.model.inference_threads
        }
    
        /// Get verbose mode (convenience method)
        pub fn verbose(&self) -> bool {
            self.processing.verbose
        }
    
        /// Get performance monitoring (convenience method)
        pub fn enable_performance_monitoring(&self) -> bool {
            self.processing.enable_performance_monitoring
        }
    
        /// Get overlap sample count
        pub fn overlap_samples(&self) -> usize {
            (self.segment_size() as f32 * self.overlap_ratio()) as usize
        }
    
        /// Get hop size (segment movement step)
        pub fn hop_size(&self) -> usize {
            self.segment_size() - self.overlap_samples()
        }
    }

#[derive(Debug, Parser)]
#[command(name = "zipenhancer", about = "Audio Denoise Processor", version, author)]
pub struct Args {
    #[arg(short = 'm', long = "model", default_value = "./model/ZipEnhancer_ONNX/ZipEnhancer.onnx", help = "ONNX model file path")]
    pub model: PathBuf,

    #[arg(short = 'i', long = "input", help = "Input audio file path (WAV format)")]
    pub input: PathBuf,

    #[arg(short = 'o', long = "output", default_value = "output.wav", help = "Output audio file path")]
    pub output: PathBuf,

    #[arg(short = 'r', long = "sample-rate", default_value = "16000", help = "Audio sample rate (Hz)")]
    pub sample_rate: u32,

    #[arg(short = 'l', long = "overlap", default_value = "0.1", help = "Inter-segment overlap ratio (0.0 - 1.0)")]
    pub overlap: f32,

    #[arg(short = 'v', long = "verbose", help = "Enable verbose output mode")]
    pub verbose: bool,

    #[arg(short = 's', long = "segment-size", default_value = "16000", help = "Audio segment size (number of samples)")]
    pub segment_size: usize,

    #[arg(long = "max-retries", default_value = "3", help = "ONNX inference max retry count")]
    pub max_retries: u32,

    #[arg(long = "disable-performance-monitoring", help = "Disable performance monitoring")]
    pub disable_performance_monitoring: bool,

    #[arg(long = "inference-threads", default_value = "4", help = "ONNX inference thread count")]
    pub inference_threads: usize,

    #[arg(short = 'c', long = "config", help = "Config file path (TOML format)")]
    pub config_file: Option<PathBuf>,

    #[arg(long = "test-only", help = "Run test mode only, do not process audio files")]
    pub test_only: bool,

    #[arg(long = "onnx-lib", help = "ONNX Runtime library file path")]
    pub onnx_lib: Option<PathBuf>,
}

impl Config {
    //! Create config from command line arguments
        pub fn from_args() -> Result<Self> {
            let args = Args::parse();
            Self::from_args_and_config(args)
        }
    
        /// Create config from command line arguments and config file
        pub fn from_args_and_config(args: Args) -> Result<Self> {
            // First load config file (if provided)
            let mut config = if let Some(config_path) = &args.config_file {
                Self::from_file(config_path)?
            } else {
                Self::default()
            };
    
            // Command line arguments override config file settings
            config.model.path = args.model;
            config.input_path = args.input;
            config.output_path = args.output;
            config.audio.sample_rate = args.sample_rate;
            config.audio.overlap_ratio = args.overlap;
            config.processing.verbose = args.verbose;
            config.audio.segment_size = args.segment_size;
            config.model.max_retries = args.max_retries;
            config.processing.enable_performance_monitoring = !args.disable_performance_monitoring;
            config.model.inference_threads = args.inference_threads;
    
            // Validate config
            config.validate()?;
    
            Ok(config)
        }
    
        /// Load config from TOML config file
        pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
            let content = std::fs::read_to_string(path)
                .map_err(|e| ZipEnhancerError::config(format!("Failed to read config file: {}", e)))?;
    
            toml::from_str(&content)
                .map_err(|e| ZipEnhancerError::config(format!("Failed to parse config file: {}", e)))
        }
    
        /// Validate configuration parameter validity
        pub fn validate(&self) -> Result<()> {
            // Validate sample rate
            if self.audio.sample_rate == 0 {
                return Err(ZipEnhancerError::config("Sample rate must be greater than 0"));
            }
            if self.audio.sample_rate > 192000 {
                return Err(ZipEnhancerError::config("Sample rate cannot exceed 192000 Hz"));
            }
    
            // Validate overlap ratio
            if self.audio.overlap_ratio < 0.0 || self.audio.overlap_ratio >= 1.0 {
                return Err(ZipEnhancerError::config("Overlap ratio must be in range [0.0, 1.0)"));
            }
    
            // Validate segment size
            if self.audio.segment_size == 0 {
                return Err(ZipEnhancerError::config("Segment size must be greater than 0"));
            }
            if self.audio.segment_size % 2 != 0 {
                return Err(ZipEnhancerError::config("Segment size must be even"));
            }
    
            // Validate retry count
            if self.model.max_retries > 10 {
                return Err(ZipEnhancerError::config("Max retries cannot exceed 10"));
            }
    
            // Validate thread count
            if self.model.inference_threads == 0 {
                return Err(ZipEnhancerError::config("Inference thread count must be greater than 0"));
            }
            if self.model.inference_threads > num_cpus::get() * 2 {
                return Err(ZipEnhancerError::config("Inference thread count cannot exceed 2x logical CPU cores"));
            }
    
            Ok(())
        }
    
        /// Save config to file
        pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let content = toml::to_string_pretty(self)
                .map_err(|e| ZipEnhancerError::config(format!("Failed to serialize config: {}", e)))?;
    
            std::fs::write(path, content)
                .map_err(|e| ZipEnhancerError::config(format!("Failed to write config file: {}", e)))
        }
    
        
        /// Create default config file
        pub fn create_default_config<P: AsRef<Path>>(path: P) -> Result<()> {
            let default_config = Self::default();
            default_config.save_to_file(path)
        }
    }

pub mod utils {
    pub fn cpu_count() -> usize {
        num_cpus::get()
    }

    pub fn recommended_segment_size(sample_rate: u32) -> usize {
        (sample_rate as usize).max(16000)
    }

    pub fn recommended_overlap_ratio(segment_size: usize) -> f32 {
        if segment_size >= 32000 {
            0.05
        } else if segment_size >= 16000 {
            0.1
        } else {
            0.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.overlap_ratio, 0.1);
        assert_eq!(config.segment_size, 16000);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.inference_threads, 4); 
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();

        assert!(config.validate().is_ok());

        config.sample_rate = 0;
        assert!(config.validate().is_err());
        config.sample_rate = 16000;

        config.overlap_ratio = 1.0;
        assert!(config.validate().is_err());
        config.overlap_ratio = 0.1;

        config.segment_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.toml");

        let config = Config::default();

        assert!(config.save_to_file(&config_path).is_ok());
        assert!(config_path.exists());

        let loaded_config = Config::from_file(&config_path).unwrap();
        assert_eq!(config.sample_rate, loaded_config.sample_rate);
        assert_eq!(config.overlap_ratio, loaded_config.overlap_ratio);
    }

    #[test]
    fn test_helper_functions() {
        assert!(utils::cpu_count() > 0);

        let segment_size = utils::recommended_segment_size(16000);
        assert_eq!(segment_size, 16000);

        let overlap = utils::recommended_overlap_ratio(16000);
        assert_eq!(overlap, 0.1);
    }

    #[test]
    fn test_overlap_calculations() {
        let config = Config {
            ..Default::default()
        };

        assert_eq!(config.overlap_samples(), 1600);
        assert_eq!(config.hop_size(), 14400);
    }
}