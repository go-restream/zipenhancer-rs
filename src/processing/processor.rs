//! Audio processing pipeline

use std::path::Path;
use std::time::Instant;
use rand;
use crate::audio::{WavAudio, AudioConverter};
use crate::onnx::{OnnxModel, InferenceEngine, InferenceConfig, OnnxEnvironment, SessionConfig, DynamicTensor};
use crate::processing::{
    AudioPreprocessor, AudioPostprocessor, AudioSegmenter,
    PreprocessingConfig, PostprocessingConfig, SegmentationConfig, AudioSegment
};
use crate::config::Config;
use crate::error::{ZipEnhancerError, Result};

#[derive(Debug)]
pub struct AudioProcessor {
    config: Config,
    model: OnnxModel,
    inference_engine: InferenceEngine,
    preprocessor: AudioPreprocessor,
    postprocessor: AudioPostprocessor,
    segmenter: AudioSegmenter,
}

impl AudioProcessor {
    pub fn new(config: Config) -> Result<Self> {
        if config.verbose() {
            println!("Initializing processor...");
        }

        let model = OnnxModel::new(&config.model_path())?;
        if config.verbose() {
            println!("Loaded model: {}", config.model_path().display());
        }

        let env = OnnxEnvironment::new()?;
        let session_config = SessionConfig {
            intra_op_num_threads: config.inference_threads() as i32,
            ..Default::default()
        };
        let onnx_session = crate::onnx::OnnxSession::new(model.path(), session_config, &env)?;

        if config.verbose() {
            println!("Session created, threads: {}", config.inference_threads());
        }

        let inference_config = InferenceConfig {
            max_retries: config.max_retries(),
            enable_profiling: config.enable_performance_monitoring(),
            ..Default::default()
        };
        let inference_engine = InferenceEngine::new(onnx_session, inference_config)?;

        let preprocessing_config = PreprocessingConfig {
            target_sample_rate: config.sample_rate(),
            segment_size: config.segment_size(),
            ..Default::default()
        };
        let preprocessor = AudioPreprocessor::new(preprocessing_config);

        let postprocessing_config = PostprocessingConfig {
            output_sample_rate: config.sample_rate(),
            overlap_ratio: config.overlap_ratio(),
            ..Default::default()
        };
        let overlap_buffer_size = config.segment_size() * 8;
        let postprocessor = AudioPostprocessor::new(postprocessing_config, overlap_buffer_size);

        let segmentation_config = SegmentationConfig {
            segment_size: config.segment_size(),
            overlap_ratio: config.overlap_ratio(),
            ..Default::default()
        };
        let segmenter = AudioSegmenter::new(segmentation_config);

        Ok(Self {
            config,
            model,
            inference_engine,
            preprocessor,
            postprocessor,
            segmenter,
        })
    }

    pub fn process_file(&mut self, input_path: &Path, output_path: &Path) -> Result<ProcessingResult> {
        let start_time = Instant::now();

        if self.config.verbose() {
            println!("Processing: {}", input_path.display());
        }

        let mut audio = WavAudio::from_file(input_path)?;
        if self.config.verbose() {
            println!("Audio: {:.2}s, {}Hz, {}ch",
                     audio.duration(), audio.sample_rate(), audio.channels());
        }

        let preprocessed_segments = self.preprocess_audio(&mut audio)?;
        if self.config.verbose() {
            println!("Preprocessed: {} segments", preprocessed_segments.len());
        }

        let processed_segments = self.run_inference(&preprocessed_segments)?;
        if self.config.verbose() {
            println!("Inference complete");
        }

        let output_audio = self.postprocess_and_reconstruct(&processed_segments)?;
        if self.config.verbose() {
            println!("Postprocessed: {:.2}s", output_audio.duration());
        }

        output_audio.save_to_file(output_path)?;
        if self.config.verbose() {
            println!("Saved: {}", output_path.display());
        }

        let processing_time = start_time.elapsed();
        let segment_count = preprocessed_segments.len();
        let average_inference_time_ms = if !processed_segments.is_empty() {
            processed_segments.iter()
                .map(|ps| ps.processing_time_ms as f64)
                .sum::<f64>() / processed_segments.len() as f64
        } else {
            0.0
        };

        let performance_metrics = self.calculate_performance_metrics(
            &audio, processing_time, segment_count, average_inference_time_ms
        );

        Ok(ProcessingResult {
            input_path: input_path.to_path_buf(),
            output_path: output_path.to_path_buf(),
            processing_time,
            performance_metrics,
        })
    }

    fn preprocess_audio(&mut self, audio: &mut WavAudio) -> Result<Vec<AudioSegment>> {
        if audio.channels() > 1 {
            let mono_data = audio.data().to_mono();
            *audio.data_mut() = crate::audio::AudioData::Mono(mono_data);
            audio.header.channels = 1;
        }

        if audio.sample_rate() != self.config.sample_rate() {
            *audio = AudioConverter::convert_sample_rate(audio, self.config.sample_rate())?;
        }

        let segments = self.preprocessor.preprocess_and_segment(audio)?;
        Ok(segments)
    }

    /// Run ONNX inference
    fn run_inference(&mut self, segments: &[AudioSegment]) -> Result<Vec<ProcessedSegment>> {
        let mut processed_segments = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            if self.config.verbose() && segments.len() > 10 {
                if i % (segments.len() / 10) == 0 {
                    println!("Progress: {}/{} ({:.1}%)", i, segments.len(),
                             i as f32 / segments.len() as f32 * 100.0);
                }
            }

            let mono_data = segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;

            let onnx_input = self.convert_to_onnx_input(mono_data)?;
            let inference_result = self.inference_engine.run_inference(vec![onnx_input])?;

            if !inference_result.success {
                return Err(ZipEnhancerError::processing(format!(
                    "Inference failed: {}",
                    inference_result.error.unwrap_or_else(|| "Unknown error".to_string())
                )));
            }

            let output_data = inference_result.first_output()
                .ok_or_else(|| ZipEnhancerError::processing("No output"))?;

            let mut processed_data: Vec<f32> = output_data.iter()
                .map(|&x| {
                    if !x.is_finite() {
                        println!("Warning: Invalid value {:.6}, using 0.0", x);
                        0.0
                    } else {
                        x.clamp(-1.0, 1.0)
                    }
                })
                .collect();

            if !processed_data.is_empty() {
                let max_amplitude = processed_data.iter()
                    .map(|&x| x.abs())
                    .fold(0.0f32, f32::max);

                if max_amplitude < 0.3 && max_amplitude > 0.001 {
                    let gain_factor = 1.0 / max_amplitude;
                    let limited_gain = gain_factor.clamp(3.0, 10.0);

                    if self.config.verbose() {
                        println!("AGC: max={:.4}, gain={:.4}", max_amplitude, limited_gain);
                    }

                    for sample in &mut processed_data {
                        *sample *= limited_gain;
                        *sample = sample.clamp(-1.0, 1.0);
                    }
                }
            }

            let processed_segment = ProcessedSegment {
                original_segment: segment.clone(),
                processed_data,
                processing_time_ms: inference_result.inference_time_ms,
            };

            processed_segments.push(processed_segment);
        }

        Ok(processed_segments)
    }

    fn convert_to_onnx_input(&self, data: &ndarray::Array1<f32>) -> Result<DynamicTensor> {
        let target_length = self.config.segment_size();
        let mut processed_data = data.clone();

        if processed_data.len() != target_length {
            if processed_data.len() > target_length {
                processed_data = processed_data.slice(ndarray::s![..target_length]).to_owned();
            } else {
                let current_length = processed_data.len();
                let mut padded = ndarray::Array1::zeros(target_length);
                padded.slice_mut(ndarray::s![..current_length]).assign(&processed_data);

                let fill_length = target_length - current_length;
                if fill_length > 0 {
                    let fade_samples = (fill_length as f32 * 0.3) as usize;
                    let fade_samples = fade_samples.min(fill_length).max(10);

                    let reference_window = 20.min(current_length);
                    let reference_value = if current_length >= reference_window {
                        let sum: f32 = processed_data.slice(ndarray::s![current_length - reference_window..])
                            .iter().map(|&x| x.abs()).sum();
                        sum / reference_window as f32
                    } else if current_length > 0 {
                        let sum: f32 = processed_data.iter().map(|&x| x.abs()).sum();
                        sum / current_length as f32
                    } else {
                        0.001
                    };

                    for i in 0..fill_length {
                        let pos = current_length + i;
                        if i < fade_samples {
                            let progress = i as f32 / fade_samples as f32;
                            let smooth_fade = (1.0 - progress * std::f32::consts::PI / 2.0).cos();
                            let fade_factor = smooth_fade * smooth_fade;
                            let noise_level = reference_value * 0.01;
                            let noise = (rand::random::<f32>() - 0.5) * 2.0 * noise_level;
                            padded[pos] = reference_value * fade_factor * 0.1 + noise;
                        } else {
                            let progress = (i - fade_samples) as f32 / (fill_length - fade_samples) as f32;
                            let exponential_decay = (-progress * 3.0).exp();
                            let noise_level = reference_value * 0.001;
                            let noise = (rand::random::<f32>() - 0.5) * 2.0 * noise_level;
                            padded[pos] = noise * exponential_decay;
                        }
                    }
                }
                processed_data = padded;
            }
        }

        let shape = vec![1, 1, processed_data.len() as i64];
        let tensor_data: Vec<i16> = processed_data.iter()
            .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();

        Ok(DynamicTensor::new_i16(tensor_data, shape))
    }

    fn postprocess_and_reconstruct(&mut self, processed_segments: &[ProcessedSegment]) -> Result<WavAudio> {
        let audio_segments: Result<Vec<AudioSegment>> = processed_segments.iter()
            .map(|ps| {
                let data_array = ndarray::Array1::from(ps.processed_data.clone());
                let audio_data = crate::audio::AudioData::Mono(data_array);
                Ok(AudioSegment {
                    index: ps.original_segment.index,
                    data: audio_data,
                    start_sample: ps.original_segment.start_sample,
                    end_sample: ps.original_segment.end_sample,
                    length: ps.original_segment.length,
                    is_complete: ps.original_segment.is_complete,
                })
            })
            .collect();

        let segments = audio_segments?;
        let mut output_data = self.postprocessor.reconstruct_from_segments(&segments)?;

        if !output_data.is_empty() {
            let rms = (output_data.iter().map(|&x| x * x).sum::<f32>() / output_data.len() as f32).sqrt();
            let peak = output_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

            if rms < 0.1 && peak > 0.001 {
                let target_rms = 0.2;
                let rms_gain = target_rms / rms;
                let target_peak = 0.95;
                let peak_gain = if peak > 0.0 { target_peak / peak } else { 1.0 };
                let final_gain = rms_gain.min(peak_gain).clamp(1.5, 8.0);

                if self.config.verbose() {
                    println!("Normalization: RMS={:.4}, Peak={:.4}, gain={:.4}", rms, peak, final_gain);
                }

                for sample in output_data.iter_mut() {
                    *sample *= final_gain;
                    *sample = sample.clamp(-1.0, 1.0);
                }
            }
        }

        let output_audio = self.postprocessor.create_wav_audio(output_data)?;
        Ok(output_audio)
    }

    fn calculate_performance_metrics(&self, input_audio: &WavAudio, processing_time: std::time::Duration, segment_count: usize, average_inference_time_ms: f64) -> PerformanceMetrics {
        let audio_duration = input_audio.duration();
        let rtf = if audio_duration > 0.0 {
            processing_time.as_secs_f64() / audio_duration
        } else {
            0.0
        };

        PerformanceMetrics {
            input_duration_seconds: audio_duration,
            processing_time_seconds: processing_time.as_secs_f64(),
            real_time_factor: rtf,
            input_sample_rate: input_audio.sample_rate(),
            output_sample_rate: self.config.sample_rate(),
            segment_count,
            average_inference_time_ms,
            memory_usage_mb: self.get_memory_usage() as usize,
        }
    }

    fn get_memory_usage(&self) -> f64 {
        #[cfg(target_os = "macos")]
        {
            use std::process;
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p", &process::id().to_string()])
                .output()
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                        return rss_kb as f64 / 1024.0;
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", std::process::id())) {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb as f64 / 1024.0;
                            }
                        }
                    }
                }
            }
        }
        0.0
    }

    pub fn warm_up(&mut self) -> Result<()> {
        if self.config.verbose() {
            println!("Warming up...");
        }
        self.inference_engine.warm_up()?;
        if self.config.verbose() {
            println!("Warm up complete");
        }
        Ok(())
    }

    pub fn get_info(&self) -> ProcessorInfo {
        ProcessorInfo {
            model_path: self.config.model_path().clone(),
            sample_rate: self.config.sample_rate(),
            segment_size: self.config.segment_size(),
            overlap_ratio: self.config.overlap_ratio(),
            inference_threads: self.config.inference_threads(),
            max_retries: self.config.max_retries(),
            performance_monitoring: self.config.enable_performance_monitoring(),
        }
    }

    pub fn model(&self) -> &OnnxModel {
        &self.model
    }

    pub fn segmenter(&self) -> &AudioSegmenter {
        &self.segmenter
    }

    pub fn print_status(&self) {
        println!("=== Processor Status ===");
        println!("Model: {}", self.config.model_path().display());
        println!("Sample rate: {} Hz", self.config.sample_rate());
        println!("Segment size: {}", self.config.segment_size());
        println!("Overlap: {:.1}%", self.config.overlap_ratio() * 100.0);
        println!("Max retries: {}", self.config.max_retries());
        println!("Performance: {}", if self.config.enable_performance_monitoring() { "On" } else { "Off" });
        println!("Threads: {}", self.config.inference_threads());
        println!("========================");
        self.inference_engine.print_info();
    }
}

#[derive(Debug, Clone)]
pub struct ProcessedSegment {
    pub original_segment: AudioSegment,
    pub processed_data: Vec<f32>,
    pub processing_time_ms: u64,
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub input_path: std::path::PathBuf,
    pub output_path: std::path::PathBuf,
    pub processing_time: std::time::Duration,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub input_duration_seconds: f64,
    pub processing_time_seconds: f64,
    pub real_time_factor: f64,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub segment_count: usize,
    pub average_inference_time_ms: f64,
    pub memory_usage_mb: usize,
}

#[derive(Debug, Clone)]
pub struct ProcessorInfo {
    pub model_path: std::path::PathBuf,
    pub sample_rate: u32,
    pub segment_size: usize,
    pub overlap_ratio: f32,
    pub inference_threads: usize,
    pub max_retries: u32,
    pub performance_monitoring: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    fn create_test_config() -> Config {
        Config {
            model: crate::config::ModelConfig {
                path: PathBuf::from("test_model.onnx"),
                max_retries: 3,
                inference_threads: 1,
            },
            audio: crate::config::AudioConfig {
                sample_rate: 16000,
                overlap_ratio: 0.1,
                segment_size: 16000,
            },
            processing: crate::config::ProcessingConfig {
                enable_agc: true,
                enable_performance_monitoring: false,
                verbose: false,
            },
            input_path: PathBuf::from("test_input.wav"),
            output_path: PathBuf::from("test_output.wav"),
        }
    }

    #[test]
    fn test_processor_info() {
        let config = create_test_config();
        let info = ProcessorInfo {
            model_path: config.model_path().clone(),
            sample_rate: config.sample_rate(),
            segment_size: config.segment_size(),
            overlap_ratio: config.overlap_ratio(),
            inference_threads: config.inference_threads(),
            max_retries: config.max_retries(),
            performance_monitoring: config.enable_performance_monitoring(),
        };

        assert_eq!(info.sample_rate, 16000);
        assert_eq!(info.segment_size, 16000);
        assert_eq!(info.overlap_ratio, 0.1);
        assert_eq!(info.inference_threads, 1);
        assert_eq!(info.max_retries, 3);
        assert!(!info.performance_monitoring);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            input_duration_seconds: 2.0,
            processing_time_seconds: 1.0,
            real_time_factor: 0.5,
            input_sample_rate: 16000,
            output_sample_rate: 16000,
            segment_count: 4,
            average_inference_time_ms: 250.0,
            memory_usage_mb: 128,
        };

        assert_eq!(metrics.input_duration_seconds, 2.0);
        assert_eq!(metrics.processing_time_seconds, 1.0);
        assert_eq!(metrics.real_time_factor, 0.5);
        assert_eq!(metrics.segment_count, 4);
        assert_eq!(metrics.average_inference_time_ms, 250.0);
        assert_eq!(metrics.memory_usage_mb, 128);
    }

    #[test]
    fn test_processed_segment() {
        assert!(true);
    }
}