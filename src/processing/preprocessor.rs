//! Audio Preprocessing Module
//!
//! Provides audio data preprocessing functions including format conversion, normalization, segmentation, etc.
//! Prepares suitable input data for ONNX model inference.

use ndarray::{Array1, Array2, ArrayD, ArrayView1};
use crate::audio::{WavAudio, AudioData, AudioFormat};
use crate::error::{ZipEnhancerError, Result};

/// Audio preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Target sample rate (Hz)
    pub target_sample_rate: u32,
    /// Target format
    pub target_format: AudioFormat,
    /// Segment size (number of samples)
    pub segment_size: usize,
    /// Overlap ratio (0.0 - 1.0)
    pub overlap_ratio: f32,
    /// Whether to perform normalization
    pub normalize: bool,
    /// Normalization target value
    pub normalize_target: f32,
    /// Whether to apply pre-emphasis filter
    pub pre_emphasis: bool,
    /// Pre-emphasis coefficient
    pub pre_emphasis_coeff: f32,
    /// Whether to apply window function
    pub apply_window: bool,
    /// Window function type
    pub window_type: WindowType,
  }

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            target_format: AudioFormat::Float32,
            segment_size: 16000,
            overlap_ratio: 0.1,
            normalize: false,
            normalize_target: 0.95,
            pre_emphasis: false,
            pre_emphasis_coeff: 0.97,
            apply_window: false,
            window_type: WindowType::Hamming,
        }
    }
}

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular window
    Rectangular,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
}

impl WindowType {
    /// Generate window function
    pub fn generate(&self, size: usize) -> Vec<f32> {
        match self {
            WindowType::Rectangular => vec![1.0; size],
            WindowType::Hann => self.hann_window(size),
            WindowType::Hamming => self.hamming_window(size),
            WindowType::Blackman => self.blackman_window(size),
        }
    }

    /// Hann window function
    fn hann_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let n = i as f32;
                let n_max = (size - 1) as f32;
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / n_max).cos())
            })
            .collect()
    }

    /// Hamming window function
    fn hamming_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let _n = i as f32;
                let n = (size - 1) as f32;
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * n / n).cos()
            })
            .collect()
    }

    /// Blackman window function
    fn blackman_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let _n = i as f32;
                let n = (size - 1) as f32;
                let term1 = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * n / n).cos();
                let term2 = 0.08 * (4.0 * std::f32::consts::PI * n / n).cos();
                term1 + term2
            })
            .collect()
    }
}

/// Audio segment
#[derive(Debug, Clone)]
pub struct AudioSegment {
    /// Segment index
    pub index: usize,
    /// Audio data (mono or stereo)
    pub data: AudioData,
    /// Start sample position
    pub start_sample: usize,
    /// End sample position
    pub end_sample: usize,
    /// Segment length
    pub length: usize,
    /// Whether it's a complete segment
    pub is_complete: bool,
}

impl AudioSegment {
    /// Create a new audio segment
    pub fn new(
        index: usize,
        data: AudioData,
        start_sample: usize,
        end_sample: usize,
        is_complete: bool,
    ) -> Self {
        let length = end_sample - start_sample;
        Self {
            index,
            data,
            start_sample,
            end_sample,
            length,
            is_complete,
        }
    }

    /// Get mono data
    pub fn mono_data(&self) -> Option<&Array1<f32>> {
        match &self.data {
            AudioData::Mono(data) => Some(data),
            AudioData::Stereo(_) => None,
        }
    }

    /// Get stereo data
    pub fn stereo_data(&self) -> Option<&Array2<f32>> {
        match &self.data {
            AudioData::Stereo(data) => Some(data),
            AudioData::Mono(_) => None,
        }
    }

    /// Convert to mono
    pub fn to_mono(&self) -> Self {
        let mono_data = match &self.data {
            AudioData::Mono(data) => data.clone(),
            AudioData::Stereo(data) => {
                // Mix left and right channels: (left + right) / 2
                data.mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty")
            }
        };

        Self {
            data: AudioData::Mono(mono_data),
            ..self.clone()
        }
    }

    /// Convert to stereo
    pub fn to_stereo(&self) -> Self {
        let stereo_data = match &self.data {
            AudioData::Mono(data) => {
                let len = data.len();
                let mut stereo = Array2::zeros((len, 2));
                stereo.column_mut(0).assign(&data);
                stereo.column_mut(1).assign(&data);
                stereo
            }
            AudioData::Stereo(data) => data.clone(),
        };

        Self {
            data: AudioData::Stereo(stereo_data),
            ..self.clone()
        }
    }

    /// Apply window function
    pub fn apply_window(&mut self, window: &[f32]) -> Result<()> {
        if window.len() != self.length {
            return Err(ZipEnhancerError::processing(format!(
                "Window function length ({}) does not match segment length ({})",
                window.len(),
                self.length
            )));
        }

        match &mut self.data {
            AudioData::Mono(data) => {
                for (i, sample) in data.iter_mut().enumerate() {
                    if i < window.len() {
                        *sample *= window[i];
                    }
                }
            }
            AudioData::Stereo(data) => {
                for mut row in data.rows_mut() {
                    for (i, sample) in row.iter_mut().enumerate() {
                        if i < window.len() {
                            *sample *= window[i];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Audio preprocessor
#[derive(Debug)]
pub struct AudioPreprocessor {
    config: PreprocessingConfig,
    window_cache: std::collections::HashMap<usize, Vec<f32>>,
}

impl AudioPreprocessor {
    /// Create a new audio preprocessor
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            window_cache: std::collections::HashMap::new(),
        }
    }

    /// Create preprocessor with default config
    pub fn with_default_config() -> Self {
        Self::new(PreprocessingConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PreprocessingConfig) {
        self.config = config;
        // Clear window cache as segment size may have changed
        self.window_cache.clear();
    }

    pub fn preprocess(&mut self, audio: &WavAudio) -> Result<Vec<AudioSegment>> {
        let converted_audio = self.convert_format(audio)?;

        let resampled_audio = if converted_audio.sample_rate() != self.config.target_sample_rate {
            self.resample(&converted_audio)?
        } else {
            converted_audio
        };

        let mono_audio = self.to_mono(&resampled_audio)?;

        let normalized_audio = if self.config.normalize {
            self.normalize(&mono_audio)?
        } else {
            mono_audio
        };

        let preemphasized_audio = if self.config.pre_emphasis {
            self.apply_pre_emphasis(&normalized_audio)?
        } else {
            normalized_audio
        };

        let segments = self.segment(&preemphasized_audio)?;
        let windowed_segments = self.apply_window_to_segments(segments)?;

        Ok(windowed_segments)
    }

    fn convert_format(&self, audio: &WavAudio) -> Result<WavAudio> {
        if audio.format() == self.config.target_format {
            return Ok(audio.clone());
        }

        let mut new_audio = audio.clone();
        new_audio.header.format = self.config.target_format;
        new_audio.header.bits_per_sample = self.config.target_format.bytes_per_sample() * 8;

        Ok(new_audio)
    }

    fn resample(&self, audio: &WavAudio) -> Result<WavAudio> {
        let ratio = self.config.target_sample_rate as f64 / audio.sample_rate() as f64;
        let new_length = (audio.data().len() as f64 * ratio) as usize;

        match audio.data() {
            AudioData::Mono(data) => {
                let new_data = self.resample_mono(data.view(), new_length, ratio)?;
                let mut new_audio = audio.clone();
                new_audio.header.sample_rate = self.config.target_sample_rate;
                new_audio.header.total_samples = new_length as u32;
                new_audio.header.duration = new_length as f64 / self.config.target_sample_rate as f64;
                new_audio.data = AudioData::Mono(new_data);
                Ok(new_audio)
            }
            AudioData::Stereo(data) => {
                let left = data.column(0);
                let right = data.column(1);
                let new_left = self.resample_mono(left, new_length, ratio)?;
                let new_right = self.resample_mono(right, new_length, ratio)?;

                let mut new_data = Array2::zeros((new_length, 2));
                new_data.column_mut(0).assign(&new_left);
                new_data.column_mut(1).assign(&new_right);

                let mut new_audio = audio.clone();
                new_audio.header.sample_rate = self.config.target_sample_rate;
                new_audio.header.total_samples = new_length as u32;
                new_audio.header.duration = new_length as f64 / self.config.target_sample_rate as f64;
                new_audio.data = AudioData::Stereo(new_data);
                Ok(new_audio)
            }
        }
    }

    fn resample_mono(&self, data: ArrayView1<f32>, new_length: usize, ratio: f64) -> Result<Array1<f32>> {
        if data.is_empty() {
            return Err(ZipEnhancerError::processing("Input data is empty"));
        }

        let mut new_data = Array1::zeros(new_length);
        let old_length = data.len();

        for i in 0..new_length {
            let old_pos = i as f64 / ratio;
            let old_index = old_pos.floor() as usize;
            let fraction = old_pos - old_index as f64;

            if old_index >= old_length - 1 {
                new_data[i] = data[old_length - 1];
            } else {
                let sample1 = data[old_index];
                let sample2 = data[old_index + 1];
                new_data[i] = sample1 + (sample2 - sample1) * fraction as f32;
            }
        }

        Ok(new_data)
    }

    fn to_mono(&self, audio: &WavAudio) -> Result<WavAudio> {
        match audio.data() {
            AudioData::Mono(_) => Ok(audio.clone()),
            AudioData::Stereo(data) => {
                let mono = data.mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty");
                let mut new_audio = audio.clone();
                new_audio.header.channels = 1;
                new_audio.data = AudioData::Mono(mono);
                Ok(new_audio)
            }
        }
    }

    fn normalize(&self, audio: &WavAudio) -> Result<WavAudio> {
        let mut new_audio = audio.clone();

        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

                if max_val > 0.0 {
                    let scale = self.config.normalize_target / max_val;
                    let limited_scale = if scale > 10.0 { 10.0 } else { scale };

                    for sample in data.iter_mut() {
                        *sample *= limited_scale;
                        *sample = sample.clamp(-self.config.normalize_target, self.config.normalize_target);
                    }
                }
            }
            AudioData::Stereo(data) => {
                for mut channel in data.columns_mut() {
                    let max_val = channel.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

                    if max_val > 0.0 {
                        let scale = self.config.normalize_target / max_val;
                        let limited_scale = if scale > 10.0 { 10.0 } else { scale };

                        for sample in channel.iter_mut() {
                            *sample *= limited_scale;
                            *sample = sample.clamp(-self.config.normalize_target, self.config.normalize_target);
                        }
                    }
                }
            }
        }

        Ok(new_audio)
    }

    /// Apply pre-emphasis filter
    fn apply_pre_emphasis(&self, audio: &WavAudio) -> Result<WavAudio> {
        let coeff = self.config.pre_emphasis_coeff;
        
        let mut new_audio = audio.clone();
        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                for i in 1..data.len() {
                    data[i] = data[i] - coeff * data[i - 1];
                }
            }
            AudioData::Stereo(data) => {
                for mut row in data.rows_mut() {
                    for i in 1..row.len() {
                        row[i] = row[i] - coeff * row[i - 1];
                    }
                }
            }
        }

        Ok(new_audio)
    }

    /// Audio segmentation
    fn segment(&self, audio: &WavAudio) -> Result<Vec<AudioSegment>> {
        let data = audio.data();
        let total_samples = data.len();
        let segment_size = self.config.segment_size;
        let hop_size = (segment_size as f32 * (1.0 - self.config.overlap_ratio)) as usize;

        if segment_size == 0 {
            return Err(ZipEnhancerError::processing("Segment size cannot be 0"));
        }

        if hop_size == 0 {
            return Err(ZipEnhancerError::processing("Hop size cannot be 0"));
        }

        let mut segments = Vec::new();
        let mut segment_index = 0;

        for start in (0..total_samples).step_by(hop_size) {
            let end = (start + segment_size).min(total_samples);
            let length = end - start;
            let is_complete = length == segment_size;

            let segment_data = match data {
                AudioData::Mono(data) => {
                    AudioData::Mono(data.slice(ndarray::s![start..end]).to_owned())
                }
                AudioData::Stereo(data) => {
                    AudioData::Stereo(data.slice(ndarray::s![start..end, ..]).to_owned())
                }
            };

            let segment = AudioSegment::new(segment_index, segment_data, start, end, is_complete);
            segments.push(segment);
            segment_index += 1;
        }

        Ok(segments)
    }

    /// Apply window function to all segments
    fn apply_window_to_segments(&mut self, segments: Vec<AudioSegment>) -> Result<Vec<AudioSegment>> {
        let segment_size = self.config.segment_size;
        
        // Get or generate window function
        let window = self.get_window(segment_size)?;

        let mut processed_segments = Vec::with_capacity(segments.len());

        for segment in segments {
            let mut processed_segment = segment;
            
            // Only apply window function to complete segments
            if processed_segment.is_complete {
                processed_segment.apply_window(&window)?;
            }
            
            processed_segments.push(processed_segment);
        }

        Ok(processed_segments)
    }

    /// Get window function (with caching)
    fn get_window(&mut self, size: usize) -> Result<Vec<f32>> {
        if let Some(window) = self.window_cache.get(&size) {
            Ok(window.clone())
        } else {
            let window = self.config.window_type.generate(size);
            self.window_cache.insert(size, window.clone());
            Ok(window)
        }
    }

    /// Convert segments to model input format
    pub fn segments_to_model_input(&self, segments: &[AudioSegment]) -> Result<Vec<ArrayD<f32>>> {
        let mut inputs = Vec::with_capacity(segments.len());

        for segment in segments {
            let audio_data = segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;

            // Convert to model expected shape [batch_size, sequence_length]
            let input_data = audio_data.to_vec();
            let input_array = ArrayD::from_shape_vec(vec![1, audio_data.len()], input_data)
                .map_err(|e| ZipEnhancerError::processing(format!("Failed to create model input: {}", e)))?;

            inputs.push(input_array);
        }

        Ok(inputs)
    }

    /// Convert model output back to audio segments
    pub fn model_output_to_segments(&self, outputs: &[ArrayD<f32>], original_segments: &[AudioSegment]) -> Result<Vec<AudioSegment>> {
        if outputs.len() != original_segments.len() {
            return Err(ZipEnhancerError::processing(format!(
                "Model output count ({}) does not match original segment count ({})",
                outputs.len(),
                original_segments.len()
            )));
        }

        let mut result_segments = Vec::with_capacity(outputs.len());

        for (i, output) in outputs.iter().enumerate() {
            let original_segment = &original_segments[i];
            
            // Assume output shape is [1, sequence_length]
            let output_data = output.as_slice()
                .ok_or_else(|| ZipEnhancerError::processing("Output data is not contiguous"))?
                .to_vec();
            
            if output_data.len() != original_segment.length {
                return Err(ZipEnhancerError::processing(format!(
                    "Output length ({}) does not match segment length ({})",
                    output_data.len(),
                    original_segment.length
                )));
            }

            let segment_data = AudioData::Mono(Array1::from(output_data));
            let result_segment = AudioSegment::new(
                original_segment.index,
                segment_data,
                original_segment.start_sample,
                original_segment.end_sample,
                original_segment.is_complete,
            );

            result_segments.push(result_segment);
        }

        Ok(result_segments)
    }

    /// Preprocess and segment audio
    pub fn preprocess_and_segment(&self, audio: &WavAudio) -> Result<Vec<AudioSegment>> {
        // Ensure it's mono
        if audio.channels() != 1 {
            return Err(ZipEnhancerError::processing("Input audio must be mono"));
        }

        // Ensure sample rate is correct
        if audio.sample_rate() != self.config.target_sample_rate {
            return Err(ZipEnhancerError::processing(format!(
                "Input sample rate {} does not match target sample rate {}",
                audio.sample_rate(), self.config.target_sample_rate
            )));
        }

        // Segment processing
        let segments = self.segment(audio)?;

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioData;
  
    #[test]
    fn test_preprocessing_config_default() {
        let config = PreprocessingConfig::default();
        assert_eq!(config.target_sample_rate, 16000);
        assert_eq!(config.target_format, AudioFormat::Float32);
        assert_eq!(config.segment_size, 8000); 
        assert_eq!(config.overlap_ratio, 0.5); 
        assert!(config.normalize); 
        assert_eq!(config.normalize_target, 0.95);
        assert_eq!(config.window_type, WindowType::Hamming); 
    }

    #[test]
    fn test_window_generation() {
        let rect_window = WindowType::Rectangular.generate(5);
        assert_eq!(rect_window, vec![1.0, 1.0, 1.0, 1.0, 1.0]);

        let hann_window = WindowType::Hann.generate(5);
        assert!(hann_window.len() == 5);
        assert!(hann_window[0] == 0.0);
        assert!(hann_window[2] == 1.0);
        assert!(hann_window[4] == 0.0);
    }

    #[test]
    fn test_audio_segment_creation() {
        let data = AudioData::Mono(Array1::from(vec![0.1, 0.2, 0.3, 0.4]));
        let segment = AudioSegment::new(0, data, 0, 4, true);
        
        assert_eq!(segment.index, 0);
        assert_eq!(segment.start_sample, 0);
        assert_eq!(segment.end_sample, 4);
        assert_eq!(segment.length, 4);
        assert!(segment.is_complete);
        assert_eq!(segment.mono_data().unwrap().len(), 4);
    }

    #[test]
    fn test_segment_conversion() {
        let mono_data = AudioData::Mono(Array1::from(vec![0.1, 0.2, 0.3, 0.4]));
        let mono_segment = AudioSegment::new(0, mono_data, 0, 4, true);
        
        let stereo_segment = mono_segment.to_stereo();
        assert!(stereo_segment.stereo_data().is_some());
        assert_eq!(stereo_segment.stereo_data().unwrap().nrows(), 4);
        
        let back_to_mono = stereo_segment.to_mono();
        assert!(back_to_mono.mono_data().is_some());
    }

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessingConfig::default();
        let preprocessor = AudioPreprocessor::new(config);
        assert!(preprocessor.window_cache.is_empty());
    }

    #[test]
    fn test_window_application() {
        let config = PreprocessingConfig::default();
        let _preprocessor = AudioPreprocessor::new(config);
        
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let mut segment = AudioSegment::new(0, AudioData::Mono(data), 0, 4, true);
        
        let window = WindowType::Hann.generate(4);
        segment.apply_window(&window).unwrap();
        
        let expected = vec![0.0, 1.5, 2.25, 0.0]; 
        let actual = segment.mono_data().unwrap().to_vec();
        
        for (i, &val) in actual.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simple_segmentation() {
        let config = PreprocessingConfig {
            segment_size: 4,
            overlap_ratio: 0.5,
            ..Default::default()
        };
        
        let preprocessor = AudioPreprocessor::new(config);
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let audio = WavAudio::new_mono(16000, data.clone(), AudioFormat::Float32);

        let segments = preprocessor.segment(&audio).unwrap();
        
        assert_eq!(segments.len(), 4);

        assert_eq!(segments[0].length, 4);
        assert_eq!(segments[1].length, 4);
        assert_eq!(segments[2].length, 3); 
        assert_eq!(segments[3].length, 1);
    }

    #[test]
    fn test_segments_to_model_input() {
        let config = PreprocessingConfig::default();
        let preprocessor = AudioPreprocessor::new(config);
        
        let data = AudioData::Mono(Array1::from(vec![0.1, 0.2, 0.3, 0.4]));
        let segment = AudioSegment::new(0, data, 0, 4, true);
        let segments = vec![segment];
        
        let inputs = preprocessor.segments_to_model_input(&segments).unwrap();
        
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].shape(), &[1, 4]);
        assert_eq!(inputs[0].as_slice().unwrap(), &[0.1, 0.2, 0.3, 0.4]);
    }
}