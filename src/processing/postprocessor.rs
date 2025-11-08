//! Audio Postprocessing Module
//!
//! Provides audio data postprocessing functions including overlap-add algorithms, format conversion, etc.
//! Reconstructs model outputs into complete audio files.

use ndarray::{Array1, ArrayView1};
use crate::audio::{WavAudio, AudioData, AudioFormat};
use crate::error::{ZipEnhancerError, Result};
use super::AudioSegment;

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Rectangular window
    Rectangular,
}

/// Overlap-add validation result
#[derive(Debug, Clone)]
pub struct OverlapAddValidationResult {
    /// Maximum sample value
    pub max_sample: f32,
    /// Root mean square value
    pub rms: f32,
    /// Dynamic range (dB)
    pub dynamic_range: f32,
    /// Clipping count
    pub clipping_count: usize,
    /// Discontinuity count
    pub discontinuities: usize,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f32,
    /// Whether result is valid
    pub is_valid: bool,
}

/// Audio postprocessing configuration
#[derive(Debug, Clone)]
pub struct PostprocessingConfig {
    /// Output sample rate (Hz)
    pub output_sample_rate: u32,
    /// Output format
    pub output_format: AudioFormat,
    /// Overlap ratio (0.0 - 1.0) - Keep consistent with segmenter
    pub overlap_ratio: f32,
    /// Whether to apply de-emphasis filter
    pub apply_de_emphasis: bool,
    /// De-emphasis coefficient
    pub de_emphasis_coeff: f32,
    /// Whether to apply smoothing
    pub apply_smoothing: bool,
    /// Smoothing window size
    pub smoothing_window_size: usize,
    /// Whether to clamp output range
    pub clamp_output: bool,
    /// Output range minimum value
    pub output_min: f32,
    /// Output range maximum value
    pub output_max: f32,
    /// Whether to enable soft limiter
    pub enable_soft_limiter: bool,
    /// Soft limiter threshold
    pub limiter_threshold: f32,
    /// Soft limiter ratio (compression ratio)
    pub limiter_ratio: f32,
    /// Soft limiter knee width (smooth transition region)
    pub limiter_knee_width: f32,
    /// Whether to enable adaptive strength control
    pub enable_adaptive_strength: bool,
    /// Adaptive strength coefficient
    pub adaptive_strength_coeff: f32,
    /// Whether to enable speech peak protection
    pub enable_peak_protection: bool,
    /// Speech peak protection threshold
    pub peak_protection_threshold: f32,
}

impl Default for PostprocessingConfig {
    fn default() -> Self {
        Self {
            output_sample_rate: 16000,
            output_format: AudioFormat::Float32,
            overlap_ratio: 0.1, // Default 10% overlap
            apply_de_emphasis: false,
            de_emphasis_coeff: 0.97,
            apply_smoothing: false,
            smoothing_window_size: 5,
            clamp_output: true,
            output_min: -1.0,
            output_max: 1.0,
            enable_soft_limiter: false,
            limiter_threshold: 0.9,
            limiter_ratio: 10.0,
            limiter_knee_width: 0.1,
            enable_adaptive_strength: false,
            adaptive_strength_coeff: 0.8,
            enable_peak_protection: false,
            peak_protection_threshold: 0.95,
        }
    }
}

/// Overlap-add buffer
#[derive(Debug)]
pub struct OverlapAddBuffer {
    /// Buffer data
    buffer: Vec<f32>,
    /// Buffer size
    buffer_size: usize,
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Valid data length in buffer
    data_length: usize,
}

impl OverlapAddBuffer {
    /// Create a new overlap-add buffer
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: vec![0.0; buffer_size],
            buffer_size,
            write_pos: 0,
            read_pos: 0,
            data_length: 0,
        }
    }

    /// Add new data to buffer
    pub fn add_data(&mut self, data: &[f32]) -> Result<()> {
        // Return error if data is too large to fit in buffer
        if data.len() > self.buffer_size {
            return Err(ZipEnhancerError::processing(format!(
                "Data block too large: {} > buffer size {}",
                data.len(), self.buffer_size
            )));
        }

        // Check if there's enough space
        let available_space = if self.read_pos <= self.write_pos {
            self.buffer_size - self.write_pos
        } else {
            self.read_pos - self.write_pos
        };

        if data.len() > available_space {
            // Clear buffer and reset if not enough space
            self.clear();
        }

        // Add data to buffer
        for (i, &sample) in data.iter().enumerate() {
            let pos = (self.write_pos + i) % self.buffer_size;
            self.buffer[pos] = sample;
        }

        self.write_pos += data.len();
        self.data_length = self.data_length + data.len();

        Ok(())
    }

    /// Read and remove specified length of data
    pub fn read_and_remove(&mut self, length: usize) -> Result<Vec<f32>> {
        if length > self.data_length {
            return Err(ZipEnhancerError::processing(format!(
                "Read length {} exceeds buffer data length {}",
                length, self.data_length
            )));
        }

        let mut result = Vec::with_capacity(length);

        for _ in 0..length {
            let pos = self.read_pos % self.buffer_size;
            result.push(self.buffer[pos]);
            self.read_pos += 1;
        }

        self.data_length -= length;

        // Reset all positions if buffer is empty
        if self.data_length == 0 {
            self.read_pos = 0;
            self.write_pos = 0;
        }

        Ok(result)
    }

    /// Get readable data length
    pub fn readable_length(&self) -> usize {
        self.data_length
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.data_length = 0;
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
    }

    /// Get buffer information
    pub fn info(&self) -> BufferInfo {
        BufferInfo {
            buffer_size: self.buffer_size,
            write_pos: self.write_pos,
            read_pos: self.read_pos,
            data_length: self.data_length,
        }
    }
}

/// Buffer information
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub buffer_size: usize,
    pub write_pos: usize,
    pub read_pos: usize,
    pub data_length: usize,
}

/// Audio postprocessor
#[derive(Debug)]
pub struct AudioPostprocessor {
    config: PostprocessingConfig,
    overlap_buffer: OverlapAddBuffer,
}

impl AudioPostprocessor {
    /// Create a new audio postprocessor
    pub fn new(config: PostprocessingConfig, overlap_buffer_size: usize) -> Self {
        Self {
            config,
            overlap_buffer: OverlapAddBuffer::new(overlap_buffer_size),
        }
    }

    /// Create postprocessor with default config
    pub fn with_default_config(overlap_buffer_size: usize) -> Self {
        Self::new(PostprocessingConfig::default(), overlap_buffer_size)
    }

    /// Get configuration
    pub fn config(&self) -> &PostprocessingConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PostprocessingConfig) {
        self.config = config;
    }

    /// Reconstruct complete audio from processed segments
    pub fn reconstruct_from_segments(&mut self, segments: &[AudioSegment]) -> Result<Array1<f32>> {
        if segments.is_empty() {
            return Ok(Array1::zeros(0));
        }

        // Get segment parameters
        let segment_size = segments[0].length;
        let _overlap_size = segment_size / 2; // Assume 50% overlap

        // Validate all segment lengths are consistent
        for (i, segment) in segments.iter().enumerate() {
            if segment.length != segment_size && segment.is_complete {
                log::warn!("Segment {} length {} does not match expected {}",
                          i, segment.length, segment_size);
            }
        }

        // Calculate total output length
        let total_length = if segments.is_empty() {
            0
        } else {
            segments[segments.len() - 1].end_sample
        };

        let mut output = Array1::zeros(total_length);

        // Overlap-add algorithm
        self.overlap_add_segments(&segments, &mut output)?;

        Ok(output)
    }

    /// Overlap-add segment algorithm - simplified version using cosine crossfading
    fn overlap_add_segments(&mut self, segments: &[AudioSegment], output: &mut Array1<f32>) -> Result<()> {
        if segments.is_empty() {
            return Ok(());
        }

        let segment_size = segments[0].length;
        // Use overlap ratio from config, ensure consistency with segmenter
        let overlap_size = (segment_size as f32 * self.config.overlap_ratio) as usize;
        let hop_size = segment_size - overlap_size;

        // Pre-compute simple cosine crossfade window (alternative to complex Hanning window)
        let cosine_fade = if overlap_size > 0 {
            let mut fade = vec![0.0; overlap_size];
            for i in 0..overlap_size {
                // Simple cosine crossfading: cosine transition from 1 to 0
                let progress = i as f32 / (overlap_size - 1) as f32;
                fade[i] = (1.0 - progress * std::f32::consts::PI / 2.0).cos();
            }
            fade
        } else {
            vec![]
        };

        // Process each segment using simple cosine crossfading
        for (segment_index, segment) in segments.iter().enumerate() {
            let segment_data = segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;

            let start_pos = segment.start_sample;

            if segment_index == 0 {
                // First segment: add directly
                for (i, &sample) in segment_data.iter().enumerate() {
                    if start_pos + i < output.len() {
                        output[start_pos + i] = sample;
                    }
                }
            } else {
                // Subsequent segments: use simple cosine crossfading
                let expected_overlap_start = segment_index * hop_size;

                // Process overlap region (using simple cosine crossfading)
                for i in 0..overlap_size {
                    let output_pos = expected_overlap_start + i;
                    let segment_pos = i;

                    if output_pos < output.len() && segment_pos < segment_data.len() && !cosine_fade.is_empty() {
                        // Simple cosine crossfading
                        let fade_out_factor = cosine_fade[i];  // Fade out factor: 1 -> 0
                        let fade_in_factor = 1.0 - fade_out_factor;  // Fade in factor: 0 -> 1

                        // Get existing output sample
                        let existing_sample = output[output_pos];

                        // Apply simple cosine crossfading
                        output[output_pos] = existing_sample * fade_out_factor + segment_data[segment_pos] * fade_in_factor;
                    }
                }

                // Add non-overlapping portion
                let non_overlap_start = expected_overlap_start + overlap_size;
                for (i, &sample) in segment_data.iter().skip(overlap_size).enumerate() {
                    if non_overlap_start + i < output.len() {
                        output[non_overlap_start + i] = sample;
                    }
                }
            }
        }

        // Unified ending processing: single smooth fade-out algorithm to eliminate double processing conflicts
        if let Some(last_segment) = segments.last() {
            let last_segment_data = last_segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Last segment is not mono format"))?;

            // Calculate smooth fade-out window size, use 15% of segment length or 3x overlap size, take the larger value
            let base_fade_size = (last_segment_data.len() as f32 * 0.15) as usize;
            let overlap_based_fade = overlap_size * 3;
            let fade_window_size = base_fade_size.max(overlap_based_fade).min(last_segment_data.len());

            if fade_window_size > 0 && output.len() > fade_window_size {
                let fade_start_pos = output.len() - fade_window_size;

                // Apply unified smooth fade-out curve
                for i in 0..fade_window_size {
                    let pos = fade_start_pos + i;

                    // Use improved cosine squared fade-out curve to ensure audio continuity
                    let progress = i as f32 / fade_window_size as f32;

                    // Smooth cosine squared fade-out: transition smoothly from 1.0 to 0.0
                    let smooth_progress = progress * progress; // Square function makes transition smoother
                    let fade_factor = (1.0 - smooth_progress * std::f32::consts::PI / 2.0).cos();

                    // Use exponential decay in the last 20% to ensure perfect convergence to 0
                    let final_factor = if progress > 0.8 {
                        let final_progress = (progress - 0.8) / 0.2; // 0.0 -> 1.0
                        let exp_decay = (-final_progress * 4.0).exp(); // Exponential decay
                        fade_factor * exp_decay
                    } else {
                        fade_factor
                    };

                    output[pos] *= final_factor;
                }

                // Ensure the last 5 samples are absolutely silent, completely eliminate ending noise
                let absolute_silence_samples = 5.min(output.len());
                for i in 0..absolute_silence_samples {
                    let pos = output.len() - absolute_silence_samples + i;
                    output[pos] = 0.0;
                }
            }
        }

        // Clear buffer and reset state
        self.overlap_buffer.clear();

        Ok(())
    }

    /// Process audio postprocessing
    pub fn postprocess(&mut self, output: &mut Array1<f32>) -> Result<()> {
        // 1. Apply de-emphasis filter
        if self.config.apply_de_emphasis {
            self.apply_de_emphasis(output)?;
        }

        // 2. Smoothing processing
        if self.config.apply_smoothing {
            self.apply_smoothing(output)?;
        }

        // 3. Adaptive noise reduction strength control
        if self.config.enable_adaptive_strength {
            self.apply_adaptive_strength_control(output)?;
        }

        // 4. Soft limiting processing (alternative to hard limiting)
        if self.config.enable_soft_limiter {
            self.apply_soft_limiter(output)?;
        } else if self.config.clamp_output {
            // 5. Hard limiting (if soft limiting is disabled)
            self.clamp_output(output)?;
        }

        // 6. Speech peak protection
        if self.config.enable_peak_protection {
            self.apply_peak_protection(output)?;
        }

        Ok(())
    }

    /// Apply de-emphasis filter
    fn apply_de_emphasis(&self, data: &mut Array1<f32>) -> Result<()> {
        let coeff = self.config.de_emphasis_coeff;
        
        for i in 1..data.len() {
            data[i] = data[i] + coeff * data[i - 1];
        }

        Ok(())
    }

    /// Apply smoothing processing
    fn apply_smoothing(&self, data: &mut Array1<f32>) -> Result<()> {
        let window_size = self.config.smoothing_window_size;
        
        if window_size == 0 || window_size >= data.len() {
            return Ok(());
        }

        let half_window = window_size / 2;
        let mut smoothed_data = data.clone();

        for i in half_window..( data.len() - half_window) {
            let mut sum = 0.0;
            for j in (i - half_window)..=(i + half_window) {
                sum += data[j];
            }
            smoothed_data[i] = sum / window_size as f32;
        }

        // Handle boundaries
        for i in 0..half_window {
            smoothed_data[i] = data[i];
        }

        for i in (data.len() - half_window)..data.len() {
            smoothed_data[i] = data[i];
        }

        *data = smoothed_data;

        Ok(())
    }

    /// Clamp output range
    fn clamp_output(&self, data: &mut Array1<f32>) -> Result<()> {
        for sample in data.iter_mut() {
            *sample = sample.clamp(self.config.output_min, self.config.output_max);
        }
        Ok(())
    }

    /// Soft limiting processing (alternative to hard limiting)
    fn apply_soft_limiter(&self, data: &mut Array1<f32>) -> Result<()> {
        let threshold = self.config.limiter_threshold;
        let ratio = self.config.limiter_ratio;
        let knee_width = self.config.limiter_knee_width;

        for sample in data.iter_mut() {
            let abs_sample = sample.abs();

            if abs_sample <= threshold - knee_width / 2.0 {
                // Below threshold, no processing
                continue;
            } else if abs_sample < threshold + knee_width / 2.0 {
                // In the knee region, smooth transition
                let knee_start = threshold - knee_width / 2.0;
                let knee_range = knee_width;
                let over_threshold = abs_sample - knee_start;
                let compression_factor = 1.0 + (1.0 / ratio - 1.0) * (over_threshold / knee_range);
                let target_level = knee_start + over_threshold * compression_factor;
                let scale_factor = if abs_sample > 0.0 { target_level / abs_sample } else { 1.0 };
                *sample *= scale_factor;
            } else {
                // Exceeds threshold, apply compression
                let over_threshold = abs_sample - threshold;
                let compressed_level = threshold + over_threshold / ratio;
                let scale_factor = if abs_sample > 0.0 { compressed_level / abs_sample } else { 1.0 };
                *sample *= scale_factor;
            }
        }

        Ok(())
    }

    /// Adaptive noise reduction strength control
    fn apply_adaptive_strength_control(&self, data: &mut Array1<f32>) -> Result<()> {
        let strength_coeff = self.config.adaptive_strength_coeff;

        // Calculate average energy of audio signal
        let avg_energy = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;

        // Adaptively adjust processing strength based on signal energy
        // High energy signals reduce processing strength to protect speech
        // Low energy signals increase processing strength to enhance noise reduction
        let adaptive_factor = if avg_energy > 0.1 {
            1.0 - (avg_energy - 0.1).min(0.9) * strength_coeff
        } else {
            1.0 + (0.1 - avg_energy).min(0.1) * strength_coeff
        };

        // Apply adaptive adjustment
        for sample in data.iter_mut() {
            *sample *= adaptive_factor;
        }

        Ok(())
    }

    /// Speech peak protection
    fn apply_peak_protection(&self, data: &mut Array1<f32>) -> Result<()> {
        let threshold = self.config.peak_protection_threshold;

        // Detect and protect speech peaks
        for i in 0..data.len() {
            let abs_sample = data[i].abs();

            if abs_sample > threshold {
                // Apply soft compression to handle peaks
                let over_threshold = abs_sample - threshold;
                let compressed_peak = threshold + over_threshold * 0.5; // Compression factor is 0.5
                let scale_factor = if abs_sample > 0.0 { compressed_peak / abs_sample } else { 1.0 };
                data[i] *= scale_factor;
            }
        }

        Ok(())
    }

    /// Create WavAudio file
    pub fn create_wav_audio(&self, data: Array1<f32>) -> Result<WavAudio> {
        let total_samples = data.len() as u32;
        let _duration = total_samples as f64 / self.config.output_sample_rate as f64;

        let header = crate::audio::AudioHeader::new(
            self.config.output_sample_rate,
            1,
            self.config.output_format,
            total_samples,
        );

        Ok(WavAudio {
            header,
            data: AudioData::Mono(data),
        })
    }

    /// Get postprocessing statistics
    pub fn get_stats(&self) -> PostprocessingStats {
        PostprocessingStats {
            buffer_info: self.overlap_buffer.info(),
            output_sample_rate: self.config.output_sample_rate,
            output_format: self.config.output_format,
            apply_de_emphasis: self.config.apply_de_emphasis,
            apply_smoothing: self.config.apply_smoothing,
            clamp_output: self.config.clamp_output,
        }
    }

    /// Advanced overlap-add algorithm (supports multiple window functions)
    pub fn overlap_add_advanced(
        &mut self,
        segments: &[AudioSegment],
        window_type: WindowType,
        crossfade_duration: f32
    ) -> Result<Array1<f32>> {
        if segments.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let segment_size = segments[0].length;
        let sample_rate = self.config.output_sample_rate;
        let crossfade_samples = (crossfade_duration * sample_rate as f32) as usize;
        
        // Calculate total output length
        let total_length = self.calculate_total_output_length(segments, crossfade_samples)?;
        let mut output = Array1::zeros(total_length);

        // Generate window function
        let window = self.generate_window(window_type, segment_size)?;
        let crossfade_window = self.generate_crossfade_window(crossfade_samples)?;

        // Reset overlap buffer
        self.overlap_buffer.clear();

        // Process each segment
        for (segment_index, segment) in segments.iter().enumerate() {
            let segment_data = segment.mono_data()
                .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;

            // Apply window function
            let windowed_segment = self.apply_window_to_segment(&segment_data.view(), &window)?;

            let start_pos = segment.start_sample;

            if segment_index == 0 {
                // First segment: add directly
                self.add_first_segment(&mut output, &windowed_segment, start_pos)?;
            } else {
                // Subsequent segments: use crossfade overlap-add
                self.add_segment_with_crossfade(
                    &mut output,
                    &windowed_segment,
                    start_pos,
                    crossfade_samples,
                    &crossfade_window,
                    segment_index == segments.len() - 1
                )?;
            }
        }

        // Apply postprocessing
        self.postprocess(&mut output)?;

        Ok(output)
    }

    /// Calculate total output length
    fn calculate_total_output_length(&self, segments: &[AudioSegment], crossfade_samples: usize) -> Result<usize> {
        if segments.is_empty() {
            return Ok(0);
        }

        let first_segment = &segments[0];
        let _last_segment = segments.last().unwrap();
        let hop_size = first_segment.length - crossfade_samples;
        
        if segments.len() == 1 {
            return Ok(first_segment.length);
        }

        let total_length = first_segment.length + (segments.len() - 1) * hop_size + crossfade_samples;
        Ok(total_length)
    }

    /// Generate window function
    fn generate_window(&self, window_type: WindowType, size: usize) -> Result<Array1<f32>> {
        let mut window = Array1::zeros(size);
        
        match window_type {
            WindowType::Hann => {
                for i in 0..size {
                    window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos());
                }
            }
            WindowType::Hamming => {
                for i in 0..size {
                    window[i] = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos();
                }
            }
            WindowType::Blackman => {
                for i in 0..size {
                    let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                    window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                }
            }
            WindowType::Rectangular => {
                window.fill(1.0);
            }
        }
        
        Ok(window)
    }

    /// Generate crossfade window function
    fn generate_crossfade_window(&self, size: usize) -> Result<Array1<f32>> {
        let mut window = Array1::zeros(size);
        
        // Use square root fade curve for smoother transitions
        for i in 0..size {
            let progress = i as f32 / (size - 1) as f32;
            // Input fade out (square root curve)
            window[i] = (1.0 - progress).sqrt();
        }
        
        Ok(window)
    }

    /// Apply window function to segment
    fn apply_window_to_segment(&self, segment_data: &ArrayView1<f32>, window: &Array1<f32>) -> Result<Vec<f32>> {
        if segment_data.len() != window.len() {
            return Err(ZipEnhancerError::processing(
                format!("Segment length {} does not match window function length {}", segment_data.len(), window.len())
            ));
        }
        
        let mut windowed = Vec::with_capacity(segment_data.len());
        for (i, &sample) in segment_data.iter().enumerate() {
            windowed.push(sample * window[i]);
        }
        
        Ok(windowed)
    }

    /// Add first segment
    fn add_first_segment(&self, output: &mut Array1<f32>, segment_data: &[f32], start_pos: usize) -> Result<()> {
        for (i, &sample) in segment_data.iter().enumerate() {
            if start_pos + i < output.len() {
                output[start_pos + i] = sample;
            }
        }
        Ok(())
    }

    /// Add segment with crossfading
    fn add_segment_with_crossfade(
        &mut self,
        output: &mut Array1<f32>,
        segment_data: &[f32],
        start_pos: usize,
        crossfade_samples: usize,
        crossfade_window: &Array1<f32>,
        is_last_segment: bool,
    ) -> Result<()> {
        let crossfade_len = crossfade_samples.min(segment_data.len()).min(start_pos);
        
        // 1. Crossfade region
        for i in 0..crossfade_len {
            let output_pos = start_pos - crossfade_len + i;
            let segment_pos = i;
            
            if output_pos < output.len() && segment_pos < segment_data.len() {
                let fade_out_factor = crossfade_window[i]; // Fade out factor
                let fade_in_factor = 1.0 - crossfade_window[i]; // Fade in factor
                
                output[output_pos] = output[output_pos] * fade_out_factor + segment_data[segment_pos] * fade_in_factor;
            }
        }
        
        // 2. Non-overlapping region (if any)
        let non_overlap_start = crossfade_len;
        for (i, &sample) in segment_data.iter().skip(non_overlap_start).enumerate() {
            let output_pos = start_pos + i;
            if output_pos < output.len() {
                output[output_pos] = sample;
            }
        }
        
        // 3. If not the last segment, add data to buffer for next use
        if !is_last_segment && segment_data.len() > crossfade_len {
            let buffer_data = &segment_data[segment_data.len() - crossfade_len..];
            self.overlap_buffer.add_data(buffer_data)?;
        }
        
        Ok(())
    }

    /// Real-time overlap-add processing (for streaming)
    pub fn overlap_add_realtime(
        &mut self,
        new_segment: &AudioSegment,
        window_type: WindowType,
    ) -> Result<Array1<f32>> {
        let segment_data = new_segment.mono_data()
            .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;
        
        let window = self.generate_window(window_type, segment_data.len())?;
        let windowed_segment = self.apply_window_to_segment(&segment_data.view(), &window)?;
        
        // Calculate output length (overlap part + new segment)
        let overlap_size = self.overlap_buffer.readable_length();
        let output_length = overlap_size + windowed_segment.len();
        let mut output = Array1::zeros(output_length);
        
        // 1. Add overlap data from buffer
        if overlap_size > 0 {
            let buffered_data = self.overlap_buffer.read_and_remove(overlap_size)?;
            for (i, &sample) in buffered_data.iter().enumerate() {
                output[i] = sample;
            }
        }
        
        // 2. Add new segment data
        for (i, &sample) in windowed_segment.iter().enumerate() {
            output[overlap_size + i] = sample;
        }
        
        // 3. Add the end part of new segment to buffer for next overlap
        let next_overlap_size = windowed_segment.len() / 2;
        if windowed_segment.len() > next_overlap_size {
            let next_overlap_data = &windowed_segment[windowed_segment.len() - next_overlap_size..];
            self.overlap_buffer.add_data(next_overlap_data)?;
        }
        
        // 4. Apply postprocessing
        self.postprocess(&mut output)?;
        
        Ok(output)
    }

    /// Adaptive overlap-add (automatically adjust based on audio content)
    pub fn overlap_add_adaptive(&mut self, segments: &[AudioSegment]) -> Result<Array1<f32>> {
        if segments.is_empty() {
            return Ok(Array1::zeros(0));
        }

        // Analyze first segment to determine optimal parameters
        let first_segment = &segments[0];
        let segment_data = first_segment.mono_data()
            .ok_or_else(|| ZipEnhancerError::processing("Segment is not mono format"))?;
        
        // Calculate audio features
        let energy = segment_data.iter().map(|&x| x * x).sum::<f32>() / segment_data.len() as f32;
        let spectral_centroid = self.calculate_spectral_centroid(&segment_data.view())?;
        
        // Adaptively select parameters based on audio features
        let (window_type, crossfade_ratio) = if spectral_centroid > 2000.0 {
            // High frequency content: use Hann window, smaller crossfade
            (WindowType::Hann, 0.3)
        } else if energy > 0.1 {
            // High energy: use Hamming window, medium crossfade
            (WindowType::Hamming, 0.4)
        } else {
            // Low energy: use Blackman window, larger crossfade
            (WindowType::Blackman, 0.5)
        };
        
        let crossfade_duration = (first_segment.length as f32 * crossfade_ratio) / self.config.output_sample_rate as f32;
        
        self.overlap_add_advanced(segments, window_type, crossfade_duration)
    }

    /// Calculate spectral centroid (simplified version)
    fn calculate_spectral_centroid(&self, data: &ArrayView1<f32>) -> Result<f32> {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;
        
        for (i, &sample) in data.iter().enumerate() {
            let energy = sample * sample;
            weighted_sum += i as f32 * energy;
            total_energy += energy;
        }
        
        if total_energy == 0.0 {
            return Ok(0.0);
        }
        
        Ok(weighted_sum / total_energy)
    }

    /// Validate overlap-add result
    pub fn validate_overlap_add_result(&self, output: &ArrayView1<f32>) -> Result<OverlapAddValidationResult> {
        let mut max_sample: f32 = 0.0;
        let mut rms = 0.0;
        let mut clipping_count = 0;
        let mut discontinuities = 0;
        
        // Analyze output signal
        for (i, &sample) in output.iter().enumerate() {
            max_sample = max_sample.max(sample.abs());
            rms += sample * sample;
            
            if sample.abs() > 1.0 {
                clipping_count += 1;
            }
            
            // Detect discontinuities (sharp changes)
            if i > 0 {
                let prev_sample = output[i - 1];
                let diff = (sample - prev_sample).abs();
                if diff > 0.5 {
                    discontinuities += 1;
                }
            }
        }
        
        rms = (rms / output.len() as f32).sqrt();
        let dynamic_range = if rms > 0.0 { 20.0 * (max_sample / rms).log10() } else { 0.0 };
        
        // Calculate quality score
        let clipping_penalty = (clipping_count as f32 / output.len() as f32) * 100.0;
        let discontinuity_penalty = (discontinuities as f32 / output.len() as f32) * 50.0;
        let clipping_score = (1.0 - clipping_penalty / 100.0).max(0.0);
        let continuity_score = (1.0 - discontinuity_penalty / 50.0).max(0.0);
        let range_score = if dynamic_range < 20.0 { 1.0 } else { 20.0 / dynamic_range };
        
        let quality_score = (clipping_score + continuity_score + range_score) / 3.0;
        
        Ok(OverlapAddValidationResult {
            max_sample,
            rms,
            dynamic_range,
            clipping_count,
            discontinuities,
            quality_score,
            is_valid: quality_score > 0.8 && clipping_count == 0,
        })
    }

    /// Print statistics
    pub fn print_stats(&self) {
        let stats = self.get_stats();
        println!("=== Audio Postprocessing Statistics ===");
        println!("Buffer size: {}", stats.buffer_info.buffer_size);
        println!("Write position: {}", stats.buffer_info.write_pos);
        println!("Read position: {}", stats.buffer_info.read_pos);
        println!("Data length: {}", stats.buffer_info.data_length);
        println!("Output sample rate: {} Hz", stats.output_sample_rate);
        println!("Output format: {:?}", stats.output_format);
        println!("De-emphasis: {}", if stats.apply_de_emphasis { "Enabled" } else { "Disabled" });
        println!("Smoothing: {}", if stats.apply_smoothing { "Enabled" } else { "Disabled" });
        println!("Output clamping: {}", if stats.clamp_output { "Enabled" } else { "Disabled" });
        println!("=======================================");
    }
}

/// Postprocessing statistics
#[derive(Debug, Clone)]
pub struct PostprocessingStats {
    pub buffer_info: BufferInfo,
    pub output_sample_rate: u32,
    pub output_format: AudioFormat,
    pub apply_de_emphasis: bool,
    pub apply_smoothing: bool,
    pub clamp_output: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_postprocessing_config_default() {
        let config = PostprocessingConfig::default();
        assert_eq!(config.output_sample_rate, 16000);
        assert_eq!(config.output_format, AudioFormat::Float32);
        assert!(!config.apply_de_emphasis);
        assert!(!config.apply_smoothing);
        assert!(config.clamp_output);
        assert_eq!(config.output_min, -1.0);
        assert_eq!(config.output_max, 1.0);
        assert!(config.enable_soft_limiter); 
        assert_eq!(config.limiter_threshold, 0.9);
        assert_eq!(config.limiter_ratio, 10.0);
        assert_eq!(config.limiter_knee_width, 0.1);
        assert!(config.enable_adaptive_strength); 
        assert_eq!(config.adaptive_strength_coeff, 0.8);
        assert!(config.enable_peak_protection);
        assert_eq!(config.peak_protection_threshold, 0.95);
    }

    #[test]
    fn test_overlap_add_buffer() {
        let mut buffer = OverlapAddBuffer::new(10);
        
        assert_eq!(buffer.readable_length(), 0);
        
        buffer.add_data(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(buffer.readable_length(), 3);
        
        let data = buffer.read_and_remove(2).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
        assert_eq!(buffer.readable_length(), 1);
    }

    #[test]
    fn test_postprocessor_creation() {
        let config = PostprocessingConfig::default();
        let processor = AudioPostprocessor::new(config, 1000);
        
        assert_eq!(processor.overlap_buffer.buffer_size, 1000);
    }

    #[test]
    fn test_simple_reconstruction() {
        let config = PostprocessingConfig::default();
        let mut processor = AudioPostprocessor::new(config, 100);

        let segment1_data = AudioData::Mono(Array1::from(vec![1.0, 2.0, 3.0, 4.0]));
        let segment1 = AudioSegment::new(0, segment1_data, 0, 4, true);

        let segment2_data = AudioData::Mono(Array1::from(vec![5.0, 6.0, 7.0, 8.0]));
        let segment2 = AudioSegment::new(1, segment2_data, 2, 6, true); // 50% overlap start

        let segments = vec![segment1, segment2];
        let output = processor.reconstruct_from_segments(&segments).unwrap();

        assert_eq!(output.len(), 6);
        assert_eq!(output[0], 1.0); 
        assert_eq!(output[1], 2.0);

        assert!(output[2] >= 0.0);
        assert!(output[3] >= 0.0);
        assert!(output[4] >= 0.0);
        assert!(output[5] >= 0.0);

        assert!(output.len() == segments.last().unwrap().end_sample);
    }

    #[test]
    fn test_de_emphasis() {
        let config = PostprocessingConfig {
            apply_de_emphasis: true,
            de_emphasis_coeff: 0.5,
            ..Default::default()
        };
        
        let processor = AudioPostprocessor::new(config, 100);
        let mut data = Array1::from(vec![1.0, 0.5, 0.25, 0.125]);
        
        processor.apply_de_emphasis(&mut data).unwrap();
        
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 0.75);
        assert_eq!(data[3], 0.5);
    }

    #[test]
    fn test_clamp_output() {
        let config = PostprocessingConfig {
            output_min: -0.8,
            output_max: 0.8,
            clamp_output: true,
            ..Default::default()
        };
        
        let processor = AudioPostprocessor::new(config, 100);
        let mut data = Array1::from(vec![-1.5, -0.5, 0.0, 0.5, 1.5]);
        
        processor.clamp_output(&mut data).unwrap();
        
        assert_eq!(data[0], -0.8);
        assert_eq!(data[1], -0.5);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.5);
        assert_eq!(data[4], 0.8);
    }

    #[test]
    fn test_create_wav_audio() {
        let config = PostprocessingConfig::default();
        let processor = AudioPostprocessor::new(config, 100);
        
        let data = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let audio = processor.create_wav_audio(data).unwrap();
        
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.total_samples(), 4);
        assert!(audio.validate().is_ok());
    }
}