//! Audio Segmentation Module
//!
//! Provides audio data segmentation functionality, supporting fixed-length and variable-length segmentation.
//! Includes overlap handling and boundary processing logic.

use ndarray::{Array1, ArrayView1};
use crate::audio::{AudioData, WavAudio};
use crate::error::{ZipEnhancerError, Result};

/// Segmentation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentationStrategy {
    /// Fixed-length segmentation
    FixedLength,
    /// Variable-length segmentation (based on silence detection)
    VariableLength,
    /// Voice activity detection segmentation
    VoiceActivityDetection,
}

/// Segmentation configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Segmentation strategy
    pub strategy: SegmentationStrategy,
    /// Segment size (number of samples)
    pub segment_size: usize,
    /// Overlap ratio (0.0 - 1.0)
    pub overlap_ratio: f32,
    /// Minimum segment length (for variable length only)
    pub min_segment_length: usize,
    /// Maximum segment length (for variable length only)
    pub max_segment_length: usize,
    /// Silence detection threshold
    pub silence_threshold: f32,
    /// Voice activity detection parameters
    pub vad_config: Option<VADConfig>,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            strategy: SegmentationStrategy::FixedLength,
            segment_size: 16000,
            overlap_ratio: 0.1,
            min_segment_length: 8000,
            max_segment_length: 32000,
            silence_threshold: 0.01,
            vad_config: None,
        }
    }
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone)]
pub struct VADConfig {
    /// Energy calculation window size
    pub energy_window_size: usize,
    /// Energy threshold
    pub energy_threshold: f32,
    /// Zero crossing rate threshold
    pub zero_crossing_threshold: f32,
    /// Minimum speech length
    pub min_speech_length: usize,
    /// Minimum silence length
    pub min_silence_length: usize,
}

impl Default for VADConfig {
    fn default() -> Self {
        Self {
            energy_window_size: 400,
            energy_threshold: 0.01,
            zero_crossing_threshold: 10.0,
            min_speech_length: 1600,
            min_silence_length: 800,
        }
    }
}

/// Segment boundary types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentBoundary {
    /// Start boundary
    Start,
    /// End boundary
    End,
    /// Speech boundary
    Speech,
    /// Silence boundary
    Silence,
}

/// Segment boundary information
#[derive(Debug, Clone)]
pub struct SegmentBoundaryInfo {
    /// Boundary position
    pub position: usize,
    /// Boundary type
    pub boundary_type: SegmentBoundary,
    /// Confidence level
    pub confidence: f32,
}

/// Audio segmenter
#[derive(Debug)]
pub struct AudioSegmenter {
    config: SegmentationConfig,
    vad_enabled: bool,
}

impl AudioSegmenter {
    /// Create a new audio segmenter
    pub fn new(config: SegmentationConfig) -> Self {
        let vad_enabled = config.vad_config.is_some();
        
        Self {
            config,
            vad_enabled,
        }
    }

    /// Create a segmenter with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SegmentationConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &SegmentationConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: SegmentationConfig) {
        self.vad_enabled = config.vad_config.is_some();
        self.config = config;
    }

    /// Segment the audio
    pub fn segment(&self, audio: &WavAudio) -> Result<Vec<SegmentInfo>> {
        let data = audio.data();
        let _total_samples = data.len();

        match self.config.strategy {
            SegmentationStrategy::FixedLength => {
                self.segment_fixed_length(audio)
            }
            SegmentationStrategy::VariableLength => {
                self.segment_variable_length(audio)
            }
            SegmentationStrategy::VoiceActivityDetection => {
                if self.vad_enabled {
                    self.segment_with_vad(audio)
                } else {
                    self.segment_fixed_length(audio)
                }
            }
        }
    }

    /// Fixed-length segmentation
    fn segment_fixed_length(&self, audio: &WavAudio) -> Result<Vec<SegmentInfo>> {
        let data = audio.data();
        let total_samples = data.len();
        let segment_size = self.config.segment_size;
        let hop_size = (segment_size as f32 * (1.0 - self.config.overlap_ratio)) as usize;

        if segment_size == 0 {
            return Err(ZipEnhancerError::processing("Segment size cannot be 0"));
        }

        let mut segments = Vec::new();

        for (index, start) in (0..total_samples).step_by(hop_size).enumerate() {
            let end = (start + segment_size).min(total_samples);
            let length = end - start;
            
            let info = SegmentInfo {
                index,
                start_sample: start,
                end_sample: end,
                length,
                segment_type: if length == segment_size {
                    SegmentType::Complete
                } else {
                    SegmentType::Partial
                },
                boundaries: vec![
                    SegmentBoundaryInfo {
                        position: start,
                        boundary_type: SegmentBoundary::Start,
                        confidence: 1.0,
                    },
                    SegmentBoundaryInfo {
                        position: end,
                        boundary_type: SegmentBoundary::End,
                        confidence: 1.0,
                    },
                ],
                energy: None,
                zero_crossings: None,
            };

            segments.push(info);
        }

        Ok(segments)
    }

    /// Variable-length segmentation
    fn segment_variable_length(&self, audio: &WavAudio) -> Result<Vec<SegmentInfo>> {
        let data = audio.data();
        let audio_array_owned: Array1<f32> = match data {
            AudioData::Mono(data) => data.to_owned(),
            AudioData::Stereo(data) => {
                // Mix stereo to mono
                data.mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty")
            }
        };
        let audio_array = audio_array_owned.view();

        // Use silence detection to find segment boundaries
        let boundaries = self.find_silence_boundaries(&audio_array)?;
        
        if boundaries.len() < 2 {
            // No silence boundaries found, entire audio as one segment
            let info = SegmentInfo {
                index: 0,
                start_sample: 0,
                end_sample: data.len(),
                length: data.len(),
                segment_type: SegmentType::Complete,
                boundaries: vec![
                    SegmentBoundaryInfo {
                        position: 0,
                        boundary_type: SegmentBoundary::Start,
                        confidence: 1.0,
                    },
                    SegmentBoundaryInfo {
                        position: data.len(),
                        boundary_type: SegmentBoundary::End,
                        confidence: 1.0,
                    },
                ],
                energy: None,
                zero_crossings: None,
            };
            
            return Ok(vec![info]);
        }

        // Create segments based on silence boundaries
        let mut segments = Vec::new();
        
        for (i, window) in boundaries.windows(2).enumerate() {
            let start = window[0].position;
            let end = window[1].position;
            
            let length = end - start;
            
            // Check if segment length is within allowed range
            if length < self.config.min_segment_length {
                log::warn!("Segment {} length {} is less than minimum length {}, skipping",
                          i, length, self.config.min_segment_length);
                continue;
            }
            
            if length > self.config.max_segment_length {
                log::warn!("Segment {} length {} exceeds maximum length {}, truncating",
                          i, length, self.config.max_segment_length);
                
                // Truncate to maximum length
                let truncated_end = start + self.config.max_segment_length;
                let truncated_length = self.config.max_segment_length;
                
                let info = SegmentInfo {
                    index: segments.len(),
                    start_sample: start,
                    end_sample: truncated_end,
                    length: truncated_length,
                    segment_type: SegmentType::Truncated,
                    boundaries: vec![
                        SegmentBoundaryInfo {
                            position: start,
                            boundary_type: SegmentBoundary::Start,
                            confidence: window[0].confidence,
                        },
                        SegmentBoundaryInfo {
                            position: truncated_end,
                            boundary_type: SegmentBoundary::End,
                            confidence: window[1].confidence,
                        },
                    ],
                    energy: None,
                    zero_crossings: None,
                };
                
                segments.push(info);
                continue;
            }
            
            let info = SegmentInfo {
                index: segments.len(),
                start_sample: start,
                end_sample: end,
                length,
                segment_type: SegmentType::Variable,
                boundaries: vec![
                    SegmentBoundaryInfo {
                        position: start,
                        boundary_type: SegmentBoundary::Start,
                        confidence: window[0].confidence,
                    },
                    SegmentBoundaryInfo {
                        position: end,
                        boundary_type: SegmentBoundary::End,
                        confidence: window[1].confidence,
                    },
                ],
                energy: None,
                zero_crossings: None,
            };
            
            segments.push(info);
        }

        Ok(segments)
    }

    /// Voice activity detection segmentation
    fn segment_with_vad(&self, audio: &WavAudio) -> Result<Vec<SegmentInfo>> {
        let data = audio.data();
        let vad_config = self.config.vad_config.as_ref()
            .cloned()
            .unwrap_or_else(VADConfig::default);

        // Extract audio data
        let audio_array_owned: Array1<f32> = match data {
            AudioData::Mono(data) => data.to_owned(),
            AudioData::Stereo(data) => {
                // Mix stereo to mono
                data.mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty")
            }
        };
        let audio_array = audio_array_owned.view();

        // Analyze voice activity
        let activity_levels = self.analyze_voice_activity(&audio_array, &vad_config)?;
        
        // Find segment boundaries based on voice activity levels
        let boundaries = self.find_activity_boundaries(&activity_levels)?;
        
        // Create segments
        let mut segments = Vec::new();

        for (_i, window) in boundaries.windows(2).enumerate() {
            let start = window[0].position;
            let end = window[1].position;
            
            let length = end - start;
            
            // Ensure segment meets minimum speech length requirement
            if length >= vad_config.min_speech_length {
                let info = SegmentInfo {
                    index: segments.len(),
                    start_sample: start,
                    end_sample: end,
                    length,
                    segment_type: SegmentType::Speech,
                    boundaries: vec![
                        SegmentBoundaryInfo {
                            position: start,
                            boundary_type: SegmentBoundary::Speech,
                            confidence: window[0].confidence,
                        },
                        SegmentBoundaryInfo {
                            position: end,
                            boundary_type: SegmentBoundary::Speech,
                            confidence: window[1].confidence,
                        },
                    ],
                    energy: None,
                    zero_crossings: None,
                };
                
                segments.push(info);
            }
        }

        // Process silence segments
        for i in 0..boundaries.len() - 1 {
            let current_end = boundaries[i].position;
            let next_start = boundaries[i + 1].position;
            
            if next_start > current_end {
                let silence_length = next_start - current_end;
                
                if silence_length >= vad_config.min_silence_length {
                    let info = SegmentInfo {
                        index: segments.len(),
                        start_sample: current_end,
                        end_sample: next_start,
                        length: silence_length,
                        segment_type: SegmentType::Silence,
                        boundaries: vec![
                            SegmentBoundaryInfo {
                                position: current_end,
                                boundary_type: if i == 0 { SegmentBoundary::Start } else { SegmentBoundary::Silence },
                                confidence: boundaries[i].confidence,
                            },
                            SegmentBoundaryInfo {
                                position: next_start,
                                boundary_type: if i == boundaries.len() - 2 { SegmentBoundary::End } else { SegmentBoundary::Silence },
                                confidence: boundaries[i + 1].confidence,
                            },
                        ],
                        energy: None,
                        zero_crossings: None,
                    };
                    
                    segments.push(info);
                }
            }
        }

        Ok(segments)
    }

    /// Analyze voice activity levels
    fn analyze_voice_activity(&self, data: &ArrayView1<f32>, config: &VADConfig) -> Result<Vec<ActivityLevel>> {
        let window_size = config.energy_window_size;
        let hop_size = window_size / 2;
        
        let mut activity_levels = Vec::new();
        
        for start in (0..data.len()).step_by(hop_size) {
            let end = (start + window_size).min(data.len());

            // Skip if we don't have any data at all
            if end - start == 0 {
                break;
            }
            
            let window_data = data.slice(ndarray::s![start..end]);
            
            let energy = self.calculate_energy(&window_data);
            let zero_crossings = self.count_zero_crossings(&window_data);
            
            // Simple VAD decision: based on energy and zero-crossing rate
            let is_speech = energy > config.energy_threshold || zero_crossings > config.zero_crossing_threshold;
            
            let confidence = if is_speech {
                (energy + zero_crossings / 1000.0).min(1.0)
            } else {
                0.0
            };
            
            activity_levels.push(ActivityLevel {
                position: start + hop_size / 2, // Window center
                energy,
                zero_crossings,
                is_speech,
                confidence,
            });
        }

        Ok(activity_levels)
    }

    /// Calculate energy
    fn calculate_energy(&self, data: &ArrayView1<f32>) -> f32 {
        data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32
    }

    /// Calculate zero-crossing rate
    fn count_zero_crossings(&self, data: &ArrayView1<f32>) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }

        let mut count = 0;
        let mut previous = data[0];
        
        for &current in data.iter().skip(1) {
            if (previous < 0.0 && current >= 0.0) || (previous > 0.0 && current <= 0.0) {
                count += 1;
            }
            previous = current;
        }

        count as f32
    }

    /// Find boundaries based on silence detection
    fn find_silence_boundaries(&self, data: &ArrayView1<f32>) -> Result<Vec<SegmentBoundaryInfo>> {
        let window_size = self.config.min_segment_length.min(1000); // Use smaller window for silence detection
        
        let mut boundaries = Vec::new();
        let in_silence = false;
        let mut _silence_start = 0;
        let _current_confidence = 1.0;
        
        boundaries.push(SegmentBoundaryInfo {
            position: 0,
            boundary_type: SegmentBoundary::Start,
            confidence: 1.0,
        });

        for (i, window) in data.windows(window_size).into_iter().enumerate() {
            let energy = self.calculate_energy(&window);
            
            let is_current_silent = energy < self.config.silence_threshold;
            
            if in_silence != is_current_silent {
                let current_confidence = 0.9; // Reduce boundary confidence
                if is_current_silent {
                    _silence_start = i;
                } else {
                    boundaries.push(SegmentBoundaryInfo {
                        position: i,
                        boundary_type: SegmentBoundary::End,
                        confidence: current_confidence,
                    });
                    boundaries.push(SegmentBoundaryInfo {
                        position: i,
                        boundary_type: SegmentBoundary::Start,
                        confidence: current_confidence,
                    });
                }
            }
        }

        boundaries.push(SegmentBoundaryInfo {
            position: data.len(),
            boundary_type: SegmentBoundary::End,
            confidence: 1.0,
        });

        Ok(boundaries)
    }

    /// Find boundaries based on voice activity levels
    fn find_activity_boundaries(&self, activity_levels: &[ActivityLevel]) -> Result<Vec<SegmentBoundaryInfo>> {
        let mut boundaries = Vec::new();
        let mut previous_is_speech = false;
        
        boundaries.push(SegmentBoundaryInfo {
            position: 0,
            boundary_type: SegmentBoundary::Start,
            confidence: 1.0,
        });

        for (_i, level) in activity_levels.iter().enumerate() {
            if previous_is_speech != level.is_speech {
                boundaries.push(SegmentBoundaryInfo {
                    position: level.position,
                    boundary_type: if level.is_speech { SegmentBoundary::Speech } else { SegmentBoundary::Silence },
                    confidence: level.confidence,
                });
                previous_is_speech = level.is_speech;
            }
        }

        boundaries.push(SegmentBoundaryInfo {
            position: activity_levels.last()
                .map(|l| l.position)
                .unwrap_or(activity_levels.len() * 100),
            boundary_type: SegmentBoundary::End,
            confidence: 1.0,
        });

        Ok(boundaries)
    }
}

/// Activity level information
#[derive(Debug, Clone)]
pub struct ActivityLevel {
    /// Position (sample index)
    pub position: usize,
    /// Energy
    pub energy: f32,
    /// Zero-crossing rate
    pub zero_crossings: f32,
    /// Whether it's speech
    pub is_speech: bool,
    /// Confidence level
    pub confidence: f32,
}

/// Segment information
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment index
    pub index: usize,
    /// Start sample position
    pub start_sample: usize,
    /// End sample position
    pub end_sample: usize,
    /// Segment length
    pub length: usize,
    /// Segment type
    pub segment_type: SegmentType,
    /// Boundary information
    pub boundaries: Vec<SegmentBoundaryInfo>,
    /// Average energy
    pub energy: Option<f32>,
    /// Average zero-crossing rate
    pub zero_crossings: Option<f32>,
}

/// Segment types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// Complete segment
    Complete,
    /// Partial segment
    Partial,
    /// Variable-length segment
    Variable,
    /// Speech segment
    Speech,
    /// Silence segment
    Silence,
    /// Truncated segment
    Truncated,
}

impl SegmentInfo {
    /// Get segment center position
    pub fn center(&self) -> usize {
        self.start_sample + self.length / 2
    }

    /// Get segment duration (seconds)
    pub fn duration(&self, sample_rate: u32) -> f64 {
        self.length as f64 / sample_rate as f64
    }

    /// Check if segment is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.segment_type, SegmentType::Complete | SegmentType::Speech | SegmentType::Silence)
    }

    /// Check if segment is speech
    pub fn is_speech(&self) -> bool {
        matches!(self.segment_type, SegmentType::Speech)
    }

    /// Check if segment is silence
    pub fn is_silence(&self) -> bool {
        matches!(self.segment_type, SegmentType::Silence)
    }
}

/// Advanced segment feature analyzer
pub struct SegmentAnalyzer {
    sample_rate: u32,
}

impl SegmentAnalyzer {
    /// Create a new segment analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Analyze spectral features of a segment
    pub fn analyze_spectral_features(&self, data: &ArrayView1<f32>) -> Result<SpectralFeatures> {
        if data.len() < 512 {
            return Err(ZipEnhancerError::processing(
                "Data length insufficient for spectral analysis, at least 512 samples required"
            ));
        }

        // Apply Hanning window
        let windowed_data = self.apply_hanning_window(data)?;
        
        // Simplified power spectrum estimation (using energy distribution)
        let spectral_centroid = self.calculate_spectral_centroid(&windowed_data)?;
        let spectral_rolloff = self.calculate_spectral_rolloff(&windowed_data)?;
        let spectral_bandwidth = self.calculate_spectral_bandwidth(&windowed_data)?;
        let zero_crossing_rate = self.calculate_zero_crossing_rate(data);

        Ok(SpectralFeatures {
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth,
            zero_crossing_rate,
            energy: data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32,
        })
    }

    /// Apply Hanning window
    fn apply_hanning_window(&self, data: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let n = data.len();
        let mut windowed = Array1::zeros(n);
        
        for (i, &sample) in data.iter().enumerate() {
            let window_value = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            windowed[i] = sample * window_value;
        }
        
        Ok(windowed)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, data: &Array1<f32>) -> Result<f32> {
        // Simplified calculation: using signal energy distribution
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

    /// Calculate spectral rolloff point
    fn calculate_spectral_rolloff(&self, data: &Array1<f32>) -> Result<f32> {
        let mut total_energy = 0.0;
        let energies: Vec<f32> = data.iter().map(|&x| {
            let energy = x * x;
            total_energy += energy;
            energy
        }).collect();
        
        if total_energy == 0.0 {
            return Ok(0.0);
        }
        
        let threshold = total_energy * 0.85; // 85% energy threshold
        let mut cumulative_energy = 0.0;
        
        for (i, &energy) in energies.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= threshold {
                return Ok(i as f32);
            }
        }
        
        Ok(data.len() as f32)
    }

    /// Calculate spectral bandwidth
    fn calculate_spectral_bandwidth(&self, data: &Array1<f32>) -> Result<f32> {
        let centroid = self.calculate_spectral_centroid(data)?;
        let mut weighted_variance = 0.0;
        let mut total_energy = 0.0;
        
        for (i, &sample) in data.iter().enumerate() {
            let energy = sample * sample;
            let deviation = i as f32 - centroid;
            weighted_variance += deviation * deviation * energy;
            total_energy += energy;
        }
        
        if total_energy == 0.0 {
            return Ok(0.0);
        }
        
        Ok((weighted_variance / total_energy).sqrt())
    }

    /// Calculate zero-crossing rate
    fn calculate_zero_crossing_rate(&self, data: &ArrayView1<f32>) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mut crossings = 0;
        for i in 1..data.len() {
            if (data[i] >= 0.0 && data[i-1] < 0.0) || (data[i] < 0.0 && data[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / (data.len() - 1) as f32
    }

    /// Detect music beats (simplified version)
    pub fn detect_beats(&self, data: &ArrayView1<f32>, window_size: usize) -> Result<Vec<usize>> {
        if data.len() < window_size {
            return Err(ZipEnhancerError::processing(
                "Data length insufficient for beat detection"
            ));
        }

        let mut beats = Vec::new();
        let mut energy_history = Vec::new();

        // Calculate sliding window energy
        for i in 0..=(data.len() - window_size) {
            let window_energy: f32 = data.slice(ndarray::s![i..i+window_size])
                .iter()
                .map(|&x| x * x)
                .sum::<f32>() / window_size as f32;
            energy_history.push(window_energy);
        }

        // Simplified beat detection: find local energy peaks
        for i in 1..energy_history.len() - 1 {
            if energy_history[i] > energy_history[i-1] * 1.3 &&
               energy_history[i] > energy_history[i+1] * 1.3 {
                beats.push(i + window_size / 2);
            }
        }

        Ok(beats)
    }

    /// Analyze audio dynamic range
    pub fn analyze_dynamic_range(&self, data: &ArrayView1<f32>) -> Result<DynamicRangeInfo> {
        let peak = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let rms = (data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32).sqrt();
        
        // Calculate short-term energy variations
        let frame_size = (self.sample_rate / 100) as usize; // 10ms frame
        let mut frame_energies = Vec::new();
        
        for i in (0..data.len()).step_by(frame_size) {
            let end_idx = (i + frame_size).min(data.len());
            if end_idx > i {
                let frame_energy: f32 = data.slice(ndarray::s![i..end_idx])
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f32>() / (end_idx - i) as f32;
                frame_energies.push(frame_energy);
            }
        }
        
        let dynamic_range = if rms > 0.0 { 20.0 * (peak / rms).log10() } else { 0.0 };
        let energy_variance = if frame_energies.len() > 1 {
            let mean = frame_energies.iter().sum::<f32>() / frame_energies.len() as f32;
            frame_energies.iter()
                .map(|&e| (e - mean) * (e - mean))
                .sum::<f32>() / frame_energies.len() as f32
        } else {
            0.0
        };
        
        Ok(DynamicRangeInfo {
            peak,
            rms,
            dynamic_range,
            energy_variance,
            compression_ratio: if peak > 0.0 { rms / peak } else { 0.0 },
        })
    }
}

/// Spectral features
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral rolloff point
    pub spectral_rolloff: f32,
    /// Spectral bandwidth
    pub spectral_bandwidth: f32,
    /// Zero-crossing rate
    pub zero_crossing_rate: f32,
    /// Energy
    pub energy: f32,
}

/// Dynamic range information
#[derive(Debug, Clone)]
pub struct DynamicRangeInfo {
    /// Peak value
    pub peak: f32,
    /// Root mean square value
    pub rms: f32,
    /// Dynamic range (dB)
    pub dynamic_range: f32,
    /// Energy variance
    pub energy_variance: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

/// Adaptive segmenter
pub struct AdaptiveSegmenter {
    analyzer: SegmentAnalyzer,
    config: SegmentationConfig,
}

impl AdaptiveSegmenter {
    /// Create a new adaptive segmenter
    pub fn new(sample_rate: u32, config: SegmentationConfig) -> Self {
        Self {
            analyzer: SegmentAnalyzer::new(sample_rate),
            config,
        }
    }

    /// Perform adaptive segmentation
    pub fn segment_adaptive(&self, audio: &WavAudio) -> Result<Vec<SegmentInfo>> {
        let data_owned = match audio.data() {
            AudioData::Mono(data) => data.to_owned(),
            AudioData::Stereo(data) => {
                data.mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty")
            }
        };
        let data = data_owned.view();

        // Analyze audio features
        let features = self.analyzer.analyze_spectral_features(&data)?;
        let dynamic_info = self.analyzer.analyze_dynamic_range(&data)?;

        // Adjust segmentation strategy based on features
        let adjusted_config = self.adjust_config_based_on_features(&features, &dynamic_info)?;

        // Perform segmentation
        self.perform_segmentation(&data, &adjusted_config)
    }

    /// Adjust configuration based on audio features
    fn adjust_config_based_on_features(
        &self,
        features: &SpectralFeatures,
        dynamic_info: &DynamicRangeInfo
    ) -> Result<SegmentationConfig> {
        let mut config = self.config.clone();

        // Adjust segment size based on spectral centroid
        if features.spectral_centroid > 2000.0 {
            // High frequency content, use smaller segments
            config.segment_size = (config.segment_size as f32 * 0.8) as usize;
        } else if features.spectral_centroid < 1000.0 {
            // Low frequency content, use larger segments
            config.segment_size = (config.segment_size as f32 * 1.2) as usize;
        }

        // Adjust overlap ratio based on dynamic range
        if dynamic_info.dynamic_range > 12.0 {
            // High dynamic range, increase overlap
            config.overlap_ratio = (config.overlap_ratio + 0.2).min(0.5);
        }

        // Adjust silence threshold based on zero-crossing rate
        if features.zero_crossing_rate > 0.15 {
            // High zero-crossing rate, may contain noise or high-frequency content
            config.silence_threshold *= 1.2;
        }

        Ok(config)
    }

    /// Perform actual segmentation
    fn perform_segmentation(
        &self,
        data: &ArrayView1<f32>,
        config: &SegmentationConfig
    ) -> Result<Vec<SegmentInfo>> {
        match config.strategy {
            SegmentationStrategy::FixedLength => {
                self.segment_fixed_length_adaptive(data, config)
            }
            SegmentationStrategy::VariableLength => {
                self.segment_variable_length_adaptive(data, config)
            }
            SegmentationStrategy::VoiceActivityDetection => {
                self.segment_vad_adaptive(data, config)
            }
        }
    }

    /// Adaptive fixed-length segmentation
    fn segment_fixed_length_adaptive(
        &self,
        data: &ArrayView1<f32>,
        config: &SegmentationConfig
    ) -> Result<Vec<SegmentInfo>> {
        let hop_size = (config.segment_size as f32 * (1.0 - config.overlap_ratio)) as usize;
        let mut segments = Vec::new();
        
        for (i, start_pos) in (0..data.len()).step_by(hop_size).enumerate() {
            let end_pos = (start_pos + config.segment_size).min(data.len());
            
            if end_pos > start_pos {
                let segment_data = data.slice(ndarray::s![start_pos..end_pos]);
                let features = self.analyzer.analyze_spectral_features(&segment_data)?;
                
                let segment_type = if features.energy > config.silence_threshold {
                    SegmentType::Speech
                } else {
                    SegmentType::Silence
                };
                
                segments.push(SegmentInfo {
                    index: i,
                    start_sample: start_pos,
                    end_sample: end_pos,
                    length: end_pos - start_pos,
                    segment_type,
                    boundaries: vec![],
                    energy: Some(features.energy),
                    zero_crossings: Some(features.zero_crossing_rate),
                });
            }
        }
        
        Ok(segments)
    }

    /// Adaptive variable-length segmentation
    fn segment_variable_length_adaptive(
        &self,
        data: &ArrayView1<f32>,
        config: &SegmentationConfig
    ) -> Result<Vec<SegmentInfo>> {
        // Content-based adaptive segmentation
        let mut segments = Vec::new();
        let mut start_pos = 0;
        let mut segment_index = 0;
        
        while start_pos < data.len() {
            let end_pos = (start_pos + config.segment_size).min(data.len());
            let segment_data = data.slice(ndarray::s![start_pos..end_pos]);
            let features = self.analyzer.analyze_spectral_features(&segment_data)?;
            
            // Adjust segment length based on content features
            let adjusted_length = if features.spectral_centroid > 1500.0 {
                (config.segment_size as f32 * 0.8) as usize
            } else if features.energy < 0.01 {
                (config.segment_size as f32 * 1.3) as usize
            } else {
                config.segment_size
            };
            
            let final_end_pos = (start_pos + adjusted_length).min(data.len());
            
            segments.push(SegmentInfo {
                index: segment_index,
                start_sample: start_pos,
                end_sample: final_end_pos,
                length: final_end_pos - start_pos,
                segment_type: SegmentType::Variable,
                boundaries: vec![],
                energy: Some(features.energy),
                zero_crossings: Some(features.zero_crossing_rate),
            });
            
            start_pos = final_end_pos;
            segment_index += 1;
        }
        
        Ok(segments)
    }

    /// Adaptive voice activity detection segmentation
    fn segment_vad_adaptive(
        &self,
        data: &ArrayView1<f32>,
        config: &SegmentationConfig
    ) -> Result<Vec<SegmentInfo>> {
        let vad_config = config.vad_config.as_ref()
            .cloned()
            .unwrap_or_else(VADConfig::default);
        
        let activity_levels = self.analyze_voice_activity_enhanced(data, &vad_config)?;
        let segments = self.create_segments_from_activity(&activity_levels, config)?;
        
        Ok(segments)
    }

    /// Enhanced voice activity analysis
    fn analyze_voice_activity_enhanced(
        &self,
        data: &ArrayView1<f32>,
        config: &VADConfig
    ) -> Result<Vec<ActivityLevel>> {
        let mut activity_levels = Vec::new();
        let window_size = config.energy_window_size;
        
        for i in (0..data.len()).step_by(window_size / 2) {
            let end_idx = (i + window_size).min(data.len());
            if end_idx > i {
                let window_data = data.slice(ndarray::s![i..end_idx]);
                let features = self.analyzer.analyze_spectral_features(&window_data)?;
                
                // Combine multiple features for voice activity detection
                let energy_score = if features.energy > config.energy_threshold {
                    1.0
                } else {
                    features.energy / config.energy_threshold
                };
                
                let zcr_score = if features.zero_crossing_rate < config.zero_crossing_threshold / 1000.0 {
                    1.0
                } else {
                    1.0 - (features.zero_crossing_rate - config.zero_crossing_threshold / 1000.0)
                };
                
                let centroid_score = if features.spectral_centroid > 500.0 && features.spectral_centroid < 4000.0 {
                    1.0
                } else {
                    0.5
                };
                
                let combined_score = (energy_score + zcr_score + centroid_score) / 3.0;
                
                activity_levels.push(ActivityLevel {
                    position: i,
                    energy: combined_score,
                    zero_crossings: features.zero_crossing_rate,
                    is_speech: combined_score > 0.5,
                    confidence: combined_score,
                });
            }
        }
        
        Ok(activity_levels)
    }

    /// Create segments from voice activity levels
    fn create_segments_from_activity(
        &self,
        activity_levels: &[ActivityLevel],
        config: &SegmentationConfig
    ) -> Result<Vec<SegmentInfo>> {
        let mut segments = Vec::new();
        let mut segment_start = None;
        let mut segment_index = 0;
        
        for (_i, activity) in activity_levels.iter().enumerate() {
            if activity.is_speech && segment_start.is_none() {
                // Speech start
                segment_start = Some(activity.position);
            } else if !activity.is_speech && segment_start.is_some() {
                // Speech end
                let start_pos = segment_start.unwrap();
                let end_pos = activity.position;
                let length = end_pos - start_pos;
                
                if length >= config.min_segment_length {
                    segments.push(SegmentInfo {
                        index: segment_index,
                        start_sample: start_pos,
                        end_sample: end_pos,
                        length,
                        segment_type: SegmentType::Speech,
                        boundaries: vec![],
                        energy: Some(activity.energy),
                        zero_crossings: None,
                    });
                    segment_index += 1;
                }
                
                segment_start = None;
            }
        }
        
        // Handle the last segment
        if let Some(start_pos) = segment_start {
            let end_pos = activity_levels.last().unwrap().position;
            let length = end_pos - start_pos;
            
            if length >= config.min_segment_length {
                segments.push(SegmentInfo {
                    index: segment_index,
                    start_sample: start_pos,
                    end_sample: end_pos,
                    length,
                    segment_type: SegmentType::Speech,
                    boundaries: vec![],
                    energy: Some(activity_levels.last().unwrap().energy),
                    zero_crossings: None,
                });
            }
        }
        
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioFormat;
    use ndarray::Array1;

    #[test]
    fn test_segmentation_config_default() {
        let config = SegmentationConfig::default();
        assert_eq!(config.strategy, SegmentationStrategy::FixedLength);
        assert_eq!(config.segment_size, 16000);
        assert_eq!(config.overlap_ratio, 0.1);
    }

    #[test]
    fn test_vad_config_default() {
        let config = VADConfig::default();
        assert_eq!(config.energy_window_size, 400);
        assert_eq!(config.energy_threshold, 0.01);
        assert_eq!(config.zero_crossing_threshold, 10.0);
    }

    #[test]
    fn test_segmenter_creation() {
        let config = SegmentationConfig::default();
        let segmenter = AudioSegmenter::new(config);
        assert!(!segmenter.vad_enabled);
        
        let config_with_vad = SegmentationConfig {
            vad_config: Some(VADConfig::default()),
            ..Default::default()
        };
        let segmenter_with_vad = AudioSegmenter::new(config_with_vad);
        assert!(segmenter_with_vad.vad_enabled);
    }

    #[test]
    fn test_fixed_length_segmentation() {
        let config = SegmentationConfig {
            segment_size: 4,
            overlap_ratio: 0.5,
            ..Default::default()
        };
        let segmenter = AudioSegmenter::new(config);
        
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let audio = WavAudio::new_mono(16000, data.clone(), AudioFormat::Float32);
        
        let segments = segmenter.segment(&audio).unwrap();
        
        assert_eq!(segments.len(), 4);
        
        for (i, segment) in segments.iter().enumerate() {
            if i < 2 {
                // First segments should be complete with full length
                assert_eq!(segment.length, 4);
            } else {
                // Later segments may be partial
                assert!(segment.length <= 4);
            }
        }
    }

    #[test]
    fn test_energy_calculation() {
        let segmenter = AudioSegmenter::with_default_config();
        let data = ArrayView1::from(&[0.1, -0.1, 0.2, -0.2]);
        
        let energy = segmenter.calculate_energy(&data);
        let expected = (0.1 * 0.1 + (-0.1) * (-0.1) + 0.2 * 0.2 + (-0.2) * (-0.2)) / 4.0;
        assert!((energy - expected).abs() < 1e-6);
    }

    #[test]
    fn test_zero_crossing_count() {
        let segmenter = AudioSegmenter::with_default_config();
        let data = ArrayView1::from(&[1.0, -1.0, 1.0, -1.0, 1.0]);
        
        let crossings = segmenter.count_zero_crossings(&data);
        assert_eq!(crossings, 4.0);
    }

    #[test]
    fn test_activity_analysis() {
        let segmenter = AudioSegmenter::new(SegmentationConfig {
            vad_config: Some(VADConfig::default()),
            ..Default::default()
        });
        
        let data = ArrayView1::from(&[0.0, 0.1, 0.5, 1.0, 0.3, 0.1, 0.0]);
        
        let vad_config = VADConfig {
            energy_window_size: 4,  // Use smaller window for test
            ..VADConfig::default()
        };
        let levels = segmenter.analyze_voice_activity(&data, &vad_config).unwrap();
        
        assert!(!levels.is_empty());
        assert_eq!(levels.len(), 4); // 7 samples, window_size=4, hop_size=2
        
        assert!(levels[1].energy > levels[0].energy);
        assert!(levels[1].is_speech);
    }

    #[test]
    fn test_silence_boundary_detection() {
        let segmenter = AudioSegmenter::new(SegmentationConfig::default());
        
        let data = ArrayView1::from(&[0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1.0, 1.0]);
        let boundaries = segmenter.find_silence_boundaries(&data).unwrap();
        
        assert!(boundaries.len() >= 2);
        assert_eq!(boundaries[0].position, 0);
        assert_eq!(boundaries.last().unwrap().position, data.len());
    }
}