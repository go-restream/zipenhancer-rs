//! WAV audio file processing

use std::path::Path;
use std::fs::File;
use hound::{WavReader, WavWriter, SampleFormat};
use ndarray::{Array1, Array2};
use crate::error::{ZipEnhancerError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Int16,
    Float32,
}

impl AudioFormat {
    pub fn name(&self) -> &'static str {
        match self {
            AudioFormat::Int16 => "int16",
            AudioFormat::Float32 => "float32",
        }
    }

    pub fn bytes_per_sample(&self) -> u16 {
        match self {
            AudioFormat::Int16 => 2,
            AudioFormat::Float32 => 4,
        }
    }

    pub fn to_sample_format(self) -> SampleFormat {
        match self {
            AudioFormat::Int16 => SampleFormat::Int,
            AudioFormat::Float32 => SampleFormat::Float,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioHeader {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: AudioFormat,
    pub total_samples: u32,
    pub bits_per_sample: u16,
    pub duration: f64,
}

impl AudioHeader {
    pub fn new(sample_rate: u32, channels: u16, format: AudioFormat, total_samples: u32) -> Self {
        let bits_per_sample = format.bytes_per_sample() * 8;
        let duration = total_samples as f64 / sample_rate as f64;

        Self {
            sample_rate,
            channels,
            format,
            total_samples,
            bits_per_sample,
            duration,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(ZipEnhancerError::audio("Sample rate cannot be 0"));
        }

        if self.channels == 0 || self.channels > 2 {
            return Err(ZipEnhancerError::audio("Channel count must be 1 or 2"));
        }

        if self.total_samples == 0 {
            return Err(ZipEnhancerError::audio("Total samples cannot be 0"));
        }

        if self.duration <= 0.0 {
            return Err(ZipEnhancerError::audio("Audio duration must be greater than 0"));
        }

        Ok(())
    }

    pub fn to_wav_spec(&self) -> hound::WavSpec {
        hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: self.format.to_sample_format(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavAudio {
    pub header: AudioHeader,
    pub data: AudioData,
}

#[derive(Debug, Clone)]
pub enum AudioData {
    Mono(Array1<f32>),
    Stereo(Array2<f32>),
}

impl AudioData {
    pub fn len(&self) -> usize {
        match self {
            AudioData::Mono(data) => data.len(),
            AudioData::Stereo(data) => data.nrows(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn channels(&self) -> u16 {
        match self {
            AudioData::Mono(_) => 1,
            AudioData::Stereo(_) => 2,
        }
    }

    pub fn to_mono(&self) -> Array1<f32> {
        match self {
            AudioData::Mono(data) => data.clone(),
            AudioData::Stereo(data) => {
                data.mapv(|x| x).mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty")
            }
        }
    }

    pub fn to_stereo(&self) -> Array2<f32> {
        match self {
            AudioData::Mono(data) => {
                let len = data.len();
                let mut stereo = Array2::zeros((len, 2));
                stereo.column_mut(0).assign(&data);
                stereo.column_mut(1).assign(&data);
                stereo
            }
            AudioData::Stereo(data) => data.clone(),
        }
    }
}

impl WavAudio {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        let file = File::open(path)
            .map_err(|e| ZipEnhancerError::Audio {
                message: format!("Cannot open audio file {}: {}", path.display(), e)
            })?;
        
        let mut reader = WavReader::new(file)
            .map_err(|e| ZipEnhancerError::Audio {
                message: format!("Cannot create WAV reader: {}", e)
            })?;

        let spec = reader.spec();
        let total_samples = reader.len() as u32;
        let duration = total_samples as f64 / spec.sample_rate as f64;

        if spec.sample_rate == 0 {
            return Err(ZipEnhancerError::audio("Invalid sample rate"));
        }

        if spec.channels == 0 || spec.channels > 2 {
            return Err(ZipEnhancerError::audio("Only mono or stereo audio supported"));
        }

        let format = match spec.bits_per_sample {
            16 => AudioFormat::Int16,
            32 => {
                match spec.sample_format {
                    SampleFormat::Float => AudioFormat::Float32,
                    _ => return Err(ZipEnhancerError::audio(
                        format!("Unsupported 32-bit format: {:?}", spec.sample_format)
                    )),
                }
            }
            _ => return Err(ZipEnhancerError::audio(
                format!("Unsupported bit depth: {}", spec.bits_per_sample)
            )),
        };

        let samples: Result<Vec<f32>> = match spec.bits_per_sample {
            16 => {
                let i16_samples: std::result::Result<Vec<i16>, _> = reader.samples::<i16>()
                    .map(|sample| sample.map_err(|e| ZipEnhancerError::Audio {
                        message: format!("Failed to read sample: {}", e)
                    }))
                    .collect();
                let i16_samples = i16_samples?;
                Ok(i16_samples.into_iter().map(|s| (s as f32) / 32767.0).collect())
            },
            32 => {
                let samples: Result<Vec<f32>> = reader.samples::<f32>()
                    .map(|sample| sample.map_err(|e| ZipEnhancerError::Audio {
                        message: format!("Failed to read sample: {}", e)
                    }))
                    .collect();
                samples
            },
            _ => return Err(ZipEnhancerError::audio(
                format!("Unsupported bit depth: {}", spec.bits_per_sample)
            )),
        };

        let samples = samples?;

        let data = if spec.channels == 1 {
            AudioData::Mono(Array1::from(samples))
        } else {
            let len = samples.len() / 2;
            let mut stereo = Array2::zeros((len, 2));
            for (i, chunk) in samples.chunks_exact(2).enumerate() {
                stereo[[i, 0]] = chunk[0];
                stereo[[i, 1]] = chunk[1];
            }
            AudioData::Stereo(stereo)
        };

        let audio_header = AudioHeader {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            format,
            total_samples,
            bits_per_sample: spec.bits_per_sample,
            duration,
        };

        audio_header.validate()?;

        Ok(WavAudio {
            header: audio_header,
            data,
        })
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ZipEnhancerError::Audio {
                    message: format!("Cannot create output directory: {}", e)
                })?;
        }

        let file = File::create(path)
            .map_err(|e| ZipEnhancerError::Audio {
                message: format!("Cannot create output file {}: {}", path.display(), e)
            })?;
        
        let spec = self.header.to_wav_spec();
        let mut writer = WavWriter::new(file, spec)
            .map_err(|e| ZipEnhancerError::Audio {
                message: format!("Cannot create WAV writer: {}", e)
            })?;

        match &self.data {
            AudioData::Mono(data) => {
                for &sample in data.iter() {
                    let clamped_sample = sample.clamp(-1.0, 1.0);
                    if spec.sample_format == SampleFormat::Float {
                        writer.write_sample(clamped_sample)
                            .map_err(|e| ZipEnhancerError::Audio {
                                message: format!("Failed to write sample: {}", e)
                            })?;
                    } else {
                        let sample_i16 = (clamped_sample * 32767.0) as i16;
                        writer.write_sample(sample_i16)
                            .map_err(|e| ZipEnhancerError::Audio {
                                message: format!("Failed to write sample: {}", e)
                            })?;
                    }
                }
            }
            AudioData::Stereo(data) => {
                for row in data.rows() {
                    for &sample in row.iter() {
                        let clamped_sample = sample.clamp(-1.0, 1.0);
                        if spec.sample_format == SampleFormat::Float {
                            writer.write_sample(clamped_sample)
                                .map_err(|e| ZipEnhancerError::Audio {
                                    message: format!("Failed to write sample: {}", e)
                                })?;
                        } else {
                            let sample_i16 = (clamped_sample * 32767.0) as i16;
                            writer.write_sample(sample_i16)
                                .map_err(|e| ZipEnhancerError::Audio {
                                    message: format!("Failed to write sample: {}", e)
                                })?;
                        }
                    }
                }
            }
        }

        writer.finalize()
            .map_err(|e| ZipEnhancerError::Audio {
                message: format!("Failed to finalize WAV writing: {}", e)
            })?;

        Ok(())
    }

    pub fn new_mono(sample_rate: u32, data: Array1<f32>, format: AudioFormat) -> Self {
        let total_samples = data.len() as u32;
        let header = AudioHeader::new(sample_rate, 1, format, total_samples);
        
        WavAudio {
            header,
            data: AudioData::Mono(data),
        }
    }

    pub fn new_stereo(sample_rate: u32, data: Array2<f32>, format: AudioFormat) -> Result<Self> {
        if data.ncols() != 2 {
            return Err(ZipEnhancerError::audio("Stereo data must have 2 columns"));
        }

        let total_samples = data.nrows() as u32;
        let header = AudioHeader::new(sample_rate, 2, format, total_samples);
        
        Ok(WavAudio {
            header,
            data: AudioData::Stereo(data),
        })
    }

    pub fn data(&self) -> &AudioData {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut AudioData {
        &mut self.data
    }

    pub fn sample_rate(&self) -> u32 {
        self.header.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.header.channels
    }

    pub fn total_samples(&self) -> u32 {
        self.header.total_samples
    }

    pub fn duration(&self) -> f64 {
        self.header.duration
    }

    pub fn format(&self) -> AudioFormat {
        self.header.format
    }

    pub fn resize(&mut self, new_length: usize) -> Result<()> {
        match &mut self.data {
            AudioData::Mono(data) => {
                if new_length > data.len() {
                    let mut new_data = Array1::zeros(new_length);
                    new_data.slice_mut(ndarray::s![..data.len()]).assign(data);
                    *data = new_data;
                } else {
                    *data = data.slice(ndarray::s![..new_length]).to_owned();
                }
            }
            AudioData::Stereo(data) => {
                if new_length > data.nrows() {
                    let mut new_data = Array2::zeros((new_length, 2));
                    new_data.slice_mut(ndarray::s![..data.nrows(), ..]).assign(data);
                    *data = new_data;
                } else {
                    *data = data.slice(ndarray::s![..new_length, ..]).to_owned();
                }
            }
        }

        self.header.total_samples = new_length as u32;
        self.header.duration = new_length as f64 / self.header.sample_rate as f64;

        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        self.header.validate()?;

        if self.data.len() as u32 != self.header.total_samples {
            return Err(ZipEnhancerError::audio(
                format!("Data length mismatch: header shows {} samples, actual {} samples",
                       self.header.total_samples, self.data.len())
            ));
        }

        if self.data.channels() != self.header.channels {
            return Err(ZipEnhancerError::audio(
                format!("Channel count mismatch: header shows {} channels, actual {} channels",
                       self.header.channels, self.data.channels())
            ));
        }

        match &self.data {
            AudioData::Mono(data) => {
                for (i, &sample) in data.iter().enumerate() {
                    if !sample.is_finite() {
                        println!("Warning: Audio data at position {} contains invalid value, replacing with 0.0", i);
                        continue;
                    }
                    if sample < -1.0 || sample > 1.0 {
                        println!("Warning: Audio data at position {} out of range [-1.0, 1.0]: {:.6}, clipping", i, sample);
                    }
                }
            }
            AudioData::Stereo(data) => {
                for (i, &sample) in data.iter().enumerate() {
                    if !sample.is_finite() {
                        println!("Warning: Audio data at position {} contains invalid value, replacing with 0.0", i);
                        continue;
                    }
                    if sample < -1.0 || sample > 1.0 {
                        println!("Warning: Audio data at position {} out of range [-1.0, 1.0]: {:.6}, clipping", i, sample);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_audio_format() {
        assert_eq!(AudioFormat::Int16.name(), "int16");
        assert_eq!(AudioFormat::Int16.bytes_per_sample(), 2);
        assert_eq!(AudioFormat::Float32.name(), "float32");
        assert_eq!(AudioFormat::Float32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_audio_header_creation() {
        let header = AudioHeader::new(16000, 1, AudioFormat::Float32, 1000);
        assert_eq!(header.sample_rate, 16000);
        assert_eq!(header.channels, 1);
        assert_eq!(header.format, AudioFormat::Float32);
        assert_eq!(header.total_samples, 1000);
        assert_eq!(header.bits_per_sample, 32);
        assert!((header.duration - 0.0625).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_header_validation() {
        let header = AudioHeader::new(16000, 1, AudioFormat::Float32, 1000);
        assert!(header.validate().is_ok());

        let invalid_header = AudioHeader::new(0, 1, AudioFormat::Float32, 1000);
        assert!(invalid_header.validate().is_err());

        let invalid_header = AudioHeader::new(16000, 3, AudioFormat::Float32, 1000);
        assert!(invalid_header.validate().is_err());

        let invalid_header = AudioHeader::new(16000, 1, AudioFormat::Float32, 0);
        assert!(invalid_header.validate().is_err());
    }

    #[test]
    fn test_audio_data_operations() {
        // Test mono data
        let mono_data = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let mono = AudioData::Mono(mono_data.clone());
        assert_eq!(mono.len(), 4);
        assert_eq!(mono.channels(), 1);
        assert!(!mono.is_empty());

        // Test mono to stereo conversion
        let stereo_from_mono = mono.to_stereo();
        assert_eq!(stereo_from_mono.nrows(), 4);
        assert_eq!(stereo_from_mono.ncols(), 2);

        // Test stereo data
        let stereo_data = Array2::from(vec![[0.1, 0.2], [0.3, 0.4]]);
        let stereo = AudioData::Stereo(stereo_data.clone());
        assert_eq!(stereo.len(), 2);
        assert_eq!(stereo.channels(), 2);
        assert!(!stereo.is_empty());

        // Test stereo to mono conversion
        let mono_from_stereo = stereo.to_mono();
        assert_eq!(mono_from_stereo.len(), 2);
    }

    #[test]
    fn test_wav_audio_creation() {
        let data = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let audio = WavAudio::new_mono(16000, data.clone(), AudioFormat::Float32);
        
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.total_samples(), 4);
        assert_eq!(audio.format(), AudioFormat::Float32);
        assert!(audio.validate().is_ok());
    }

    #[test]
    fn test_wav_audio_resize() {
        let data = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        let mut audio = WavAudio::new_mono(16000, data.clone(), AudioFormat::Float32);
        
        // Test expansion
        audio.resize(6).unwrap();
        assert_eq!(audio.total_samples(), 6);
        
        // Test truncation
        audio.resize(2).unwrap();
        assert_eq!(audio.total_samples(), 2);
    }

    #[test]
    fn test_stereo_creation() {
        let data = Array2::from(vec![[0.1, 0.2], [0.3, 0.4]]);
        let audio = WavAudio::new_stereo(16000, data.clone(), AudioFormat::Float32).unwrap();
        
        assert_eq!(audio.sample_rate(), 16000);
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.total_samples(), 2);
        assert!(audio.validate().is_ok());
    }

    #[test]
    fn test_invalid_stereo_creation() {
        // 3-column data should fail
        let data = Array2::from(vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let result = WavAudio::new_stereo(16000, data, AudioFormat::Float32);
        assert!(result.is_err());
    }

    #[test]
    fn test_wav_roundtrip() {
        // Create test data
        let data = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let original_audio = WavAudio::new_mono(16000, data.clone(), AudioFormat::Float32);

        // Create temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();

        // Save to file
        original_audio.save_to_file(&temp_path).unwrap();

        // Read from file
        let loaded_audio = WavAudio::from_file(&temp_path).unwrap();

        // Verify data
        assert_eq!(loaded_audio.sample_rate(), original_audio.sample_rate());
        assert_eq!(loaded_audio.channels(), original_audio.channels());
        assert_eq!(loaded_audio.total_samples(), original_audio.total_samples());

        match (loaded_audio.data(), original_audio.data()) {
            (AudioData::Mono(loaded_data), AudioData::Mono(original_data)) => {
                assert_eq!(loaded_data.len(), original_data.len());
                for (loaded, original) in loaded_data.iter().zip(original_data.iter()) {
                    assert!((loaded - original).abs() < 1e-6);
                }
            }
            _ => panic!("Audio data format mismatch"),
        }
    }
}