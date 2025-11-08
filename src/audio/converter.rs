//! Audio Format Converter
//!
//! Provides audio data format conversion, type conversion, and processing functions.

use ndarray::{Array1, Array2, ArrayView1};
use crate::audio::{AudioData, AudioFormat, WavAudio};
use crate::error::{ZipEnhancerError, Result};

pub struct AudioConverter;

impl AudioConverter {
    pub fn convert_format(audio: &WavAudio, target_format: AudioFormat) -> Result<WavAudio> {
        if audio.format() == target_format {
            return Ok(audio.clone());
        }

        let mut new_audio = audio.clone();
        new_audio.header.format = target_format;
        new_audio.header.bits_per_sample = target_format.bytes_per_sample() * 8;

        Ok(new_audio)
    }

    pub fn convert_sample_rate(audio: &WavAudio, target_sample_rate: u32) -> Result<WavAudio> {
        if audio.sample_rate() == target_sample_rate {
            return Ok(audio.clone());
        }

        let ratio = target_sample_rate as f64 / audio.sample_rate() as f64;
        let new_length = (audio.data().len() as f64 * ratio) as usize;

        match audio.data() {
            AudioData::Mono(data) => {
                let new_data = Self::resample_mono(data.view(), new_length, ratio)?;
                let mut new_audio = audio.clone();
                new_audio.header.sample_rate = target_sample_rate;
                new_audio.header.total_samples = new_length as u32;
                new_audio.header.duration = new_length as f64 / target_sample_rate as f64;
                new_audio.data = AudioData::Mono(new_data);
                Ok(new_audio)
            }
            AudioData::Stereo(data) => {
                let left = data.column(0);
                let right = data.column(1);
                let new_left = Self::resample_mono(left, new_length, ratio)?;
                let new_right = Self::resample_mono(right, new_length, ratio)?;

                let mut new_data = Array2::zeros((new_length, 2));
                new_data.column_mut(0).assign(&new_left);
                new_data.column_mut(1).assign(&new_right);

                let mut new_audio = audio.clone();
                new_audio.header.sample_rate = target_sample_rate;
                new_audio.header.total_samples = new_length as u32;
                new_audio.header.duration = new_length as f64 / target_sample_rate as f64;
                new_audio.data = AudioData::Stereo(new_data);
                Ok(new_audio)
            }
        }
    }

    fn resample_mono(data: ArrayView1<f32>, new_length: usize, ratio: f64) -> Result<Array1<f32>> {
        if data.is_empty() {
            return Err(ZipEnhancerError::audio("Input data is empty"));
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

    pub fn convert_channels(audio: &WavAudio, target_channels: u16) -> Result<WavAudio> {
        if audio.channels() == target_channels {
            return Ok(audio.clone());
        }

        if target_channels != 1 && target_channels != 2 {
            return Err(ZipEnhancerError::audio(
                format!("Unsupported channel count: {}, only 1 or 2 channels supported", target_channels)
            ));
        }

        let new_data = match (audio.data(), target_channels) {
            (AudioData::Mono(data), 2) => {
                let len = data.len();
                let mut stereo = Array2::zeros((len, 2));
                stereo.column_mut(0).assign(data);
                stereo.column_mut(1).assign(data);
                AudioData::Stereo(stereo)
            }
            (AudioData::Stereo(data), 1) => {
                let mono = data.mapv(|x| x).mean_axis(ndarray::Axis(1))
                    .expect("Stereo data cannot be empty");
                AudioData::Mono(mono)
            }
            _ => unreachable!(),
        };

        let mut new_audio = audio.clone();
        new_audio.header.channels = target_channels;
        new_audio.data = new_data;

        Ok(new_audio)
    }

    pub fn normalize(audio: &WavAudio) -> Result<WavAudio> {
        let mut new_audio = audio.clone();

        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                if let Some(max_abs) = data.iter().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    if max_abs > 0.0 {
                        *data /= max_abs;
                    }
                }
            }
            AudioData::Stereo(data) => {
                if let Some(max_abs) = data.iter().map(|&x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    if max_abs > 0.0 {
                        *data /= max_abs;
                    }
                }
            }
        }

        Ok(new_audio)
    }

    pub fn apply_gain(audio: &WavAudio, gain_db: f32) -> Result<WavAudio> {
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);
        let mut new_audio = audio.clone();

        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                *data *= gain_linear;
            }
            AudioData::Stereo(data) => {
                *data *= gain_linear;
            }
        }

        Ok(new_audio)
    }

    pub fn fade_in(audio: &WavAudio, duration_ms: f64) -> Result<WavAudio> {
        let fade_samples = (duration_ms / 1000.0 * audio.sample_rate() as f64) as usize;
        let mut new_audio = audio.clone();

        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                let fade_len = fade_samples.min(data.len());
                for (i, sample) in data.iter_mut().take(fade_len).enumerate() {
                    let fade_factor = (i as f32 / fade_len as f32).powf(2.0);
                    *sample *= fade_factor;
                }
            }
            AudioData::Stereo(data) => {
                let fade_len = fade_samples.min(data.nrows());
                for (i, mut row) in data.rows_mut().into_iter().take(fade_len).enumerate() {
                    let fade_factor = (i as f32 / fade_len as f32).powf(2.0);
                    row *= fade_factor;
                }
            }
        }

        Ok(new_audio)
    }

    pub fn fade_out(audio: &WavAudio, duration_ms: f64) -> Result<WavAudio> {
        let fade_samples = (duration_ms / 1000.0 * audio.sample_rate() as f64) as usize;
        let mut new_audio = audio.clone();
        let total_samples = new_audio.data().len();

        match new_audio.data_mut() {
            AudioData::Mono(data) => {
                let fade_len = fade_samples.min(data.len());
                let start_idx = total_samples - fade_len;
                for (i, sample) in data.iter_mut().skip(start_idx).enumerate() {
                    let progress = i as f32 / fade_len as f32;
                    let fade_factor = (1.0 - progress).powf(2.0);
                    *sample *= fade_factor;
                }
            }
            AudioData::Stereo(data) => {
                let fade_len = fade_samples.min(data.nrows());
                let start_idx = total_samples - fade_len;
                for (i, mut row) in data.rows_mut().into_iter().skip(start_idx).enumerate() {
                    let progress = i as f32 / fade_len as f32;
                    let fade_factor = (1.0 - progress).powf(2.0);
                    row *= fade_factor;
                }
            }
        }

        Ok(new_audio)
    }

    pub fn concatenate(audio1: &WavAudio, audio2: &WavAudio) -> Result<WavAudio> {
        if audio1.sample_rate() != audio2.sample_rate() {
            return Err(ZipEnhancerError::audio(
                "Cannot concatenate audio files with different sample rates"
            ));
        }

        if audio1.channels() != audio2.channels() {
            return Err(ZipEnhancerError::audio(
                "Cannot concatenate audio files with different channel counts"
            ));
        }

        if audio1.format() != audio2.format() {
            return Err(ZipEnhancerError::audio(
                "Cannot concatenate audio files with different formats"
            ));
        }

        let new_data = match (audio1.data(), audio2.data()) {
            (AudioData::Mono(data1), AudioData::Mono(data2)) => {
                let mut combined = Array1::zeros(data1.len() + data2.len());
                combined.slice_mut(ndarray::s![..data1.len()]).assign(data1);
                combined.slice_mut(ndarray::s![data1.len()..]).assign(data2);
                AudioData::Mono(combined)
            }
            (AudioData::Stereo(data1), AudioData::Stereo(data2)) => {
                let total_rows = data1.nrows() + data2.nrows();
                let mut combined = Array2::zeros((total_rows, 2));
                combined.slice_mut(ndarray::s![..data1.nrows(), ..]).assign(data1);
                combined.slice_mut(ndarray::s![data1.nrows().., ..]).assign(data2);
                AudioData::Stereo(combined)
            }
            _ => unreachable!(),
        };

        let total_samples = new_data.len() as u32;
        let duration = total_samples as f64 / audio1.sample_rate() as f64;

        let mut header = audio1.header.clone();
        header.total_samples = total_samples;
        header.duration = duration;

        Ok(WavAudio {
            header,
            data: new_data,
        })
    }

    pub fn extract_segment(audio: &WavAudio, start_ms: f64, duration_ms: f64) -> Result<WavAudio> {
        let sample_rate = audio.sample_rate() as f64;
        let start_sample = (start_ms / 1000.0 * sample_rate) as usize;
        let end_sample = ((start_ms + duration_ms) / 1000.0 * sample_rate) as usize;

        if start_sample >= audio.data().len() {
            return Err(ZipEnhancerError::audio("Start position exceeds audio range"));
        }

        let end_sample = end_sample.min(audio.data().len());
        let segment_length = end_sample - start_sample;

        let new_data = match audio.data() {
            AudioData::Mono(data) => {
                AudioData::Mono(data.slice(ndarray::s![start_sample..end_sample]).to_owned())
            }
            AudioData::Stereo(data) => {
                AudioData::Stereo(data.slice(ndarray::s![start_sample..end_sample, ..]).to_owned())
            }
        };

        let mut header = audio.header.clone();
        header.total_samples = segment_length as u32;
        header.duration = segment_length as f64 / audio.sample_rate() as f64;

        Ok(WavAudio {
            header,
            data: new_data,
        })
    }

    pub fn convert_data_type(data: &[i16]) -> Vec<f32> {
        data.iter().map(|&x| x as f32 / 32767.0).collect()
    }

    pub fn convert_data_type_reverse(data: &[f32]) -> Vec<i16> {
        data.iter()
            .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    pub fn validate_audio_range(data: &[f32]) -> Result<()> {
        for (i, &sample) in data.iter().enumerate() {
            if !sample.is_finite() {
                return Err(ZipEnhancerError::audio(
                    format!("Audio data at position {} contains invalid value (NaN or Inf)", i)
                ));
            }
            if sample < -1.0 || sample > 1.0 {
                return Err(ZipEnhancerError::audio(
                    format!("Audio data at position {} out of range [-1.0, 1.0]: {}", i, sample)
                ));
            }
        }
        Ok(())
    }

    pub fn calculate_rms(audio: &WavAudio) -> f32 {
        match audio.data() {
            AudioData::Mono(data) => {
                let sum_squares: f32 = data.iter().map(|&x| x * x).sum();
                (sum_squares / data.len() as f32).sqrt()
            }
            AudioData::Stereo(data) => {
                let sum_squares: f32 = data.iter().map(|&x| x * x).sum();
                (sum_squares / (data.len() * 2) as f32).sqrt()
            }
        }
    }

    pub fn calculate_peak(audio: &WavAudio) -> f32 {
        match audio.data() {
            AudioData::Mono(data) => {
                data.iter().map(|&x| x.abs()).fold(0.0, f32::max)
            }
            AudioData::Stereo(data) => {
                data.iter().map(|&x| x.abs()).fold(0.0, f32::max)
            }
        }
    }

    pub fn detect_silence(audio: &WavAudio, threshold: f32) -> bool {
        let peak = Self::calculate_peak(audio);
        peak < threshold
    }

    pub fn trim_silence(audio: &WavAudio, threshold: f32, margin_ms: f64) -> Result<WavAudio> {
        let sample_rate = audio.sample_rate() as f64;
        let margin_samples = (margin_ms / 1000.0 * sample_rate) as usize;

        match audio.data() {
            AudioData::Mono(data) => {
                let start_idx = Self::find_start_non_silent(data.view(), threshold)?.max(margin_samples) - margin_samples;
                let end_idx = Self::find_end_non_silent(data.view(), threshold)? + margin_samples;
                let end_idx = end_idx.min(data.len());

                if start_idx >= end_idx {
                    return Err(ZipEnhancerError::audio("Audio is completely silent"));
                }

                let new_data = data.slice(ndarray::s![start_idx..end_idx]).to_owned();
                let mut header = audio.header.clone();
                header.total_samples = new_data.len() as u32;
                header.duration = new_data.len() as f64 / audio.sample_rate() as f64;

                Ok(WavAudio {
                    header,
                    data: AudioData::Mono(new_data),
                })
            }
            AudioData::Stereo(data) => {
                let left = data.column(0);
                let right = data.column(1);

                let start_idx_left = Self::find_start_non_silent(left.view(), threshold)?;
                let start_idx_right = Self::find_start_non_silent(right.view(), threshold)?;
                let start_idx = start_idx_left.max(start_idx_right).max(margin_samples) - margin_samples;

                let end_idx_left = Self::find_end_non_silent(left.view(), threshold)?;
                let end_idx_right = Self::find_end_non_silent(right.view(), threshold)?;
                let end_idx = end_idx_left.min(end_idx_right) + margin_samples;
                let end_idx = end_idx.min(data.nrows());

                if start_idx >= end_idx {
                    return Err(ZipEnhancerError::audio("Audio is completely silent"));
                }

                let new_data = data.slice(ndarray::s![start_idx..end_idx, ..]).to_owned();
                let mut header = audio.header.clone();
                header.total_samples = new_data.nrows() as u32;
                header.duration = new_data.nrows() as f64 / audio.sample_rate() as f64;

                Ok(WavAudio {
                    header,
                    data: AudioData::Stereo(new_data),
                })
            }
        }
    }

    fn find_start_non_silent(data: ArrayView1<f32>, threshold: f32) -> Result<usize> {
        for (i, &sample) in data.iter().enumerate() {
            if sample.abs() > threshold {
                return Ok(i);
            }
        }
        Err(ZipEnhancerError::audio("Audio is completely silent"))
    }

    fn find_end_non_silent(data: ArrayView1<f32>, threshold: f32) -> Result<usize> {
        for (i, &sample) in data.iter().enumerate().rev() {
            if sample.abs() > threshold {
                return Ok(data.len() - i - 1);
            }
        }
        Err(ZipEnhancerError::audio("Audio is completely silent"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_data_type_conversion() {
        let int_data = vec![32767, 0, -32767];
        let float_data = AudioConverter::convert_data_type(&int_data);
        assert!((float_data[0] - 1.0).abs() < f32::EPSILON);
        assert!((float_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((float_data[2] + 1.0).abs() < f32::EPSILON);

        let int_data_back = AudioConverter::convert_data_type_reverse(&float_data);
        assert_eq!(int_data, int_data_back);
    }

    #[test]
    fn test_audio_range_validation() {
        let valid_data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        assert!(AudioConverter::validate_audio_range(&valid_data).is_ok());

        let invalid_data1 = vec![1.5]; // Out of range
        assert!(AudioConverter::validate_audio_range(&invalid_data1).is_err());

        let invalid_data2 = vec![f32::NAN]; // NaN
        assert!(AudioConverter::validate_audio_range(&invalid_data2).is_err());

        let invalid_data3 = vec![f32::INFINITY]; // Inf
        assert!(AudioConverter::validate_audio_range(&invalid_data3).is_err());
    }

    #[test]
    fn test_rms_and_peak_calculation() {
        let data = Array1::from(vec![0.5, -0.5, 0.8, -0.8]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);

        let rms = AudioConverter::calculate_rms(&audio);
        let expected_rms = ((0.25 + 0.25 + 0.64 + 0.64) / 4.0_f32).sqrt();
        assert!((rms - expected_rms).abs() < f32::EPSILON);

        let peak = AudioConverter::calculate_peak(&audio);
        assert_eq!(peak, 0.8);
    }

    #[test]
    fn test_silence_detection() {
        let silent_data = Array1::from(vec![0.0, 0.0, 0.0]);
        let silent_audio = WavAudio::new_mono(16000, silent_data, AudioFormat::Float32);
        assert!(AudioConverter::detect_silence(&silent_audio, 0.01));

        let audio_data = Array1::from(vec![0.0, 0.5, 0.0]);
        let audio = WavAudio::new_mono(16000, audio_data, AudioFormat::Float32);
        assert!(!AudioConverter::detect_silence(&audio, 0.01));
    }

    #[test]
    fn test_normalize() {
        let data = Array1::from(vec![0.5, -1.0, 0.8]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);
        let normalized = AudioConverter::normalize(&audio).unwrap();

        let peak = AudioConverter::calculate_peak(&normalized);
        assert!((peak - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_application() {
        let data = Array1::from(vec![0.5, 0.0, -0.5]);
        let audio = WavAudio::new_mono(16000, data, AudioFormat::Float32);
        
        // Apply 6dB gain (approximately 2.0x)
        let with_gain = AudioConverter::apply_gain(&audio, 6.0).unwrap();
        
        match with_gain.data() {
            AudioData::Mono(data) => {
                assert!((data[0] - 1.0).abs() < 1e-2, "Expected {} to be close to 1.0", data[0]);
                assert!((data[2] + 1.0).abs() < 1e-2, "Expected {} to be close to -1.0", data[2]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_concatenation() {
        let data1 = Array1::from(vec![0.1, 0.2]);
        let data2 = Array1::from(vec![0.3, 0.4]);
        let audio1 = WavAudio::new_mono(16000, data1, AudioFormat::Float32);
        let audio2 = WavAudio::new_mono(16000, data2, AudioFormat::Float32);

        let combined = AudioConverter::concatenate(&audio1, &audio2).unwrap();
        assert_eq!(combined.total_samples(), 4);
        
        match combined.data() {
            AudioData::Mono(data) => {
                assert_eq!(data[0], 0.1);
                assert_eq!(data[1], 0.2);
                assert_eq!(data[2], 0.3);
                assert_eq!(data[3], 0.4);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_concatenation_incompatible() {
        let data1 = Array1::from(vec![0.1, 0.2]);
        let data2 = Array1::from(vec![0.3, 0.4]);
        let audio1 = WavAudio::new_mono(16000, data1, AudioFormat::Float32);
        let audio2 = WavAudio::new_mono(8000, data2, AudioFormat::Float32); // Different sample rate

        let result = AudioConverter::concatenate(&audio1, &audio2);
        assert!(result.is_err());
    }

    #[test]
    fn test_channel_conversion() {
        let mono_data = Array1::from(vec![0.1, 0.2]);
        let mono_audio = WavAudio::new_mono(16000, mono_data, AudioFormat::Float32);
        let stereo_audio = AudioConverter::convert_channels(&mono_audio, 2).unwrap();

        assert_eq!(stereo_audio.channels(), 2);
        match stereo_audio.data() {
            AudioData::Stereo(data) => {
                assert_eq!(data[[0, 0]], 0.1);
                assert_eq!(data[[0, 1]], 0.1);
                assert_eq!(data[[1, 0]], 0.2);
                assert_eq!(data[[1, 1]], 0.2);
            }
            _ => unreachable!(),
        }

        let stereo_data = Array2::from(vec![[0.2, 0.4], [0.6, 0.8]]);
        let stereo_audio2 = WavAudio::new_stereo(16000, stereo_data, AudioFormat::Float32).unwrap();
        let mono_audio2 = AudioConverter::convert_channels(&stereo_audio2, 1).unwrap();

        assert_eq!(mono_audio2.channels(), 1);
        match mono_audio2.data() {
            AudioData::Mono(data) => {
                assert!((data[0] - 0.3).abs() < 1e-6, "Expected {} to be close to 0.3", data[0]);
                assert!((data[1] - 0.7).abs() < 1e-6, "Expected {} to be close to 0.7", data[1]);
            }
            _ => unreachable!(),
        }
    }
}