use ndarray::Ix3;
use ndarray::{Array2, Array3};
use ort::session::Session;

use std::path::Path;

#[derive(Debug, PartialEq)]
enum VadStateMachine {
    StartPointNotDetected,
    InSpeechSegment,
    EndPointDetected
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum FrameState {
    Speech,
    Sil,
}

// 语音状态变化
/// Represents transitions between audio states.
///
/// This enum defines the possible changes between silence and speech states.
#[derive(Debug, PartialEq)]
enum AudioChangeState {
    /// Transition from silence to silence
    Sil2Sil,
    /// Transition from silence to speech
    Sil2Speech,
    /// Transition from speech to speech
    Speech2Speech,
    /// Transition from speech to silence
    Speech2Sil,
}

// 滑動窗口檢測器
/// A sliding window detector for identifying speech transitions.
///
/// This structure maintains a window of frame states to detect changes between silence and speech.
#[derive(Debug)]
struct WindowDetector {
    /// Size of the window in frames.
    win_size_frame: usize,
    /// Threshold of consecutive speech frames to transition from silence to speech.
    sil_to_speech_frmcnt_thres: usize,
    /// Threshold of consecutive silence frames to transition from speech to silence.
    speech_to_sil_frmcnt_thres: usize,
    /// Sum of speech frames (1s) within the window.
    win_sum: usize,
    /// Vector of frame states within the window (0 for silence, 1 for speech).
    win_state: Vec<usize>,
    /// Current position within the window, wraps around when reaching the end.
    cur_win_pos: usize,
    /// Previous frame state for tracking transitions.
    pre_frame_state: FrameState,
}

/// Implementation of methods for `WindowDetector`.
impl WindowDetector {
    /// Creates a new `WindowDetector` instance with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `window_size_ms` - Size of the sliding window in milliseconds.
    /// * `sil_to_speech_time` - Time threshold in milliseconds to transition from silence to speech.
    /// * `speech_to_sil_time` - Time threshold in milliseconds to transition from speech to silence.
    /// * `frame_size_ms` - Size of each frame in milliseconds.
    ///
    /// # Returns
    ///
    /// A new `WindowDetector` instance initialized with the given parameters.
    fn new(
        window_size_ms: usize,
        sil_to_speech_time: usize,
        speech_to_sil_time: usize,
        frame_size_ms: usize,
    ) -> Self {
        let win_size_frame = window_size_ms / frame_size_ms;
        WindowDetector {
            win_size_frame,
            sil_to_speech_frmcnt_thres: sil_to_speech_time / frame_size_ms,
            speech_to_sil_frmcnt_thres: speech_to_sil_time / frame_size_ms,
            win_sum: 0,
            win_state: vec![0; win_size_frame],
            cur_win_pos: 0,
            pre_frame_state: FrameState::Sil,
        }
    }

    /// Detects the state change for a single frame within the sliding window.
    ///
    /// Updates the window state and returns the detected transition.
    ///
    /// # Arguments
    ///
    /// * `frame_state` - The state of the current frame (silence or speech).
    ///
    /// # Returns
    ///
    /// An `AudioChangeState` indicating the transition detected based on the window.
    fn detect_one_frame(&mut self, frame_state: FrameState) -> AudioChangeState {
        let cur_state = if frame_state == FrameState::Speech {
            1
        } else {
            0
        };
        self.win_sum -= self.win_state[self.cur_win_pos]; // 移除舊狀態
        self.win_sum += cur_state; // 添加新狀態
        self.win_state[self.cur_win_pos] = cur_state;
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame;

        if self.pre_frame_state == FrameState::Sil
            && self.win_sum >= self.sil_to_speech_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Speech;
            AudioChangeState::Sil2Speech
        } else if self.pre_frame_state == FrameState::Speech
            && self.win_sum < self.speech_to_sil_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Sil;
            AudioChangeState::Speech2Sil
        } else if self.pre_frame_state == FrameState::Sil {
            AudioChangeState::Sil2Sil
        } else {
            AudioChangeState::Speech2Speech
        }
    }
}

/// Configuration options for the FSMN VAD system.
///
/// This structure defines the parameters used to configure voice activity detection.
#[derive(Debug, Clone, Copy)]
pub struct VADXOptions {
    /// Sample rate of the audio in Hz (e.g., 16000).
    sample_rate: u32,
    /// Frame interval in milliseconds.
    frame_in_ms: usize,
    /// Frame length in milliseconds.
    frame_length_ms: usize,
    /// Decibel threshold for silence detection.
    decibel_thres: f32,
    /// Signal-to-noise ratio threshold.
    snr_thres: f32,
    /// Threshold for the difference between speech and silence probabilities.
    speech_noise_thres: f32,
    /// Maximum silence duration at the end of a segment in milliseconds.
    max_end_silence_time: usize,
    /// Size of the sliding window in milliseconds.
    window_size_ms: usize,
    /// Time threshold in milliseconds to transition from silence to speech.
    sil_to_speech_time_thres: usize,
    /// Time threshold in milliseconds to transition from speech to silence.
    speech_to_sil_time_thres: usize,
    /// Maximum duration of a single speech segment in milliseconds.
    max_single_segment_time: usize,
    /// Detection mode (single or multiple utterances).
    detect_mode: VadDetectMode,
    /// Ratio of speech to noise for probability calculation.
    speech_2_noise_ratio: f32,
    /// Number of noise frames used for SNR calculation.
    noise_frame_num_used_for_snr: f32,
}

/// Implementation of methods for `VADXOptions`.
impl VADXOptions {
    /// Creates a `VADXOptions` instance with default values.
    ///
    /// # Returns
    ///
    /// A `VADXOptions` instance initialized with sensible defaults.
    pub fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_in_ms: 10,
            frame_length_ms: 25,
            decibel_thres: -100.0,
            snr_thres: -100.0,
            speech_noise_thres: 0.6,
            max_end_silence_time: 500,
            window_size_ms: 200,
            sil_to_speech_time_thres: 150,
            speech_to_sil_time_thres: 150,
            max_single_segment_time: 60000,
            speech_2_noise_ratio: 1.0,
            noise_frame_num_used_for_snr: 100.0,
            detect_mode: VadDetectMode::default(),
        }
    }
}

/// Represents metadata for a single audio frame.
///
/// This structure stores timing and segment boundary information for each frame.
#[derive(Debug, Clone)]
struct FrameData {
    /// Start time of the frame in milliseconds.
    start_ms: i32,
    /// End time of the frame in milliseconds.
    end_ms: i32,
    /// Indicates if the frame contains the start of a speech segment.
    contain_seg_start_point: bool,
    /// Indicates if the frame contains the end of a speech segment.
    contain_seg_end_point: bool,
}

/// Defines the detection mode for VAD.
///
/// This enum specifies whether to detect a single utterance or multiple utterances.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum VadDetectMode {
    /// Detects a single continuous utterance.
    SingleUtterance,
    /// Detects multiple utterances within the audio (default).
    #[default]
    MutipleUtterance,
}

/// Implements voice activity detection using an FSMN model.
///
/// This structure manages the VAD process, including feature processing, state tracking, and segment detection.
#[derive(Debug)]
pub struct FSMNVad {
    /// ONNX session for running the FSMN model.
    session: Session,
    /// Decibel values for each frame.
    decibel: Vec<f32>,
    /// Buffer storing frame metadata.
    output_data_buf: Vec<FrameData>,
    /// Offset into the output buffer for processed segments.
    output_data_buf_offset: usize,
    /// Configuration options for VAD.
    opts: VADXOptions,
    /// Current state of the VAD state machine.
    vad_state: VadStateMachine,
    /// Sliding window detector for state transitions.
    window_detector: WindowDetector,
    /// Total number of frames processed.
    frm_cnt: usize,
    /// Count of consecutive silence frames.
    continous_silence_frame_count: usize,
    /// Index of the confirmed start frame of a speech segment.
    confirmed_start_frame: Option<usize>,
    /// Index of the confirmed end frame of a speech segment.
    confirmed_end_frame: Option<usize>,
    /// Index of the most recent confirmed speech frame.
    latest_confirmed_speech_frame: usize,
}

/// Implementation of methods for `FSMNVad`.
impl FSMNVad {
    /// Creates a new `FSMNVad` instance with a specified model and options.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file for VAD.
    /// * `opts` - Configuration options for VAD.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `FSMNVad` instance or an error if model loading fails.
    ///
    /// # Errors
    ///
    /// Returns an `ort::Error` if the ONNX session cannot be created from the model file.
    ///
    /// # Example
    ///
    /// ```
    /// use fsmn_vad::{FSMNVad, VADXOptions};
    /// use std::path::Path;
    ///
    /// let opts = VADXOptions::default();
    /// let vad = FSMNVad::new(Path::new("model.onnx"), opts)
    ///     .expect("Failed to initialize FSMNVad");
    /// ```
    pub fn new(model_path: impl AsRef<Path>, opts: VADXOptions) -> Result<Self, ort::Error> {
        let session = ort::session::Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            decibel: Vec::new(),
            output_data_buf: Vec::new(),
            output_data_buf_offset: 0,
            opts,
            vad_state: VadStateMachine::StartPointNotDetected,
            window_detector: WindowDetector::new(
                opts.window_size_ms,
                opts.sil_to_speech_time_thres,
                opts.speech_to_sil_time_thres,
                opts.frame_in_ms,
            ),
            frm_cnt: 0,
            continous_silence_frame_count: 0,
            confirmed_start_frame: None,
            confirmed_end_frame: None,
            latest_confirmed_speech_frame: 0,
        })
    }

    /// Determines the state of a single frame based on decibel, SNR, and model scores.
    ///
    /// # Arguments
    ///
    /// * `frame_idx` - Index of the frame within the scores array.
    /// * `noise_average_decibel` - Mutable reference to the average noise decibel, updated during processing.
    /// * `scores` - 3D array of model output scores.
    ///
    /// # Returns
    ///
    /// A `FrameState` indicating whether the frame is silence or speech.
    fn get_frame_state(
        &self,
        frame_idx: usize,
        noise_average_decibel: &mut f32,
        scores: &Array3<f32>,
    ) -> FrameState {
        let cur_decibel = self.decibel[frame_idx];
        let cur_snr = cur_decibel - *noise_average_decibel;
        if cur_decibel < self.opts.decibel_thres {
            return FrameState::Sil;
        }

        let sil_pdf_ids = &[0]; // 假設 sil_pdf_ids = [0]，可根據實際情況調整
        let sil_pdf_scores: Vec<f32> = sil_pdf_ids
            .iter()
            .map(|&id| scores[[0, frame_idx, id as usize]])
            .collect();
        let sum_score: f32 = sil_pdf_scores.iter().sum();

        let epsilon = 1e-10; // 避免 log(0)
        let noise_prob = (sum_score + epsilon).ln() * self.opts.speech_2_noise_ratio;
        let speech_prob = (1.0 - sum_score + epsilon).ln();

        if speech_prob.exp() >= noise_prob.exp() + self.opts.speech_noise_thres {
            if cur_snr >= self.opts.snr_thres && cur_decibel >= self.opts.decibel_thres {
                return FrameState::Speech;
            }
        } else {
            if *noise_average_decibel < -99.9 {
                *noise_average_decibel = cur_decibel;
            } else {
                *noise_average_decibel = (cur_decibel
                    + *noise_average_decibel * (self.opts.noise_frame_num_used_for_snr - 1.0))
                    / self.opts.noise_frame_num_used_for_snr;
            }
        }
        FrameState::Sil
    }

    /// Performs voice activity detection on audio features and waveform.
    ///
    /// Processes the input audio to detect speech segments, returning their start and end times.
    ///
    /// # Arguments
    ///
    /// * `feats` - 2D array of audio features.
    /// * `waveform` - Raw audio samples as 16-bit integers.
    /// * `is_final` - Indicates whether this is the final batch of audio to process.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of tuples, each representing the start and end times (in milliseconds) of detected speech segments.
    ///
    /// # Errors
    ///
    /// Returns an error if feature processing, score computation, or segment detection fails.
    ///
    /// # Example
    ///
    /// ```
    /// use fsmn_vad::{FSMNVad, VADXOptions};
    /// use ndarray::Array2;
    /// use std::path::Path;
    ///
    /// let opts = VADXOptions::default();
    /// let mut vad = FSMNVad::new(Path::new("model.onnx"), opts).expect("Failed to initialize");
    /// let feats = Array2::zeros((100, 400)); // Example features
    /// let waveform = vec![0_i16; 16000]; // Example waveform
    /// let segments = vad.infer_vad(feats, &waveform, true)
    ///     .expect("Failed to perform VAD");
    /// println!("Detected segments: {:?}", segments);
    /// ```
    pub fn infer_vad(
        &mut self,
        feats: Array2<f32>,
        waveform: &[i16],
        is_final: bool,
    ) -> Result<Vec<(i32, i32)>, Box<dyn std::error::Error>> {
        self.reset_detection(); // 確保重置所有狀態
        self.compute_decibel(waveform);

        let score = self.compute_scores(feats)?;

        if !is_final {
            self.detect_common_frames(&score)?;
        } else {
            self.detect_last_frames(&score)?;
        }

        let mut segments = Vec::new();
        let mut i = self.output_data_buf_offset;
        while i < self.output_data_buf.len() {
            let frame = &self.output_data_buf[i];
            if frame.contain_seg_start_point && frame.contain_seg_end_point {
                segments.push((frame.start_ms, frame.end_ms)); // 收集完整的語音段
                self.output_data_buf_offset += 1; // 更新偏移量
            }
            i += 1;
        }

        if is_final {
            self.reset_detection();
        }

        Ok(segments)
    }

    /// Computes decibel values for each frame from the waveform.
    ///
    /// Updates the `decibel` field with calculated values based on frame energy.
    ///
    /// # Arguments
    ///
    /// * `waveform` - Raw audio samples as 16-bit integers.
    fn compute_decibel(&mut self, waveform: &[i16]) {
        let frame_size =
            (self.opts.frame_length_ms as f32 * self.opts.sample_rate as f32 / 1000.0) as usize;
        let frame_shift =
            (self.opts.frame_in_ms as f32 * self.opts.sample_rate as f32 / 1000.0) as usize;
        self.decibel = (0..(waveform.len() - frame_size + 1))
            .step_by(frame_shift)
            .map(|i| {
                let frame = &waveform[i..i + frame_size];
                // 正規化到 [-1.0, 1.0]
                let energy: f32 = frame.iter().map(|&x| (x as f32 / 32768.0).powi(2)).sum();
                10.0 * (energy + 1e-6).log10()
            })
            .collect();
    }

    /// Computes VAD scores from audio features using the ONNX model.
    ///
    /// # Arguments
    ///
    /// * `feats` - 2D array of audio features.
    ///
    /// # Returns
    ///
    /// A `Result` containing a 3D array of scores from the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX session fails to run or if tensor extraction fails.
    fn compute_scores(
        &mut self,
        feats: Array2<f32>,
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        // input:name: speech, tensor: float32[1,feats_length,400]
        // output:name: logits, tensor: float32[1,Softmaxlogits_dim_1,248] (不要再眼瞎看錯了，是3維，那不是逗點)

        let feats_3d = feats.insert_axis(ndarray::Axis(0));
        let inputs = ort::inputs![
            "speech" => feats_3d.clone()
        ]?;
        let outputs = self.session.run(inputs)?;
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let dyn_to_fix3array = logits.into_dimensionality::<Ix3>()?;

        Ok(dyn_to_fix3array.to_owned())
    }

    /// Detects speech frames in a common (non-final) batch.
    ///
    /// Updates the state machine and output buffer based on computed scores.
    ///
    /// # Arguments
    ///
    /// * `scores` - 3D array of VAD scores.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the detection process.
    ///
    /// # Errors
    ///
    /// Returns an error if internal processing fails (though unlikely in this implementation).
    fn detect_common_frames(
        &mut self,
        scores: &Array3<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.vad_state == VadStateMachine::EndPointDetected {
            return Ok(());
        }

        let mut noise_average_decibel = -100.0; // 這裡需要動態計算，暫用默認值
        for i in (0..scores.dim().1).rev() {
            let frame_idx = self.frm_cnt + scores.dim().1 - 1 - i; // 從最新幀向前，不要再改了 適應realtime
            let frame_state = self.get_frame_state(i, &mut noise_average_decibel, scores); // 使用本地索引 i
            let state_change = self.window_detector.detect_one_frame(frame_state);

            match state_change {
                AudioChangeState::Sil2Speech => {
                    self.continous_silence_frame_count = 0; // 清空靜音計數
                    if self.vad_state == VadStateMachine::StartPointNotDetected {
                        let start_frame =
                            frame_idx.saturating_sub(self.window_detector.win_size_frame); // 計算語音開始幀
                        self.confirmed_start_frame = Some(start_frame); // 記錄開始幀
                        self.vad_state = VadStateMachine::InSpeechSegment; // 進入語音段狀態
                        self.output_data_buf.push(FrameData {
                            start_ms: (start_frame * self.opts.frame_in_ms) as i32,
                            end_ms: (frame_idx * self.opts.frame_in_ms) as i32, // 初始結束時間為當前幀
                            contain_seg_start_point: true,
                            contain_seg_end_point: false,
                        }); // 添加新的語音段到緩衝區
                    }
                }
                AudioChangeState::Speech2Speech => {
                    self.continous_silence_frame_count = 0;
                    if self.vad_state == VadStateMachine::InSpeechSegment {
                        if let Some(start) = self.confirmed_start_frame {
                            if frame_idx >= start
                                && frame_idx - start
                                    > self.opts.max_single_segment_time / self.opts.frame_in_ms
                            {
                                self.confirmed_end_frame = Some(frame_idx);
                                self.vad_state = VadStateMachine::EndPointDetected;
                                self.output_data_buf.last_mut().unwrap().end_ms =
                                    (frame_idx * self.opts.frame_in_ms) as i32;
                                self.output_data_buf
                                    .last_mut()
                                    .unwrap()
                                    .contain_seg_end_point = true;
                                // 多語句模式下重置
                                if self.opts.detect_mode == VadDetectMode::MutipleUtterance {
                                    self.reset_detection();
                                }
                            } else {
                                self.latest_confirmed_speech_frame = frame_idx;
                                self.output_data_buf.last_mut().unwrap().end_ms =
                                    (frame_idx * self.opts.frame_in_ms) as i32;
                            }
                        }
                    }
                }
                AudioChangeState::Speech2Sil => {
                    self.continous_silence_frame_count = 0; // 重置靜音計數
                    if self.vad_state == VadStateMachine::InSpeechSegment {
                        self.confirmed_end_frame = Some(frame_idx); // 記錄語音結束幀
                        self.vad_state = VadStateMachine::EndPointDetected; // 更新狀態
                        self.output_data_buf.last_mut().unwrap().end_ms =
                            (frame_idx * self.opts.frame_in_ms) as i32; // 更新結束時間
                        self.output_data_buf
                            .last_mut()
                            .unwrap()
                            .contain_seg_end_point = true; // 標記段落結束
                        if self.opts.detect_mode == VadDetectMode::MutipleUtterance {
                            self.reset_detection(); // 在多語句模式下重置狀態機
                        }
                    }
                }
                AudioChangeState::Sil2Sil => {
                    self.continous_silence_frame_count += 1;
                    if self.vad_state == VadStateMachine::InSpeechSegment
                        && self.continous_silence_frame_count * self.opts.frame_in_ms
                            >= self.opts.max_end_silence_time
                    {
                        self.confirmed_end_frame = Some(frame_idx);
                        self.vad_state = VadStateMachine::EndPointDetected;
                        self.output_data_buf.last_mut().unwrap().end_ms =
                            (frame_idx * self.opts.frame_in_ms) as i32;
                        self.output_data_buf
                            .last_mut()
                            .unwrap()
                            .contain_seg_end_point = true;
                    }
                }
            }
        }
        self.frm_cnt += scores.dim().1;

        Ok(())
    }

    /// Detects speech frames in the final batch of audio.
    ///
    /// Ensures that any ongoing speech segment is properly closed when processing the last batch.
    ///
    /// # Arguments
    ///
    /// * `scores` - 3D array of VAD scores.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the detection process.
    ///
    /// # Errors
    ///
    /// Returns an error if common frame detection fails.
    fn detect_last_frames(
        &mut self,
        scores: &Array3<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.detect_common_frames(scores)?;
        if self.vad_state == VadStateMachine::InSpeechSegment {
            let last_frame = self.frm_cnt + scores.dim().1 - 1;
            self.confirmed_end_frame = Some(last_frame);
            self.output_data_buf.last_mut().unwrap().end_ms =
                (self.latest_confirmed_speech_frame * self.opts.frame_in_ms) as i32;
            self.output_data_buf
                .last_mut()
                .unwrap()
                .contain_seg_end_point = true;
            self.vad_state = VadStateMachine::EndPointDetected;
        }
        Ok(())
    }

    /// Resets the VAD detection state to its initial values.
    ///
    /// Clears all tracking variables and reinitializes the window detector.
    fn reset_detection(&mut self) {
        self.vad_state = VadStateMachine::StartPointNotDetected; // 重置狀態機
        self.continous_silence_frame_count = 0; // 重置靜音幀計數
        self.confirmed_start_frame = None; // 清空語音開始幀
        self.confirmed_end_frame = None; // 清空語音結束幀
        self.latest_confirmed_speech_frame = 0; // 重置最近確認的語音幀
        self.window_detector = WindowDetector::new(
            self.opts.window_size_ms,
            self.opts.sil_to_speech_time_thres,
            self.opts.speech_to_sil_time_thres,
            self.opts.frame_in_ms,
        ); // 重新創建 WindowDetector，確保其內部狀態清零
    }
}
