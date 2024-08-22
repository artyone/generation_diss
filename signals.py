import numpy as np

class SinusSignal:
    def __init__(
        self, chunk_size, sampling_rate,
        freq1, freq2, freq3,
        amp1, amp2, amp3,
        phase1, phase2, phase3,
    ):
        self.freq_unch0 = freq1
        self.freq_unch1 = freq2
        self.freq_unch2 = freq3
        self.amp_unch0 = amp1
        self.amp_unch1 = amp2
        self.amp_unch2 = amp3
        self.phase_unch0_ch2 = np.deg2rad(phase1)
        self.phase_unch1_ch2 = np.deg2rad(phase2)
        self.phase_unch2_ch2 = np.deg2rad(phase3)
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.current_index = 0
        self.current_channel = None
        self.default_length_signal = 2 * np.pi
        self.init_signals()
    
    def init_signals(self):
        self.signal_noise = self.gen_signal(ch=None)
        self.signal_ch0 = self.gen_signal(ch=0)
        self.signal_ch1 = self.gen_signal(ch=1)
        self.signal_ch2 = self.gen_signal(ch=2)

    def gen_signal(self, ch):
        if ch is not None:
            freq = getattr(self, f'freq_unch{ch}')
            amp = getattr(self, f'amp_unch{ch}')
            phase_ch2 = getattr(self, f'phase_unch{ch}_ch2')
        else:
            freq = amp = phase_ch2 = 0

        t = np.arange(0, 1, 1 / self.sampling_rate)
        signal_left = amp * np.sin(t * 2 * np.pi * freq)
        signal_right = amp * np.sin(t * 2 * np.pi * freq + phase_ch2)
        signal_stereo = np.empty((len(t), 2))
        signal_stereo[:, 0] = signal_left
        signal_stereo[:, 1] = signal_right
        return signal_stereo.astype(np.float32)
    
    def get_next_chunk(self):

        if self.current_channel is not None:
            current_signal = getattr(self, f'signal_ch{self.current_channel}')
        else:
            current_signal = self.signal_noise
        current_index = self.current_index
        if len(current_signal) <= (current_index + self.chunk_size):
            chunk = np.vstack((current_signal[current_index:], current_signal[:self.chunk_size - (len(current_signal) - current_index)]))
            self.current_index = self.chunk_size - (len(current_signal) - current_index)
        else:
            chunk = current_signal[current_index:current_index + self.chunk_size]
            self.current_index += self.chunk_size
        return chunk
    
    def update_settings(
        self,
        freq1, freq2, freq3,
        amp1, amp2, amp3,
        phase1, phase2, phase3,
    ):
        self.freq_unch0 = freq1
        self.freq_unch1 = freq2
        self.freq_unch2 = freq3
        self.amp_unch0 = amp1
        self.amp_unch1 = amp2
        self.amp_unch2 = amp3
        self.phase_unch0_ch2 = np.deg2rad(phase1)
        self.phase_unch1_ch2 = np.deg2rad(phase2)
        self.phase_unch2_ch2 = np.deg2rad(phase3)
        self.init_signals()
        self.current_index = 0

    def switch_signal(self, channel):
        self.current_channel = channel
        # self.current_index = 0