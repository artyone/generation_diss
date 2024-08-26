import pyaudio

class CallbackPlayer:
    def __init__(
            self, parent, signal, device_index: int
    ):
        self.parent = parent
        self.signal = signal
        self.chunk_size = self.signal.chunk_size
        self.sampling_rate = self.signal.sampling_rate
        self.device_index = device_index
        self.p = pyaudio.PyAudio()
        self.stream = None
        super().__init__()

    def callback_play(self, in_data, frame_count, time_info, status):
        current_signal = self.signal.get_next_chunk()
        return (current_signal.tobytes(), pyaudio.paContinue)
    
    def run(self):
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=self.sampling_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=self.device_index,
                stream_callback=self.callback_play,
            )
        except:
            self.stream = None
            self.p.terminate()
    
    def stop(self):
        if self.stream and self.stream.is_active():
            self.stream.close()
            self.stream = None
            self.p.terminate()
    
    def switch_signal(self, channel):
        self.signal.switch_signal(channel)