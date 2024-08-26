import dpkt
import numpy as np
from librosa import resample


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
        self.init_signals()

    def init_signals(self):
        self.signal_null = self.gen_signal(ch=None)
        self.signal_ch0 = self.gen_signal(ch=0)
        self.signal_ch1 = self.gen_signal(ch=1)
        self.signal_ch2 = self.gen_signal(ch=2)

    def gen_signal(self, ch):
        t = np.arange(0, 1, 1 / self.sampling_rate)
        if ch is None:
            signal_stereo = np.zeros((len(t), 2))
        else:
            freq = getattr(self, f'freq_unch{ch}')
            amp = getattr(self, f'amp_unch{ch}')
            phase_ch2 = getattr(self, f'phase_unch{ch}_ch2')
            signal_left = amp * np.sin(t * 2 * np.pi * freq)
            signal_right = amp * np.sin(t * 2 * np.pi * freq + phase_ch2)
            signal_stereo = np.empty((len(t), 2))
            signal_stereo[:, 0] = signal_left
            signal_stereo[:, 1] = signal_right
        return signal_stereo.astype(np.float32)

    def get_next_chunk(self):
        if self.current_channel is None:
            current_signal = self.signal_null
        elif self.current_channel == 'noise':
            current_signal = np.random.random_sample(
                (self.chunk_size, 2)) * 2 - 1
        else:
            current_signal = getattr(self, f'signal_ch{self.current_channel}')
        current_index = self.current_index
        if len(current_signal) <= (current_index + self.chunk_size):
            chunk = np.vstack(
                (current_signal[current_index:], current_signal[:self.chunk_size - (len(current_signal) - current_index)]))
            self.current_index = self.chunk_size - \
                (len(current_signal) - current_index)
        else:
            chunk = current_signal[current_index:current_index +
                                   self.chunk_size]
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
        if self.current_channel == 'noise':
            return
        self.current_channel = channel
        # self.current_index = 0

    def set_to_noise(self, param):
        if param:
            self.current_channel = 'noise'
        else:
            self.current_channel = None


class FileSignal:
    def __init__(self, chunk_size, sampling_rate, filenames, cycle_play=False):
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.filenames = filenames
        self.current_index_in_signal = 0
        self.current_index_in_channel = 0
        self.current_channel = None
        self.dt = np.dtype([
            ('cs', np.uint8),
            ('adr', np.uint8),
            ('time', np.uint32),
            ('data', np.int8, 31)
        ]).newbyteorder('>')
        self.cycle_play = cycle_play

        self.init_signals()

    def init_signals(self):
        list_bytes_data = self.get_bytes_data()
        unpacketed_data = np.frombuffer(list_bytes_data, dtype=self.dt)
        channels_data, self.times, self.fd = self.get_data_from_unpacketed_data(
            unpacketed_data)
        resampled_channels_data = self.resample_channels(channels_data)
        self.signal_ch0 = self.gen_signal(0, resampled_channels_data)
        self.signal_ch1 = self.gen_signal(1, resampled_channels_data)
        self.signal_ch2 = self.gen_signal(2, resampled_channels_data)

    def get_bytes_data(self):
        list_bytes = []
        cache = b''
        for filename in self.filenames:
            with open(filename, 'rb') as f:
                pcap = dpkt.pcapng.Reader(f)
                # способ без проверки ошибок
                # bytes_list = [pkt[42:-32] for _, pkt in pcap if len(pkt) == 1098]
                # list_bytes.extend(bytes_list)
                for num_pkt, (_, pkt) in enumerate(pcap):
                    if len(pkt) != 1098:
                        continue
                    pkt = pkt[42:-32]
                    start_byte = self.get_start_byte(pkt)
                    if start_byte is None:
                        print(f'Ошибка в cap {filename} не найден старт байт')
                        continue
                    cached_packet = cache + pkt[:start_byte]
                    if len(cached_packet) in {0, 37}:
                        list_bytes.append(cached_packet)
                    elif list_bytes:
                        print(
                            f'Ошибка в cap {filename}, номер пакета {num_pkt + 1}')
                    end_byte = (
                        start_byte + (len(pkt) - start_byte) // 37 * 37)
                    list_bytes.append(pkt[start_byte:end_byte])
                    cache = pkt[end_byte:]
        data_bytes = b''.join(list_bytes)
        return data_bytes

    @staticmethod
    def get_start_byte(byte_data):
        # начинаем итерацию по пакету данных
        for i in range(len(byte_data) - 74):
            # проверяем что контрольная сумма совпадает с контрольной суммой через 37 байт
            if not ((byte_data[i] == 83) and (byte_data[i + 37] == 83)):
                continue
            # проверяем что адр меньше 5 (хотя их всего 3 но может быть больше потом)
            if not ((byte_data[i + 1] < 5) and (byte_data[i + 38] < 5)):
                continue
            # проверяем что разница во времени между меньше 100 мкс
            current_time = int.from_bytes(
                byte_data[i + 2: i + 6], byteorder='big')
            next_time = int.from_bytes(
                byte_data[i + 39: i + 43], byteorder='big')
            time = (0 < (next_time - current_time) < 1000)
            if not time:
                continue
            # если проверки прошли выдаем стартовый индекс
            return i
        return None

    def get_data_from_unpacketed_data(self, unpacketed_data):
        only_adr1 = unpacketed_data[unpacketed_data['adr'] == 1]
        only_adr2 = unpacketed_data[unpacketed_data['adr'] == 2]

        channel_switches_adr1 = np.where(
            np.diff((only_adr1['data'][:, 12] & 0b0110_0000) >> 5) != 0)[0] + 1
        channel_switches_adr2 = np.where(
            np.diff(only_adr2['data'][:, 0] >> 2) != 0)[0] + 1

        times = []
        fd = []
        for val in channel_switches_adr1:
            times.append(only_adr1[val]['time'])
            current_data = only_adr1[val]['data']
            fd1 = (current_data[0] << 8).astype(
                np.int16) + current_data[1].astype(np.uint8)
            fd2 = (current_data[2] << 8).astype(
                np.int16) + current_data[3].astype(np.uint8)
            fd3 = (current_data[4] << 8).astype(
                np.int16) + current_data[5].astype(np.uint8)
            fd.append((fd1, fd2, fd3))

        channels = {
            0: {'unch0': [], 'unch1': []},
            1: {'unch0': [], 'unch1': []},
            2: {'unch0': [], 'unch1': []},
        }

        for j, k in zip(channel_switches_adr2, channel_switches_adr2[1:]):
            current_channel = only_adr2[j]['data'][0] >> 2
            channels[current_channel]['unch0'].append(
                only_adr2[j:k]['data'][:, 1::2].flatten())
            channels[current_channel]['unch1'].append(
                only_adr2[j:k]['data'][:, 2::2].flatten())

        return channels, times, fd

    def resample_channels(self, channels):
        for channel_data in channels.values():
            for unch in channel_data.values():
                for index, value in enumerate(unch):
                    prepared_data_for_resample = value.astype(np.float32)
                    resample_data = resample(
                        prepared_data_for_resample, orig_sr=50000, target_sr=self.sampling_rate)
                    # если потребуется экономия памяти
                    # result_data = resample_data / max(abs(resample_data)) * 127
                    # unch[index] = result_data.astype(np.int8)
                    unch[index] = resample_data / np.max(np.abs(resample_data))
        return channels

    def gen_signal(self, channel, channels_data):
        unch0 = channels_data[channel]['unch0']
        unch1 = channels_data[channel]['unch1']

        signal_stereo = [
            np.column_stack((i, j)) for i, j in zip(unch0, unch1)
        ]
        return signal_stereo

    def get_next_chunk(self):
        if self.current_channel is None:
            return np.zeros((self.chunk_size, 2))

        current_channel_data = getattr(
            self, f'signal_ch{self.current_channel}')
        if self.current_index_in_signal >= len(current_channel_data):
            if self.cycle_play:
                self.current_index_in_signal = 0
            else:
                return np.array([])

        current_signal = current_channel_data[self.current_index_in_signal]
        if len(current_signal) <= (self.current_index_in_channel + self.chunk_size):
            self.current_index_in_channel = 0
        current_index = self.current_index_in_channel
        chunk = current_signal[current_index:current_index +
                               self.chunk_size]
        self.current_index_in_channel += self.chunk_size
        return chunk

    def switch_signal(self, channel):
        if channel == 0:
            self.current_index_in_signal += 1
        self.current_channel = channel
        self.current_index_in_channel = 0

    def get_fd_and_times(self):
        try:
            fd = self.fd[self.current_index_in_signal * 3]
            times = self.times[self.current_index_in_signal * 3]
        except:
            return (0, 0, 0), 0
        return fd, times


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # signal = FileSignal(chunk_size=64,
    #                     sampling_rate=44100,
    #                     filenames=['sinus_to2.pcapng'])
    # b = signal.get_next_chunk()
    # signal.switch_signal(2)
    # b = signal.get_next_chunk()
    # signal.switch_signal(0)
    # a = signal.get_next_chunk()
    # print(a.shape)

    signal_sinus = SinusSignal(
        6400, 44100, 4000, 4000, 4000, 1, 1, 1, 90, 90, 90
    )
    signal_sinus.set_to_noise(False)
    signal_sinus.switch_signal(0)
    a_sinus = signal_sinus.get_next_chunk()
    # print(a_sinus)
    plt.plot(a_sinus)

    # plt.plot(a)
    # plt.plot(b)

    plt.show()
