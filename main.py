import sys
import time
from functools import partial

import numpy as np
import pyaudio
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtNetwork import QUdpSocket
from PyQt6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget)


class Signal:
    def __init__(
        self,
        freq1, freq2, freq3,
        amp1, amp2, amp3,
        phase1, phase2, phase3,
        chunk_size,
        sampling_rate, *args
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
        *args
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
        self.current_index = 0

class CallbackPlayer(QThread):
    def __init__(
            self, signal: Signal, chunk_size: int, sampling_rate: int, device_index: int
    ):
        
        self.signal = signal
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
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

class IndicatorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
            }
        """)

        layout = QHBoxLayout(self)

        fd1_layout = QVBoxLayout()
        self.fd1_lbl = QLabel('Fd1: -')
        fd1_layout.addWidget(self.fd1_lbl)
        self.tracking1_lbl = QLabel('Tracking1: ✘')
        fd1_layout.addWidget(self.tracking1_lbl)

        fd2_layout = QVBoxLayout()
        self.fd2_lbl = QLabel('Fd2: -')
        fd2_layout.addWidget(self.fd2_lbl)
        self.tracking2_lbl = QLabel('Tracking2: ✘')
        fd2_layout.addWidget(self.tracking2_lbl)

        fd3_layout = QVBoxLayout()
        self.fd3_lbl = QLabel('Fd3: -')
        fd3_layout.addWidget(self.fd3_lbl)
        self.tracking3_lbl = QLabel('Tracking3: ✘')
        fd3_layout.addWidget(self.tracking3_lbl)

        layout.addLayout(fd1_layout)
        layout.addLayout(fd2_layout)
        layout.addLayout(fd3_layout)

    def update_values(self, fd1, fd2, fd3, tracking1, tracking2, tracking3):
        self.fd1_lbl.setText(f'Fd1: {fd1}')
        self.tracking1_lbl.setText(f'Tracking1: {"✓" if tracking1 else "✘"}')
        self.fd2_lbl.setText(f'Fd2: {fd2}')
        self.tracking2_lbl.setText(f'Tracking2: {"✓" if tracking2 else "✘"}')
        self.fd3_lbl.setText(f'Fd3: {fd3}')
        self.tracking3_lbl.setText(f'Tracking3: {"✓" if tracking3 else "✘"}')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.socket = QUdpSocket()
        self.received_packets = 0
        self.fd1, self.fd2, self.fd3 = 0, 0, 0
        self.tracking1, self.tracking2, self.tracking3 = False, False, False
        self.counter_timer = QTimer(self)
        self.counter_timer.timeout.connect(self.update_info)
        self.cache_data = b''
        self.current_channel = None
        self.player = None
        self.last_change_time = 0
        self.initUI()

    def initUI(self):
        main_widget = QWidget(self)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        device_layout = QHBoxLayout()
        self.device_cmb = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(self.device_cmb)

        self.refresh_devices_btn = QPushButton('Обновить')
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)
        self.refresh_devices_btn.setFixedWidth(100)
        device_layout.addWidget(self.refresh_devices_btn)

        main_layout.addLayout(device_layout)

        sampling_rate_layout = QHBoxLayout()
        main_layout.addLayout(sampling_rate_layout)

        sampling_rate_label = QLabel('Частота дискретизации')
        sampling_rate_label.setFixedWidth(155)
        sampling_rate_layout.addWidget(sampling_rate_label)

        self.sampling_rate_le = QLineEdit(f'44100')
        self.sampling_rate_le.setPlaceholderText('Частота дискретизации:')
        self.sampling_rate_le.textChanged.connect(self.update_player_settings)
        sampling_rate_layout.addWidget(self.sampling_rate_le)

        freq_layout = QHBoxLayout()
        main_layout.addLayout(freq_layout)

        freq_label = QLabel('Частота (от 0 до 20000 Гц):')
        freq_label.setFixedWidth(155)
        freq_layout.addWidget(freq_label)

        self.freq1_le = QLineEdit('20')
        self.freq1_le.setPlaceholderText('Частота унч1')
        self.freq1_le.textChanged.connect(self.update_player_settings)
        freq_layout.addWidget(self.freq1_le)

        self.freq2_le = QLineEdit('200')
        self.freq2_le.setPlaceholderText('Частота унч2')
        self.freq2_le.textChanged.connect(self.update_player_settings)
        freq_layout.addWidget(self.freq2_le)

        self.freq3_le = QLineEdit('1000')
        self.freq3_le.setPlaceholderText('Частота унч3')
        self.freq3_le.textChanged.connect(self.update_player_settings)
        freq_layout.addWidget(self.freq3_le)

        amp_layout = QHBoxLayout()
        main_layout.addLayout(amp_layout)

        amp_label = QLabel('Амплитуда (от 0 до 1):')
        amp_label.setFixedWidth(155)
        amp_layout.addWidget(amp_label)

        self.amp1_le = QLineEdit('1')
        self.amp1_le.setPlaceholderText('Амплитуда унч1')
        self.amp1_le.textChanged.connect(self.update_player_settings)
        amp_layout.addWidget(self.amp1_le)

        self.amp2_le = QLineEdit('1')
        self.amp2_le.setPlaceholderText('Амплитуда унч2')
        self.amp2_le.textChanged.connect(self.update_player_settings)
        amp_layout.addWidget(self.amp2_le)

        self.amp3_le = QLineEdit('1')
        self.amp3_le.setPlaceholderText('Амплитуда унч3')
        self.amp3_le.textChanged.connect(self.update_player_settings)
        amp_layout.addWidget(self.amp3_le)

        ph_layout = QHBoxLayout()
        main_layout.addLayout(ph_layout)

        ph1_label = QLabel('Фаза 2 канала (от 0° до 360°):')
        ph1_label.setFixedWidth(155)
        ph_layout.addWidget(ph1_label)

        self.ph1_le = QLineEdit('90')
        self.ph1_le.setPlaceholderText('Фаза 2 канала унч1')
        self.ph1_le.textChanged.connect(self.update_player_settings)
        ph_layout.addWidget(self.ph1_le)

        self.ph2_le = QLineEdit('90')
        self.ph2_le.setPlaceholderText('Фаза 2 канала унч2')
        self.ph2_le.textChanged.connect(self.update_player_settings)
        ph_layout.addWidget(self.ph2_le)

        self.ph3_le = QLineEdit('90')
        self.ph3_le.setPlaceholderText('Фаза 2 канала унч3')
        self.ph3_le.textChanged.connect(self.update_player_settings)
        ph_layout.addWidget(self.ph3_le)

        chunk_size_label = QLabel('Размер чанка: ')
        self.chunk_size_le = QLineEdit('64')

        main_layout.addWidget(chunk_size_label)
        main_layout.addWidget(self.chunk_size_le)

        latency_label = QLabel('Задержка: ')
        self.latency_le = QLineEdit('1')

        main_layout.addWidget(latency_label)
        main_layout.addWidget(self.latency_le)

        self.send_btn = QPushButton('Старт')
        self.send_btn.clicked.connect(self.start_process)
        main_layout.addWidget(
            self.send_btn, alignment=Qt.AlignmentFlag.AlignBottom)

        self.indicator_widget = IndicatorWidget()
        main_layout.addWidget(self.indicator_widget)

        self.counter_label = QLabel(
            f'Получено пакетов: {self.received_packets}')
        main_layout.addWidget(self.counter_label)

        self.error_label = QLabel()
        main_layout.addWidget(self.error_label)

        self.setCentralWidget(main_widget)
        self.setWindowTitle('Генератор сигнала')
        self.show()

    def refresh_devices(self):
        self.device_cmb.clear()
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        devices = []
        for index in range(device_count):
            device = pa.get_device_info_by_index(index)
            device_name = device['name']
            device_host_api = device['hostApi']
            host_api_name = pa.get_host_api_info_by_index(device_host_api)['name']
            devices.append(f'{index}. {device_name} - {host_api_name}')
        self.device_cmb.addItems(devices)
        self.device_cmb.setCurrentIndex(int(pa.get_default_output_device_info()['index']))

    def get_values_from_form(self):
        settings = {
            'freq1': int(self.freq1_le.text()),
            'freq2': int(self.freq2_le.text()),
            'freq3': int(self.freq3_le.text()),
            'amp1': float(self.amp1_le.text()),
            'amp2': float(self.amp2_le.text()),
            'amp3': float(self.amp3_le.text()),
            'ph1': int(self.ph1_le.text()),
            'ph2': int(self.ph2_le.text()),
            'ph3': int(self.ph3_le.text()),
            'chunk_size': int(self.chunk_size_le.text()),
            'sampling_rate': int(self.sampling_rate_le.text()),
            'device_id': int(self.device_cmb.currentIndex())
        }
        return settings

    def update_player_settings(self):
        if self.player:
            try:
                settings = self.get_values_from_form()
            except:
                return
            self.signal.update_settings(*settings.values())

    def start_process(self):
        self.error_label.setText('')
        self.received_packets = 0

        try:
            settings = self.get_values_from_form()
        except:
            self.error_label.setText('Неправильно введены параметры')
            return

        self.signal = Signal(*settings.values())
        self.player = CallbackPlayer(
            signal= self.signal,
            chunk_size=settings['chunk_size'],
            sampling_rate=settings['sampling_rate'],
            device_index=settings['device_id']
        )

        self.player.start()
        self.player.wait()

        if self.player.stream is None:
            self.error_label.setText('Ошибка аудиосутройства')
            return
        
        self.socket.bind(2015)
        self.socket.readyRead.connect(self.read_udp_data)

        self.counter_timer.start(500)

        self.send_btn.setText('Стоп')
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.stop_process)

        self.block_line_edit(True)


    def stop_process(self):
        self.send_btn.setText('Старт')
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.start_process)

        self.block_line_edit(False)
        
        if self.player:
            self.player.stop()
            self.player = None

        self.socket.close()
        self.counter_timer.stop()
        
    def block_line_edit(self, param):
        self.refresh_devices_btn.setDisabled(param)
        self.sampling_rate_le.setDisabled(param)
        self.device_cmb.setDisabled(param)
        self.chunk_size_le.setDisabled(param)

    @staticmethod
    def get_start_byte(byte_data):
        start = None
        for index, value in enumerate(byte_data):
            if value == 83 and byte_data[index + 1] in [1, 2, 3] and byte_data[index+37] == 83:
                return index
        return start

    def channel_changed(self, byte_data):
        for i in range(0, len(byte_data), 37):
            # распаков по адр1
            # if byte_data[i+1] == 1:
            #     current_unch_channel = (byte_data[i+18] & 0b0110_0000) >> 5
            #     if self.current_channel != current_unch_channel and current_unch_channel in [0, 1, 2]:
            #         self.current_channel = current_unch_channel
            #         return True
            if byte_data[i+1] == 2:
                current_unch_channel = byte_data[i+6]  >> 2
                if self.current_channel != current_unch_channel and current_unch_channel in [0, 1, 2]:
                    self.current_channel = current_unch_channel
                    return True
        return False

    def read_udp_data(self):
        while self.socket.hasPendingDatagrams():
            self.received_packets += 1
            packet_data, * _ = self.socket.readDatagram(1056)
            packet_data = packet_data[:-32]

            if self.cache_data:
                packet_data = self.cache_data + packet_data

            start_byte = self.get_start_byte(packet_data)

            if start_byte is None:
                self.error_label.setText('Не нашёл начало данных')
                return

            end = (len(packet_data) - start_byte) // 37 * 37 + start_byte
            current_data = packet_data[start_byte: end]

            self.cache_data = packet_data[end:]

            self.udpate_fd(current_data)

            if self.player:
                
                if not self.channel_changed(current_data):
                    return
                
                switch_channel_timer = QTimer(self)
                switch_channel_timer.setSingleShot(True)
                switch_channel_timer.timeout.connect(partial(self.player.switch_signal, self.current_channel))
                switch_channel_timer.start(int(self.latency_le.text()) if self.latency_le.text() else 1)

                current_time = time.perf_counter_ns()
                print(f'Канал изменён на {self.current_channel}, Время смены: {(current_time - self.last_change_time) / 1e6:2f} мс')
                self.last_change_time = current_time

            

    def udpate_fd(self, byte_data):
        for i in range(0, len(byte_data), 37):
            if byte_data[i+1] == 1:
                self.fd1 = int.from_bytes(byte_data[i+6:i+8], signed=True)
                self.fd2 = int.from_bytes(byte_data[i+8:i+10], signed=True)
                self.fd3 = int.from_bytes(byte_data[i+10:i+12], signed=True)
                self.tracking1 = bool((byte_data[i+23] & 0b0100_0000) >> 6)
                self.tracking2 = bool((byte_data[i+23] & 0b1000_0000) >> 7)
                self.tracking3 = bool(byte_data[i+22] & 0b1)

    def update_info(self):
        self.counter_label.setText(
            f'Получено пакетов: {self.received_packets}')
        self.indicator_widget.update_values(
            self.fd1, self.fd2, self.fd3, self.tracking1, self.tracking2, self.tracking3)

    def close(self):
        if self.player:
            self.player.stop()
        self.socket.close()
        super().close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    sys.exit(app.exec())
