
import sys
import time
from collections import namedtuple
from functools import partial

import pyaudio
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtNetwork import QUdpSocket
from PyQt6.QtWidgets import (QApplication, QComboBox, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow,
                             QPushButton, QTabWidget, QVBoxLayout, QWidget, QGroupBox)

from player import CallbackPlayer
from signals import SinusSignal, FileSignal
from tabs import SinusTab, FilesTab, SinusSettings



class IndicatorFilesWidget(QGroupBox):
    def __init__(self):
        super().__init__()
        self.setTitle('Показания из файла')
        self.fd1, self.fd2, self.fd3 = 0, 0, 0
        self.time = 0
        self.initUI()
    
    def initUI(self):
        self.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(self)
        
        self.time_lbl = QLabel()
        layout.addWidget(self.time_lbl)

        fd_layout = QHBoxLayout()
        self.fd1_lbl = QLabel()
        fd_layout.addWidget(self.fd1_lbl)

        self.fd2_lbl = QLabel()
        fd_layout.addWidget(self.fd2_lbl)

        self.fd3_lbl = QLabel()
        fd_layout.addWidget(self.fd3_lbl)

        layout.addLayout(fd_layout)
        self.update_values()

    def update_values(self):
        self.fd1_lbl.setText(f'Fd1: {self.fd1}')
        self.fd2_lbl.setText(f'Fd2: {self.fd2}')
        self.fd3_lbl.setText(f'Fd3: {self.fd3}')
        self.time_lbl.setText(f'Время {self.time}')

class IndicatorUdpWidget(QGroupBox):
    def __init__(self):
        super().__init__()
        self.setTitle('Показания по UDP')
        self.fd1, self.fd2, self.fd3 = 0, 0, 0
        self.tracking1, self.tracking2, self.tracking3 = False, False, False
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
        self.fd1_lbl = QLabel()
        fd1_layout.addWidget(self.fd1_lbl)
        self.tracking1_lbl = QLabel()
        fd1_layout.addWidget(self.tracking1_lbl)

        fd2_layout = QVBoxLayout()
        self.fd2_lbl = QLabel()
        fd2_layout.addWidget(self.fd2_lbl)
        self.tracking2_lbl = QLabel()
        fd2_layout.addWidget(self.tracking2_lbl)

        fd3_layout = QVBoxLayout()
        self.fd3_lbl = QLabel()
        fd3_layout.addWidget(self.fd3_lbl)
        self.tracking3_lbl = QLabel()
        fd3_layout.addWidget(self.tracking3_lbl)

        layout.addLayout(fd1_layout)
        layout.addLayout(fd2_layout)
        layout.addLayout(fd3_layout)
        self.update_values()

    def update_values(self):
        self.fd1_lbl.setText(f'Fd1: {self.fd1}')
        self.fd2_lbl.setText(f'Fd2: {self.fd2}')
        self.fd3_lbl.setText(f'Fd3: {self.fd3}')
        self.tracking1_lbl.setText(
            f'Tracking1: {"✓" if self.tracking1 else "✘"}')
        self.tracking2_lbl.setText(
            f'Tracking2: {"✓" if self.tracking2 else "✘"}')
        self.tracking3_lbl.setText(
            f'Tracking3: {"✓" if self.tracking3 else "✘"}')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.socket = QUdpSocket()
        self.received_packets = 0
        self.updater_information_timer = QTimer(self)
        self.updater_information_timer.timeout.connect(self.update_info)
        self.cache_data = b''
        self.current_channel = None
        self.player = None
        self.signal = None
        self.last_change_time = 0
        self.initUI()

    def initUI(self):
        main_widget = QTabWidget(self)
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

        self.signals_tab_widget = QTabWidget()


        self.sinus_tab = SinusTab(self)
        self.signals_tab_widget.addTab(self.sinus_tab, 'Генератор синуса')

        self.from_file_tab = FilesTab(self)
        self.signals_tab_widget.addTab(
            self.from_file_tab, 'Генератор из *.cap')

        main_layout.addWidget(self.signals_tab_widget)

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

        self.indicator_udp_widget = IndicatorUdpWidget()
        main_layout.addWidget(self.indicator_udp_widget)
        
        self.indicator_files_widget = IndicatorFilesWidget()
        main_layout.addWidget(self.indicator_files_widget)

        self.recieved_udp_packet_label = QLabel(
            f'Получено пакетов: {self.received_packets}')
        main_layout.addWidget(self.recieved_udp_packet_label)

        self.error_label = QLabel()
        main_layout.addWidget(self.error_label)
        
        self.signals_tab_widget.currentChanged.connect(self.show_files_indicator)
        self.show_files_indicator()

        self.setCentralWidget(main_widget)
        self.setWindowTitle('Генератор сигнала')
        self.setGeometry(100, 100, 700, 700)
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
            host_api_name = pa.get_host_api_info_by_index(
                int(device_host_api))['name']
            devices.append(f'{index}. {device_name} - {host_api_name}')
        self.device_cmb.addItems(devices)
        self.device_cmb.setCurrentIndex(
            int(pa.get_default_output_device_info()['index']))
        
    def show_files_indicator(self):
        if self.signals_tab_widget.currentIndex() == 0:
            self.indicator_files_widget.hide()
        else:
            self.indicator_files_widget.show()

    def update_sinus_signal(self, settings):
        if not isinstance(self.signal, SinusSignal):
            return
        if self.signals_tab_widget.currentIndex() == 0:
            self.signal.update_settings(
                freq1=settings.freq1,
                freq2=settings.freq2,
                freq3=settings.freq3,
                amp1=settings.amp1,
                amp2=settings.amp2,
                amp3=settings.amp3,
                phase1=settings.phase1,
                phase2=settings.phase2,
                phase3=settings.phase3
            )

    def get_sinus_signal(self):
        settings = self.sinus_tab.get_values_from_le()
        try:
            chunk_size = int(self.chunk_size_le.text())
            chunk_size = chunk_size if chunk_size in [
                32, 64, 128, 256, 512, 1024, 2048] else 64
        except:
            chunk_size = 64

        if settings:
            signal = SinusSignal(
                sampling_rate=settings.sampling_rate,
                chunk_size=chunk_size,
                freq1=settings.freq1,
                freq2=settings.freq2,
                freq3=settings.freq3,
                amp1=settings.amp1,
                amp2=settings.amp2,
                amp3=settings.amp3,
                phase1=settings.phase1,
                phase2=settings.phase2,
                phase3=settings.phase3
            )
            return signal

    def get_from_file_signal(self):
        sampling_rate = 44100
        chunk_size = int(self.chunk_size_le.text())
        if not chunk_size in [32, 64, 128, 256, 512, 1024, 2048]:
            raise ValueError(
                'Чанк должен быть: [32, 64, 128, 256, 512, 1024, 2048]')

        filenames = self.from_file_tab.get_filenames()
        if not filenames:
            raise ValueError('Не выбраны файлы')
        signal = FileSignal(
            chunk_size=chunk_size,
            sampling_rate=sampling_rate,
            filenames=self.from_file_tab.get_filenames(),
            cycle_play=self.from_file_tab.cycle_chkbx.isChecked()
        )
        return signal

    def start_process(self):
        self.error_label.setText('')
        self.received_packets = 0
        index = self.signals_tab_widget.currentIndex()
        if index == 0:
            self.signal = self.get_sinus_signal()

        elif index == 1:
            try:
                self.signal = self.get_from_file_signal()
            except Exception as e:
                self.error_label.setText(str(e))
                return
        else:
            return
        
        if self.signal is None:
            self.error_label.setText(
                'Невозможно создать сигнал по заданным параметрам')
            return

        self.player = CallbackPlayer(
            parent=self, 
            signal=self.signal,
            device_index=self.device_cmb.currentIndex()
        )

        self.player.run()

        if self.player.stream is None:
            self.error_label.setText('Ошибка аудиосутройства')
            return

        self.socket.bind(2015)
        self.socket.readyRead.connect(self.read_udp_data)

        self.updater_information_timer.start(500)

        self.send_btn.setText('Стоп')
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.stop_process)

        self.block_interface(True)

    def stop_process(self):
        self.send_btn.setText('Старт')
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.start_process)

        self.block_interface(False)

        if self.player:
            self.player.stop()
            self.player = None

        self.socket.close()
        self.updater_information_timer.stop()

    def block_interface(self, param):
        self.refresh_devices_btn.setDisabled(param)
        if self.signals_tab_widget.currentIndex() == 0:
            self.sinus_tab.block_line_edit(param)
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
                current_unch_channel = byte_data[i+6] >> 2
                if self.current_channel != current_unch_channel and current_unch_channel in [0, 1, 2]:
                    self.current_channel = current_unch_channel
                    return True
        return False

    def read_udp_data(self):
        while self.socket.hasPendingDatagrams():
            if not self.player.stream.is_active():
                self.stop_process()
                return
            
            self.received_packets += 1
            packet_data, * _ = self.socket.readDatagram(1056)
            packet_data = packet_data[:-32]

            if self.cache_data:
                packet_data = self.cache_data + packet_data

            start_byte = self.get_start_byte(packet_data)

            if start_byte is None:
                self.error_label.setText('Не нашёл начало данных')
                continue

            end = (len(packet_data) - start_byte) // 37 * 37 + start_byte
            current_data = packet_data[start_byte: end]

            self.cache_data = packet_data[end:]

            self.udpate_time_from_udp(current_data)

            if self.player:
                if not self.channel_changed(current_data):
                    continue

                switch_channel_timer = QTimer(self)
                switch_channel_timer.setSingleShot(True)
                switch_channel_timer.timeout.connect(
                    partial(self.player.switch_signal, self.current_channel))
                switch_channel_timer.start(
                    int(self.latency_le.text()) if self.latency_le.text() else 1)

                current_time = time.perf_counter_ns()
                print(
                    f'Канал изменён на {self.current_channel}, Время смены: {(current_time - self.last_change_time) / 1e6:2f} мс')
                self.last_change_time = current_time

    def udpate_time_from_udp(self, byte_data):
        for i in range(0, len(byte_data), 37):
            if byte_data[i+1] == 1:
                self.indicator_udp_widget.fd1 = int.from_bytes(
                    byte_data[i+6:i+8], signed=True)
                self.indicator_udp_widget.fd2 = int.from_bytes(
                    byte_data[i+8:i+10], signed=True)
                self.indicator_udp_widget.fd3 = int.from_bytes(
                    byte_data[i+10:i+12], signed=True)
                self.indicator_udp_widget.tracking1 = bool(
                    (byte_data[i+23] & 0b0100_0000) >> 6)
                self.indicator_udp_widget.tracking2 = bool(
                    (byte_data[i+23] & 0b1000_0000) >> 7)
                self.indicator_udp_widget.tracking3 = bool(byte_data[i+22] & 0b1)
        if isinstance(self.signal, FileSignal):
            (fd1, fd2, fd3), time_from_sig = self.signal.get_fd_and_times()
            self.indicator_files_widget.fd1 = fd1
            self.indicator_files_widget.fd2 = fd2
            self.indicator_files_widget.fd3 = fd3
            self.indicator_files_widget.time = time_from_sig

    def update_info(self):
        self.recieved_udp_packet_label.setText(
            f'Получено пакетов: {self.received_packets}')
        self.indicator_udp_widget.update_values()
        if isinstance(self.signal, FileSignal):
            self.indicator_files_widget.update_values()
        
    def set_to_noise_mode(self, param):
        if isinstance(self.signal, SinusSignal):
            self.signal.set_to_noise(param)

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
