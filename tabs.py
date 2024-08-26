from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget,
                             QPushButton, QVBoxLayout, QWidget, QCheckBox)
from PyQt6.QtCore import Qt
import os
from collections import namedtuple

SinusSettings = namedtuple(
    'Sinus_settings', [
        'sampling_rate',
        'freq1', 'freq2', 'freq3',
        'amp1', 'amp2', 'amp3',
        'phase1', 'phase2', 'phase3'
    ]
)

class SinusTab(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        sampling_rate_layout = QHBoxLayout()

        sampling_rate_label = QLabel('Частота дискретизации')
        sampling_rate_label.setFixedWidth(155)
        sampling_rate_layout.addWidget(sampling_rate_label)

        self.sampling_rate_le = QLineEdit(f'44100')
        self.sampling_rate_le.setPlaceholderText('Частота дискретизации:')
        self.sampling_rate_le.textChanged.connect(self.text_changed_handler)
        sampling_rate_layout.addWidget(self.sampling_rate_le)

        layout.addLayout(sampling_rate_layout)

        freq_layout = QHBoxLayout()

        freq_label = QLabel('Частота (от 0 до 20000 Гц):')
        freq_label.setFixedWidth(155)
        freq_layout.addWidget(freq_label)

        self.freq1_le = QLineEdit('20')
        self.freq1_le.setPlaceholderText('Частота унч1')
        self.freq1_le.textChanged.connect(self.text_changed_handler)
        freq_layout.addWidget(self.freq1_le)

        self.freq2_le = QLineEdit('200')
        self.freq2_le.setPlaceholderText('Частота унч2')
        self.freq2_le.textChanged.connect(self.text_changed_handler)
        freq_layout.addWidget(self.freq2_le)

        self.freq3_le = QLineEdit('1000')
        self.freq3_le.setPlaceholderText('Частота унч3')
        self.freq3_le.textChanged.connect(self.text_changed_handler)
        freq_layout.addWidget(self.freq3_le)

        layout.addLayout(freq_layout)

        amp_layout = QHBoxLayout()

        amp_label = QLabel('Амплитуда (от 0 до 1):')
        amp_label.setFixedWidth(155)
        amp_layout.addWidget(amp_label)

        self.amp1_le = QLineEdit('1')
        self.amp1_le.setPlaceholderText('Амплитуда унч1')
        self.amp1_le.textChanged.connect(self.text_changed_handler)
        amp_layout.addWidget(self.amp1_le)

        self.amp2_le = QLineEdit('1')
        self.amp2_le.setPlaceholderText('Амплитуда унч2')
        self.amp2_le.textChanged.connect(self.text_changed_handler)
        amp_layout.addWidget(self.amp2_le)

        self.amp3_le = QLineEdit('1')
        self.amp3_le.setPlaceholderText('Амплитуда унч3')
        self.amp3_le.textChanged.connect(self.text_changed_handler)
        amp_layout.addWidget(self.amp3_le)

        layout.addLayout(amp_layout)

        ph_layout = QHBoxLayout()

        ph1_label = QLabel('Фаза 2 канала (от 0° до 360°):')
        ph1_label.setFixedWidth(155)
        ph_layout.addWidget(ph1_label)

        self.ph1_le = QLineEdit('90')
        self.ph1_le.setPlaceholderText('Фаза 2 канала унч1')
        self.ph1_le.textChanged.connect(self.text_changed_handler)
        ph_layout.addWidget(self.ph1_le)

        self.ph2_le = QLineEdit('90')
        self.ph2_le.setPlaceholderText('Фаза 2 канала унч2')
        self.ph2_le.textChanged.connect(self.text_changed_handler)
        ph_layout.addWidget(self.ph2_le)

        self.ph3_le = QLineEdit('90')
        self.ph3_le.setPlaceholderText('Фаза 2 канала унч3')
        self.ph3_le.textChanged.connect(self.text_changed_handler)
        ph_layout.addWidget(self.ph3_le)
        layout.addLayout(ph_layout)

        self.noise_chkbx = QCheckBox('Шум активен')
        self.noise_chkbx.checkStateChanged.connect(self.set_to_noise_mode)
        self.noise_chkbx.setDisabled(True)

        layout.addWidget(self.noise_chkbx)

        self.setLayout(layout)

    def text_changed_handler(self):
        settings = self.get_values_from_le()
        if settings:
            self.parent.update_sinus_signal(settings)

    def block_line_edit(self, param):
        self.sampling_rate_le.setDisabled(param)
        self.noise_chkbx.setDisabled(not param)
        if not param:
            self.noise_chkbx.setChecked(param)

    def get_values_from_le(self):
        try:
            values = SinusSettings(
                int(self.sampling_rate_le.text()),
                int(self.freq1_le.text()),
                int(self.freq2_le.text()),
                int(self.freq3_le.text()),
                float(self.amp1_le.text()),
                float(self.amp2_le.text()),
                float(self.amp3_le.text()),
                int(self.ph1_le.text()),
                int(self.ph2_le.text()),
                int(self.ph3_le.text()),
            )
        except:
            return
        return values
    
    def set_to_noise_mode(self):
        self.parent.set_to_noise_mode(self.noise_chkbx.isChecked())


class FilesTab(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self.button = QPushButton("Выбрать папку", self)
        self.file_list = QListWidget(self)

        self.cycle_chkbx = QCheckBox('Бесконечное воспроизведение')

        layout.addWidget(self.button)
        layout.addWidget(self.file_list)
        layout.addWidget(self.cycle_chkbx)

        self.setLayout(layout)

        self.button.clicked.connect(self.select_folder)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку")

        if folder_path:
            self.file_list.clear()

            files = os.listdir(folder_path)
            for file_name in files:
                full_path = os.path.join(folder_path, file_name)
                normalized_path = os.path.normpath(full_path)
                if os.path.isfile(normalized_path) and file_name[-4:] in {".cap", "apng"}:
                    self.file_list.addItem(normalized_path)

    def get_filenames(self):
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]
