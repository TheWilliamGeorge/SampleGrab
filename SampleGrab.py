"""
SampleGrab - Simple, Universal Instrument Sampler

Features:
- Toggle keys (click to start, click again to stop).
- Realistic black & white piano key layout (C..B across octaves).
- Shared audio input engine allowing multiple simultaneous recordings.
- Per-key level meter while recording (smoothed peak).
- Trim start using threshold and configurable pre-roll (ms).
- Project manager: select/create projects from dropdown ("New Project…" entry), recordings saved into project subfolders under `recordings/`.
- Per-project file browser with preview (double-click), delete, rename, duplicate projects, export current project to ZIP.
- Automatically loads last used project on startup (stored in config.json).

Dependencies:
 pip install PySide6 sounddevice soundfile numpy

Run:
 python pyside6_keyboard_recorder.py

"""

import sys
import os
import queue
import threading
import datetime
import zipfile
import json
import shutil
from typing import Optional

from PySide6 import QtCore, QtWidgets, QtGui
import sounddevice as sd
import soundfile as sf
import numpy as np

import struct
import wave
import re

# --- Config / paths ---
DEFAULT_SR = 44100
DEFAULT_CHANNELS = 1
BASE_RECORDINGS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "recordings")
os.makedirs(BASE_RECORDINGS_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "recorder_config.json")

# --- Key sizes ---
WHITE_KEY_W = 60
WHITE_KEY_H = 220
BLACK_KEY_W = int(WHITE_KEY_W * 0.62)
BLACK_KEY_H = int(WHITE_KEY_H * 0.60)

NOTE_OFFSETS = {
    'C': 0, 'C#': 1, 'DB': 1,
    'D': 2, 'D#': 3, 'EB': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'GB': 6,
    'G': 7, 'G#': 8, 'AB': 8,
    'A': 9, 'A#': 10, 'BB': 10,
    'B': 11
}

def parse_note_token(token: str):
    m = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', token.strip())
    if not m:
        return None
    note = m.group(1).upper()
    acc = m.group(2)
    octave = int(m.group(3))
    if acc == 'b':
        note = note + 'B'
    elif acc == '#':
        note = note + '#'
    return note, octave

def note_to_midi(note: str, octave: int, mapping: str = 'c4') -> int:
    note_key = note.upper()
    if note_key not in NOTE_OFFSETS:
        raise ValueError(f"Unknown note name: {note}")
    offset = NOTE_OFFSETS[note_key]
    if mapping == 'c4':
        midi = (octave - 4) * 12 + 60 + offset
    elif mapping == 'c3':
        midi = (octave - 3) * 12 + 60 + offset
    else:
        raise ValueError("mapping must be 'c4' or 'c3'")
    if not (0 <= midi <= 127):
        raise ValueError(f"MIDI note out of range: {midi}")
    return midi

def build_smpl_chunk(sample_rate: int, midi_unity_note: int) -> bytes:
    manufacturer = 0
    product = 0
    sample_period = max(1, int(1_000_000_000 / sample_rate))
    midi_pitch_fraction = 0
    smpte_format = 0
    smpte_offset = 0
    num_sample_loops = 0
    sampler_data = 0
    payload = struct.pack(
        '<9I',
        manufacturer,
        product,
        sample_period,
        midi_unity_note,
        midi_pitch_fraction,
        smpte_format,
        smpte_offset,
        num_sample_loops,
        sampler_data
    )
    chunk = b'smpl' + struct.pack('<I', len(payload)) + payload
    if len(payload) % 2 == 1:
        chunk += b'\x00'
    return chunk

def read_riff_chunks(f):
    chunks = []
    while True:
        header = f.read(8)
        if len(header) < 8:
            break
        cid, csize = struct.unpack('<4sI', header)
        data = f.read(csize)
        if csize % 2 == 1:
            _ = f.read(1)
        chunks.append((cid, data))
    return chunks

def write_riff_file(out_path: str, chunks):
    total = 4 + sum(8 + len(d) + (len(d) % 2) for (_, d) in chunks)
    with open(out_path, 'wb') as fo:
        fo.write(b'RIFF' + struct.pack('<I', total) + b'WAVE')
        for cid, data in chunks:
            fo.write(cid)
            fo.write(struct.pack('<I', len(data)))
            fo.write(data)
            if len(data) % 2 == 1:
                fo.write(b'\x00')

def inject_smpl(in_filepath: str, midi_note: int, backup: bool = True):
    with open(in_filepath, 'rb') as fi:
        header = fi.read(12)
        riff, riff_size, wave_id = struct.unpack('<4sI4s', header)
        if riff != b'RIFF' or wave_id != b'WAVE':
            raise IOError("Not a valid WAV file")
        chunks = read_riff_chunks(fi)

    fmt_chunk = next((d for (c, d) in chunks if c == b'fmt '), None)
    if fmt_chunk and len(fmt_chunk) >= 12:
        sample_rate = struct.unpack_from('<I', fmt_chunk, 4)[0]
    else:
        with wave.open(in_filepath, 'rb') as wf:
            sample_rate = wf.getframerate()

    new_chunks = [(c, d) for (c, d) in chunks if c != b'smpl']
    smpl_chunk = build_smpl_chunk(sample_rate, midi_note)
    cid = smpl_chunk[0:4]
    size = struct.unpack('<I', smpl_chunk[4:8])[0]
    payload = smpl_chunk[8:8 + size]
    new_chunks.append((cid, payload))

    if backup:
        # Determine project name from recordings folder structure
        recordings_dir = os.path.dirname(in_filepath)
        project_name = os.path.basename(recordings_dir)
        backup_dir = os.path.join("backups", project_name)
        os.makedirs(backup_dir, exist_ok=True)

        bak_path = os.path.join(backup_dir, os.path.basename(in_filepath) + ".bak")
        if not os.path.exists(bak_path):
            shutil.copy2(in_filepath, bak_path)

    tmp_out = in_filepath + '.tmp'
    write_riff_file(tmp_out, new_chunks)
    os.replace(tmp_out, in_filepath)


# --- Audio Engine (shared InputStream) ---
class AudioEngine(QtCore.QObject):
    level_updated = QtCore.Signal(float)

    def __init__(self, device: Optional[int]=None, samplerate:int=DEFAULT_SR, channels:int=DEFAULT_CHANNELS, dtype='float32'):
        super().__init__()
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._stream = None
        self._q = queue.Queue(maxsize=300)
        self._running = False
        self._lock = threading.Lock()
        self.buffers = {}  # note -> list of chunks
        self._worker = None

    def start(self):
        with self._lock:
            if self._running:
                return
            try:
                self._stream = sd.InputStream(device=self.device, channels=self.channels,
                                              samplerate=self.samplerate, callback=self._callback,
                                              dtype=self.dtype)
                self._stream.start()
            except Exception as e:
                raise
            self._running = True
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def stop(self):
        with self._lock:
            if not self._running:
                return
            self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio input status:", status)
        try:
            self._q.put(indata.copy(), block=False)
        except queue.Full:
            pass

    def _worker_loop(self):
        while self._running:
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                for note, buf in list(self.buffers.items()):
                    if buf is not None:
                        buf.append(chunk)
            try:
                peak = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0
            except Exception:
                peak = 0.0
            try:
                self.level_updated.emit(peak)
            except Exception:
                pass

    def start_recording(self, note: str):
        with self._lock:
            if note in self.buffers and self.buffers[note] is not None:
                return
            self.buffers[note] = []

    def stop_recording(self, note: str):
        with self._lock:
            buf = self.buffers.pop(note, None)
        if not buf:
            return np.zeros((0, self.channels), dtype=self.dtype)
        try:
            data = np.concatenate(buf, axis=0)
        except Exception:
            data = np.vstack(buf) if len(buf) > 0 else np.zeros((0, self.channels), dtype=self.dtype)
        return data

    def active_recordings(self):
        with self._lock:
            return list(self.buffers.keys())

# --- Key widget ---
class KeyWidget(QtWidgets.QWidget):
    toggled = QtCore.Signal(str, bool)

    def __init__(self, note: str, is_black: bool = False, parent=None):
        super().__init__(parent)
        self.note = note
        self.is_black = is_black

        self.button = QtWidgets.QPushButton(note)
        self.button.setCheckable(True)
        self.button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.meter = QtWidgets.QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(8)
        self.meter.hide()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self.button)
        layout.addWidget(self.meter)

        if self.is_black:
            self.button.setFixedSize(BLACK_KEY_W, BLACK_KEY_H)
            self.setFixedSize(BLACK_KEY_W + 4, BLACK_KEY_H + 24)
            self.button.setStyleSheet(
                "QPushButton { background: #111; color: white; border: 1px solid #333; }"
                "QPushButton:checked { background: #2a2aff; }"
            )
        else:
            self.button.setFixedSize(WHITE_KEY_W, WHITE_KEY_H)
            self.setFixedSize(WHITE_KEY_W + 4, WHITE_KEY_H + 24)
            self.button.setStyleSheet(
                "QPushButton { background: white; color: black; border: 1px solid #222; }"
                "QPushButton:checked { background: #88baff; }"
            )

        self._smoothed = 0.0
        self._alpha = 0.6

        self.button.toggled.connect(self._on_toggled)

    def _on_toggled(self, on: bool):
        if on:
            self.meter.show()
            self.meter.setValue(0)
        else:
            self.meter.hide()
            self.meter.setValue(0)
            self._smoothed = 0.0
        self.toggled.emit(self.note, on)

    def set_level(self, level_float: float):
        if not self.button.isChecked():
            return
        level = max(0.0, min(1.0, level_float))
        self._smoothed = (self._alpha * self._smoothed) + ((1.0 - self._alpha) * level)
        v = int(self._smoothed * 100)
        self.meter.setValue(v)

    def force_reset(self):
        self.button.setChecked(False)
        self.meter.hide()
        self.meter.setValue(0)
        self._smoothed = 0.0

# --- Main application window ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SampleGrab")
        self.resize(1200, 620)

        self.engine: Optional[AudioEngine] = None
        self.current_project: Optional[str] = None
        self.saved_files = []
        self.playing_stream = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Top controls: project dropdown (with New Project…), rename, duplicate; device and audio settings; threshold/pre-roll; export
        top = QtWidgets.QHBoxLayout()

        top.addWidget(QtWidgets.QLabel("Project:"))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.currentIndexChanged.connect(self.on_project_changed)
        top.addWidget(self.project_combo)

        '''
        # rename / duplicate
        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.clicked.connect(self.rename_project)
        top.addWidget(self.rename_btn)
        self.duplicate_btn = QtWidgets.QPushButton("Duplicate")
        self.duplicate_btn.clicked.connect(self.duplicate_project)
        top.addWidget(self.duplicate_btn)
        '''

        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Input device:"))
        self.device_combo = QtWidgets.QComboBox()
        top.addWidget(self.device_combo)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.populate_devices)
        top.addWidget(self.refresh_btn)

        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("SR:"))
        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setValue(DEFAULT_SR)
        top.addWidget(self.sr_spin)

        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(3)
        self.threshold_slider.setFixedWidth(120)
        top.addWidget(self.threshold_slider)
        self.threshold_label = QtWidgets.QLabel("0.03")
        top.addWidget(self.threshold_label)

        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Pre-roll (ms):"))
        self.pr_spin = QtWidgets.QSpinBox()
        self.pr_spin.setRange(0, 1000)
        self.pr_spin.setValue(100)
        top.addWidget(self.pr_spin)

        top.addStretch()
        self.export_btn = QtWidgets.QPushButton("Export Project (ZIP)")
        self.export_btn.clicked.connect(self.export_project)
        top.addWidget(self.export_btn)

        vbox.addLayout(top)

        self.threshold_slider.valueChanged.connect(self._update_threshold_label)

        # Main area: left = keyboard, right = project file browser
        main_h = QtWidgets.QHBoxLayout()

        # Keyboard scroll area
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        main_h.addWidget(self.scroll, 3)

        self.kb_container = QtWidgets.QWidget()
        self.scroll.setWidget(self.kb_container)
        self.kb_container.setMinimumHeight(WHITE_KEY_H + 40)
        self.kb_container.setLayout(None)  # absolute placement

        self.key_widgets = {}
        self.build_keyboard(start_octave=2, end_octave=5)

        # Right side: file list and controls
        right_v = QtWidgets.QVBoxLayout()
        right_v.addWidget(QtWidgets.QLabel("Project files: "))
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemDoubleClicked.connect(self.play_file)
        right_v.addWidget(self.file_list, 1)

        btns_h = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.play_selected)
        btns_h.addWidget(self.play_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        btns_h.addWidget(self.stop_btn)
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)
        btns_h.addWidget(self.delete_btn)
        right_v.addLayout(btns_h)

        v2 = QtWidgets.QHBoxLayout()
        self.rename_proj_btn = QtWidgets.QPushButton("Rename Project")
        self.rename_proj_btn.clicked.connect(self.rename_project)
        v2.addWidget(self.rename_proj_btn)

        self.duplicate_proj_btn = QtWidgets.QPushButton("Duplicate Project")
        self.duplicate_proj_btn.clicked.connect(self.duplicate_project)
        v2.addWidget(self.duplicate_proj_btn)

        right_v.addLayout(v2)

        v3 = QtWidgets.QHBoxLayout()
        self.assign_root_btn = QtWidgets.QPushButton("Assign Root Note Metadata to All Files")
        self.assign_root_btn.clicked.connect(self.assign_root_notes)
        v3.addWidget(self.assign_root_btn)

        right_v.addLayout(v3)  # Add it below the previous row

        main_h.addLayout(right_v, 1)
        vbox.addLayout(main_h)

        # Status bar
        self.status = QtWidgets.QLabel("")
        vbox.addWidget(self.status)

        # init devices/projects
        self.populate_devices()
        self.load_projects()
        self.restore_last_project()

    # --- projects and config ---
    def projects_dir(self) -> str:
        return BASE_RECORDINGS_DIR

    def load_projects(self):
        self.project_combo.clear()
        projs = [d for d in os.listdir(self.projects_dir()) if os.path.isdir(os.path.join(self.projects_dir(), d))]
        projs.sort()
        for p in projs:
            self.project_combo.addItem(p)
        self.project_combo.addItem("New Project…")

    def restore_last_project(self):
        last = self._read_config().get('last_project')
        if last and last in [self.project_combo.itemText(i) for i in range(self.project_combo.count())]:
            self.project_combo.setCurrentText(last)
        elif self.project_combo.count() > 1:
            self.project_combo.setCurrentIndex(0)

    def on_project_changed(self):
        txt = self.project_combo.currentText()
        if txt == "New Project…":
            # create new project
            name, ok = QtWidgets.QInputDialog.getText(self, "New project", "Enter new project name:")
            if ok and name.strip():
                safe = ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
                if safe:
                    proj_path = os.path.join(self.projects_dir(), safe)
                    if not os.path.exists(proj_path):
                        os.makedirs(proj_path, exist_ok=True)
                        self.load_projects()
                        idx = self.project_combo.findText(safe)
                        if idx >= 0:
                            self.project_combo.setCurrentIndex(idx)
                        self._write_config({'last_project': safe})
                        self.current_project = safe
                        self.refresh_file_list()
                        return
            # revert selection if canceled
            if self.project_combo.count() > 1:
                self.project_combo.setCurrentIndex(0)
            return
        self.current_project = txt
        self._write_config({'last_project': txt})
        self.refresh_file_list()
        self.status.setText(f"Project: {txt}")

    def rename_project(self):
        if not self.current_project:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename project", "Enter new name:", text=self.current_project)
        if not ok:
            return
        new = ''.join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not new:
            QtWidgets.QMessageBox.warning(self, "Invalid name", "Project name invalid.")
            return
        old_path = os.path.join(self.projects_dir(), self.current_project)
        new_path = os.path.join(self.projects_dir(), new)
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename failed", f"Could not rename project: {e}")
            return
        self.load_projects()
        idx = self.project_combo.findText(new)
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)
        self.current_project = new
        self._write_config({'last_project': new})

    def duplicate_project(self):
        if not self.current_project:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Duplicate project", "Enter name for duplicate:", text=self.current_project + "_copy")
        if not ok:
            return
        new = ''.join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not new:
            QtWidgets.QMessageBox.warning(self, "Invalid name", "Project name invalid.")
            return
        src = os.path.join(self.projects_dir(), self.current_project)
        dst = os.path.join(self.projects_dir(), new)
        try:
            shutil.copytree(src, dst)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Duplicate failed", f"Could not duplicate: {e}")
            return
        self.load_projects()
        idx = self.project_combo.findText(new)
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)
        self.current_project = new
        self._write_config({'last_project': new})

    def refresh_file_list(self):
        self.file_list.clear()
        if not self.current_project:
            return
        proj_path = os.path.join(self.projects_dir(), self.current_project)
        try:
            files = [f for f in os.listdir(proj_path) if os.path.isfile(os.path.join(proj_path, f))]
            files.sort()
            for f in files:
                self.file_list.addItem(f)
        except Exception:
            pass

    # --- devices ---
    def populate_devices(self):
        self.device_combo.clear()
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []
        for i, d in enumerate(devs):
            if d.get('max_input_channels', 0) > 0:
                self.device_combo.addItem(f"{i}: {d['name']}", i)
        try:
            default = sd.default.device
            inp = default[0] if isinstance(default, (list, tuple)) else default
            for j in range(self.device_combo.count()):
                if self.device_combo.itemData(j) == inp:
                    self.device_combo.setCurrentIndex(j)
                    break
        except Exception:
            pass

    # --- keyboard construction ---
    def build_keyboard(self, start_octave=2, end_octave=5):
        white_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        black_map = {0: 'C#', 1: 'D#', 3: 'F#', 4: 'G#', 5: 'A#'}
        octaves = list(range(start_octave, end_octave + 1))
        white_count = len(octaves) * len(white_notes)
        total_w = white_count * WHITE_KEY_W
        self.kb_container.setMinimumWidth(total_w + 40)

        x = 10
        for octv in octaves:
            for n in white_notes:
                note = f"{n}{octv}"
                kw = KeyWidget(note, is_black=False, parent=self.kb_container)
                kw.move(x, 10)
                kw.toggled.connect(self.on_key_toggled)
                kw.show()
                self.key_widgets[note] = kw
                x += WHITE_KEY_W

        x = 10
        for oi, octv in enumerate(octaves):
            base_x = x + oi * 7 * WHITE_KEY_W
            for wi in range(7):
                if wi in black_map:
                    black_note = black_map[wi]
                    note = f"{black_note}{octv}"
                    bx = base_x + (wi + 1) * WHITE_KEY_W - (BLACK_KEY_W // 2)
                    by = 10
                    kw = KeyWidget(note, is_black=True, parent=self.kb_container)
                    kw.move(bx, by)
                    kw.toggled.connect(self.on_key_toggled)
                    kw.raise_()
                    kw.show()
                    self.key_widgets[note] = kw

    # --- engine and recording control ---
    def ensure_engine(self):
        device_index = self.device_combo.currentData()
        sr = int(self.sr_spin.value())
        if self.engine is None:
            self.engine = AudioEngine(device=device_index, samplerate=sr, channels=DEFAULT_CHANNELS)
            self.engine.level_updated.connect(self.on_level_updated)
        else:
            active = self.engine.active_recordings()
            if len(active) == 0 and (self.engine.device != device_index or self.engine.samplerate != sr):
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self.engine = AudioEngine(device=device_index, samplerate=sr, channels=DEFAULT_CHANNELS)
                self.engine.level_updated.connect(self.on_level_updated)

    def on_key_toggled(self, note: str, on: bool):
        if on:
            if not self.current_project:
                QtWidgets.QMessageBox.warning(self, "No project", "Please create or select a project before recording.")
                kw = self.key_widgets.get(note)
                if kw:
                    kw.force_reset()
                return
            self.ensure_engine()
            if self.engine is None:
                QtWidgets.QMessageBox.critical(self, "Audio error", "Could not initialize audio engine.")
                kw = self.key_widgets.get(note)
                if kw:
                    kw.force_reset()
                return
            try:
                if not self.engine._running:
                    self.engine.start()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Audio error", f"Could not start audio input: {e}")
                kw = self.key_widgets.get(note)
                if kw:
                    kw.force_reset()
                return
            self.engine.start_recording(note)
            self._set_device_controls_enabled(False)
            self.status.setText(f"Recording: {note}")
        else:
            if not self.engine:
                return
            data = self.engine.stop_recording(note)
            if len(self.engine.active_recordings()) == 0:
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self._set_device_controls_enabled(True)
            self._process_and_save(note, data)
            self.status.setText(f"Saved: {note}")

    def _set_device_controls_enabled(self, enabled: bool):
        self.device_combo.setEnabled(enabled)
        self.sr_spin.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
        self.project_combo.setEnabled(enabled)

    def on_level_updated(self, peak: float):
        for note, kw in self.key_widgets.items():
            if kw.button.isChecked():
                kw.set_level(peak)

    def _process_and_save(self, note: str, raw: np.ndarray):
        if raw is None or raw.size == 0:
            QtWidgets.QMessageBox.information(self, "No audio", f"No audio recorded for {note}.")
            return
        threshold = self.threshold_slider.value() / 100.0
        pre_ms = int(self.pr_spin.value())
        sr = int(self.sr_spin.value())
        pre_samples = int(pre_ms * sr / 1000.0)

        if raw.ndim == 1:
            amplitudes = np.abs(raw)
        else:
            amplitudes = np.max(np.abs(raw), axis=1)
        idxs = np.where(amplitudes > threshold)[0]
        if idxs.size > 0:
            first = max(0, idxs[0] - pre_samples)
            trimmed = raw[first:]
        else:
            trimmed = raw

        proj_folder = os.path.join(self.projects_dir(), self.current_project)
        os.makedirs(proj_folder, exist_ok=True)
        outname = self._make_filename(note, proj_folder)
        try:
            sf.write(outname, trimmed, sr, subtype='PCM_16')
            self.saved_files.append(outname)
            self.status.setText(f"Saved {os.path.basename(outname)}")
            self.refresh_file_list()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not save file: {e}")

    def _make_filename(self, note: str, folder: str) -> str:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{note}_{ts}.wav"
        return os.path.join(folder, fname)

    def export_project(self):
        if not self.current_project:
            QtWidgets.QMessageBox.warning(self, "No project", "Please create or select a project first.")
            return
        project_path = os.path.join(self.projects_dir(), self.current_project)
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ZIP", f"{self.current_project}.zip", "Zip Files (*.zip)")
        if save_path:
            try:
                with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(project_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, project_path)
                            zf.write(file_path, arcname)
                QtWidgets.QMessageBox.information(self, "Exported", f"Exported project to {save_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not create zip: {e}")

    def _update_threshold_label(self, v: int):
        self.threshold_label.setText(f"{v/100.0:.2f}")

    # --- project file browser actions ---
    def play_file(self, item: QtWidgets.QListWidgetItem):
        filepath = os.path.join(self.projects_dir(), self.current_project, item.text())
        try:
            data, sr = sf.read(filepath, dtype='float32')
            sd.play(data, sr)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Play failed", f"Could not play file: {e}")

    def play_selected(self):
        item = self.file_list.currentItem()
        if item:
            self.play_file(item)

    def stop_playback(self):
        try:
            sd.stop()
        except Exception:
            pass

    def delete_selected(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        confirm = QtWidgets.QMessageBox.question(self, "Delete", f"Delete {len(items)} files?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        for item in items:
            filepath = os.path.join(self.projects_dir(), self.current_project, item.text())
            try:
                os.remove(filepath)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Delete failed", f"Could not delete {item.text()}: {e}")
        self.refresh_file_list()

    # --- config read/write ---
    def _read_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_config(self, data: dict):
        cfg = self._read_config()
        cfg.update(data)
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f)
        except Exception:
            pass

    def assign_root_notes(self):
        if not self.current_project:
            QtWidgets.QMessageBox.warning(self, "No project", "Please select a project first.")
            return
        proj_path = os.path.join(self.projects_dir(), self.current_project)
        files = sorted(f for f in os.listdir(proj_path) if f.lower().endswith('.wav'))
        if not files:
            QtWidgets.QMessageBox.information(self, "No files", "No WAV files found in this project.")
            return

        skipped = []
        updated_count = 0
        for fn in files:
            token = fn.split('_', 1)[0]
            parsed = parse_note_token(token)
            if not parsed:
                skipped.append(fn)
                continue
            note_name, octave = parsed
            try:
                midi = note_to_midi(note_name, octave, mapping='c4')
            except Exception:
                skipped.append(fn)
                continue
            try:
                inject_smpl(os.path.join(proj_path, fn), midi, backup=True)
                updated_count += 1
            except Exception as e:
                skipped.append(f"{fn} ({e})")

        msg = f"Updated {updated_count} files with root note metadata."
        if skipped:
            msg += f"\nSkipped {len(skipped)} files:\n" + "\n".join(skipped)
        QtWidgets.QMessageBox.information(self, "Assign Root Notes", msg)
        self.status.setText(f"Root note assignment done for {updated_count} files.")


# --- run ---
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
