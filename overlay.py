import argparse
import contextlib
import os
import sys
import threading
from datetime import datetime
import time
from pathlib import Path

import deepl
import numpy as np
import pyaudiowpatch as pa
from deepgram import DeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 512
INTERIM_RESULTS = True
TARGET_LANG = "ZH-HANS"
MAX_RUN_MINUTES = 30


def downmix_and_resample_linear16(
    in_bytes: bytes, in_channels: int, in_rate: int, out_rate: int
) -> bytes:
    audio = np.frombuffer(in_bytes, dtype=np.int16)
    if audio.size == 0:
        return b""
    if in_channels > 1:
        frames = audio.size // in_channels
        audio = audio[: frames * in_channels].reshape(frames, in_channels).mean(axis=1)
    audio = audio.astype(np.float32)
    if in_rate != out_rate and audio.size > 1:
        src_len = audio.shape[0]
        dst_len = max(1, int(src_len * out_rate / in_rate))
        src_x = np.arange(src_len, dtype=np.float32)
        dst_x = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
    return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()


def resolve_default_loopback_device(p: pa.PyAudio) -> dict:
    wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    for loopback in p.get_loopback_device_info_generator():
        if default_out["name"] in loopback["name"]:
            return loopback
    return p.get_default_wasapi_loopback()


class StreamWorker(QThread):
    caption_update = pyqtSignal(str, str, str)  # interim_en, final_en, final_zh
    status_update = pyqtSignal(str)
    error_update = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        try:
            self._run_sync()
        except Exception as exc:
            self.error_update.emit(str(exc))

    def _run_sync(self) -> None:
        if load_dotenv is not None:
            load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        deepl_api_key = os.getenv("DEEPL_API_KEY")
        if not deepgram_api_key:
            raise RuntimeError("Missing DEEPGRAM_API_KEY")
        if not deepl_api_key:
            raise RuntimeError("Missing DEEPL_API_KEY")

        translator = deepl.Translator(deepl_api_key)
        deepgram = DeepgramClient(api_key=deepgram_api_key)

        p = pa.PyAudio()
        stream = None
        receiver_thread = None
        try:
            device = resolve_default_loopback_device(p)
            in_channels = int(device["maxInputChannels"])
            in_rate = int(device["defaultSampleRate"])
            self.status_update.emit(
                f"Running on {device['name']} ({in_channels}ch @ {in_rate} Hz)"
            )

            with deepgram.listen.v1.connect(
                model="nova-2",
                language="en-US",
                smart_format="true",
                encoding="linear16",
                channels="1",
                sample_rate=str(TARGET_SAMPLE_RATE),
                interim_results="true" if INTERIM_RESULTS else "false",
            ) as dg_connection:
                stream = p.open(
                    format=pa.paInt16,
                    channels=in_channels,
                    rate=in_rate,
                    input=True,
                    input_device_index=int(device["index"]),
                    frames_per_buffer=CHUNK_SIZE,
                )
                stream.start_stream()

                last_interim = ""

                def receiver() -> None:
                    nonlocal last_interim
                    for message in dg_connection:
                        if self._stop_flag.is_set():
                            break
                        if not isinstance(message, ListenV1Results):
                            continue
                        if not message.channel.alternatives:
                            continue
                        text = (message.channel.alternatives[0].transcript or "").strip()
                        if not text:
                            continue
                        if message.is_final:
                            try:
                                zh = translator.translate_text(text, target_lang=TARGET_LANG).text
                            except Exception as exc:
                                zh = f"[DeepL error: {exc}]"
                            self.caption_update.emit("", text, zh)
                            last_interim = ""
                        elif text != last_interim:
                            self.caption_update.emit(text, "", "")
                            last_interim = text

                receiver_thread = threading.Thread(target=receiver, daemon=True)
                receiver_thread.start()

                while not self._stop_flag.is_set():
                    available = 0
                    with contextlib.suppress(Exception):
                        available = int(stream.get_read_available())
                    if available < CHUNK_SIZE:
                        time.sleep(0.02)
                        continue
                    try:
                        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    except TypeError:
                        raw = stream.read(CHUNK_SIZE)
                    except Exception:
                        if self._stop_flag.is_set():
                            break
                        continue
                    payload = downmix_and_resample_linear16(
                        raw, in_channels=in_channels, in_rate=in_rate, out_rate=TARGET_SAMPLE_RATE
                    )
                    if payload:
                        with contextlib.suppress(Exception):
                            dg_connection.send_media(payload)
                if receiver_thread is not None:
                    receiver_thread.join(timeout=1.0)
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.stop_stream()
                    stream.close()
            p.terminate()
            self.status_update.emit("Stopped")


class CaptionOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.worker: StreamWorker | None = None
        self._last_en_final = ""
        self._last_zh_final = ""
        self._last_en_interim = ""
        self.auto_stop_timer = QTimer(self)
        self.auto_stop_timer.setSingleShot(True)
        self.auto_stop_timer.timeout.connect(self.on_auto_stop_timeout)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown_label)
        self.session_end_monotonic: float | None = None
        self.auto_stopped = False
        self._build_ui()
        self._position_bottom_center()

    def _build_ui(self) -> None:
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Live Caption Overlay")
        self.setWindowOpacity(0.92)
        self.setStyleSheet(
            "QWidget { background-color: #141414; color: #EDEDED; }"
            "QPushButton { background-color: #262626; color: #F5F5F5; border: 1px solid #3A3A3A; "
            "border-radius: 8px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #303030; }"
            "QPushButton:disabled { background-color: #1E1E1E; color: #888888; border-color: #2A2A2A; }"
        )

        root = QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_streaming)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_labels)
        self.always_on_top_checkbox = QCheckBox("Always on top")
        self.always_on_top_checkbox.setChecked(True)
        self.always_on_top_checkbox.stateChanged.connect(self.toggle_always_on_top)
        self.countdown_label = QLabel(f"{MAX_RUN_MINUTES:02d}:00 left")
        self.countdown_label.setStyleSheet("color: #D0D0D0; font-size: 12px; font-weight: 600;")
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #A9A9A9; font-size: 12px;")

        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.always_on_top_checkbox)
        controls.addWidget(self.countdown_label)
        controls.addWidget(self.status_label, 1)

        self.en_label = QLabel("Waiting for audio...")
        self.en_label.setWordWrap(True)
        self.en_label.setStyleSheet(
            "color: #F4F4F4; font-size: 24px; font-weight: 700;"
            "background-color: rgba(25, 25, 25, 210); border: 1px solid #303030;"
            "border-radius: 10px; padding: 10px 14px;"
        )

        self.zh_label = QLabel("Waiting for translation...")
        self.zh_label.setWordWrap(True)
        self.zh_label.setStyleSheet(
            "color: #89F0A4; font-size: 30px; font-weight: 700;"
            "background-color: rgba(20, 20, 20, 220); border: 1px solid #2A2A2A;"
            "border-radius: 10px; padding: 10px 14px;"
        )

        self.history_box = QPlainTextEdit()
        self.history_box.setReadOnly(True)
        self.history_box.setPlaceholderText("Transcript history will appear here...")
        self.history_box.setStyleSheet(
            "color: #DADADA; font-size: 13px; background-color: #101010; "
            "border: 1px solid #2B2B2B; border-radius: 8px; padding: 8px;"
        )
        self.history_box.setMinimumHeight(180)

        root.addLayout(controls)
        root.addWidget(self.en_label)
        root.addWidget(self.zh_label)
        root.addWidget(self.history_box)
        self.setLayout(root)
        self.resize(1000, 460)
        self.setMinimumSize(580, 300)

        self.toggle_shortcut = QShortcut(QKeySequence("Ctrl+Shift+H"), self)
        self.toggle_shortcut.activated.connect(self.toggle_visibility)

    def _position_bottom_center(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - 48
        self.move(x, y)

    def start_streaming(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.auto_stopped = False
        self.worker = StreamWorker()
        self.worker.caption_update.connect(self.on_update_caption)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.error_update.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Starting... (auto-stop in {MAX_RUN_MINUTES} min)")
        self.session_end_monotonic = time.monotonic() + (MAX_RUN_MINUTES * 60)
        self.update_countdown_label()
        self.countdown_timer.start(1000)
        self.auto_stop_timer.start(MAX_RUN_MINUTES * 60 * 1000)
        self.worker.start()

    def stop_streaming(self) -> None:
        if self.worker is None:
            return
        self.status_label.setText("Stopping...")
        self.stop_button.setEnabled(False)
        self.auto_stop_timer.stop()
        self.countdown_timer.stop()
        self.worker.stop()

    def on_worker_finished(self) -> None:
        self.auto_stop_timer.stop()
        self.countdown_timer.stop()
        self.session_end_monotonic = None
        self.countdown_label.setText(f"{MAX_RUN_MINUTES:02d}:00 left")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.auto_stopped:
            self.status_label.setText(f"Auto-stopped at {MAX_RUN_MINUTES} min. Press Start to run again.")
        elif self.status_label.text() == "Stopping...":
            self.status_label.setText("Stopped")

    def on_status_update(self, message: str) -> None:
        self.status_label.setText(message)

    def on_error(self, message: str) -> None:
        self.auto_stop_timer.stop()
        self.countdown_timer.stop()
        self.session_end_monotonic = None
        self.countdown_label.setText(f"{MAX_RUN_MINUTES:02d}:00 left")
        self.status_label.setText(f"Error: {message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_update_caption(self, en_interim: str, en_final: str, zh_final: str) -> None:
        if en_interim and en_interim != self._last_en_interim:
            self.en_label.setText(en_interim)
            self._last_en_interim = en_interim
        if en_final and en_final != self._last_en_final:
            self.en_label.setText(en_final)
            self._last_en_final = en_final
            self._last_en_interim = ""
            self.append_history(en_final, zh_final)
        if zh_final and zh_final != self._last_zh_final:
            self.zh_label.setText(zh_final)
            self._last_zh_final = zh_final

    def append_history(self, en_final: str, zh_final: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [f"[{ts}] EN: {en_final}"]
        if zh_final:
            lines.append(f"[{ts}] ZH: {zh_final}")
        lines.append("")
        self.history_box.appendPlainText("\n".join(lines))

    def clear_labels(self) -> None:
        self._last_en_final = ""
        self._last_zh_final = ""
        self._last_en_interim = ""
        self.en_label.setText("Waiting for audio...")
        self.zh_label.setText("Waiting for translation...")
        self.history_box.clear()

    def toggle_visibility(self) -> None:
        self.setVisible(not self.isVisible())

    def toggle_always_on_top(self, state: int) -> None:
        is_on_top = state == int(Qt.CheckState.Checked.value)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, is_on_top)
        self.show()

    def closeEvent(self, event) -> None:
        self.auto_stop_timer.stop()
        self.countdown_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(event)

    def on_auto_stop_timeout(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.auto_stopped = True
        self.status_label.setText(f"Max runtime reached ({MAX_RUN_MINUTES} min). Stopping...")
        self.stop_streaming()

    def update_countdown_label(self) -> None:
        if self.session_end_monotonic is None:
            self.countdown_label.setText(f"{MAX_RUN_MINUTES:02d}:00 left")
            return
        remaining = max(0, int(self.session_end_monotonic - time.monotonic()))
        mins = remaining // 60
        secs = remaining % 60
        self.countdown_label.setText(f"{mins:02d}:{secs:02d} left")


def run_demo(overlay: CaptionOverlay) -> None:
    demo_events = [
        ("we can", "", ""),
        ("we can actually translate", "", ""),
        ("", "we can actually translate", "we can actually translate"),
        ("our voices in real time", "", ""),
        ("", "our voices in real time", "our voices in real time"),
        ("using Azure Cognitive Services", "", ""),
        ("", "using Azure Cognitive Services", "using Azure Cognitive Services"),
    ]
    idx = {"value": 0}

    def push_next() -> None:
        i = idx["value"]
        if i >= len(demo_events):
            return
        overlay.on_update_caption(*demo_events[i])
        idx["value"] += 1

    timer = QTimer(overlay)
    timer.timeout.connect(push_next)
    timer.start(1100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live caption overlay")
    parser.add_argument("--demo", action="store_true", help="Show demo caption updates")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    overlay = CaptionOverlay()
    overlay.show()
    if args.demo:
        run_demo(overlay)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
