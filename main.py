import asyncio
import contextlib
import os
import signal
from pathlib import Path

import deepl
import numpy as np
import pyaudiowpatch as pa
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    load_dotenv = None

# Env vars:
#   set DEEPGRAM_API_KEY=...
#   set DEEPL_API_KEY=...

TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 512
INTERIM_RESULTS = True
TARGET_LANG = "ZH"  # DeepL: "ZH" auto-selects Simplified/Traditional


def downmix_and_resample_linear16(
    in_bytes: bytes, in_channels: int, in_rate: int, out_rate: int
) -> bytes:
    audio = np.frombuffer(in_bytes, dtype=np.int16)
    if audio.size == 0:
        return b""

    if in_channels > 1:
        frame_count = audio.size // in_channels
        audio = audio[: frame_count * in_channels].reshape(frame_count, in_channels).mean(axis=1)

    audio = audio.astype(np.float32)

    if in_rate != out_rate and audio.size > 1:
        src_len = audio.shape[0]
        dst_len = max(1, int(src_len * out_rate / in_rate))
        src_x = np.arange(src_len, dtype=np.float32)
        dst_x = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)

    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio.tobytes()


def resolve_default_loopback_device(p: pa.PyAudio) -> dict:
    wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    for loopback in p.get_loopback_device_info_generator():
        if default_out["name"] in loopback["name"]:
            return loopback
    return p.get_default_wasapi_loopback()


async def main():
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    deepl_api_key = os.getenv("DEEPL_API_KEY")

    if not deepgram_api_key:
        raise RuntimeError(
            "Missing DEEPGRAM_API_KEY env var. Install python-dotenv or set env var in this shell."
        )
    if not deepl_api_key:
        raise RuntimeError(
            "Missing DEEPL_API_KEY env var. Install python-dotenv or set env var in this shell."
        )

    translator = deepl.Translator(deepl_api_key)
    deepgram = AsyncDeepgramClient(api_key=deepgram_api_key)

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    p = pa.PyAudio()
    device = resolve_default_loopback_device(p)
    in_channels = int(device["maxInputChannels"])
    in_rate = int(device["defaultSampleRate"])

    print(
        f"Using loopback: {device['index']} | {device['name']} "
        f"({in_channels}ch @ {in_rate} Hz)"
    )

    last_interim = ""

    def request_stop():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    print("Opening Deepgram stream...")
    async with deepgram.listen.v1.connect(
        model="nova-2",
        language="en-US",
        smart_format="true",
        encoding="linear16",
        channels="1",
        sample_rate=str(TARGET_SAMPLE_RATE),
        interim_results="true" if INTERIM_RESULTS else "false",
    ) as dg_connection:
        def audio_callback(in_data, frame_count, time_info, status):
            def push_chunk():
                if audio_queue.full():
                    try:
                        _ = audio_queue.get_nowait()  # drop oldest to avoid lag buildup
                    except asyncio.QueueEmpty:
                        pass
                try:
                    audio_queue.put_nowait(in_data)
                except asyncio.QueueFull:
                    pass

            loop.call_soon_threadsafe(push_chunk)
            return (None, pa.paContinue)

        stream = p.open(
            format=pa.paInt16,
            channels=in_channels,
            rate=in_rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=audio_callback,
        )
        stream.start_stream()

        async def sender():
            while not stop_event.is_set():
                raw = await audio_queue.get()
                payload = downmix_and_resample_linear16(
                    raw, in_channels=in_channels, in_rate=in_rate, out_rate=TARGET_SAMPLE_RATE
                )
                if payload:
                    await dg_connection.send_media(payload)

        async def receiver():
            nonlocal last_interim
            async for message in dg_connection:
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
                    print(f"[EN final] {text}")
                    print(f"[ZH final] {zh}")
                    print("-" * 30)
                    last_interim = ""
                elif text != last_interim:
                    print(f"[EN live ] {text}")
                    last_interim = text

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        print("Listening... Press Ctrl+C to stop.")
        try:
            await stop_event.wait()
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            with contextlib.suppress(Exception):
                await sender_task
            with contextlib.suppress(Exception):
                await receiver_task
            with contextlib.suppress(Exception):
                await dg_connection.send_finalize()
            with contextlib.suppress(Exception):
                await dg_connection.send_close_stream()
            with contextlib.suppress(Exception):
                stream.stop_stream()
                stream.close()
            p.terminate()
            print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
