import os
import time

import ffmpeg
import numpy as np
from numpy._typing import NDArray
from pydub import AudioSegment
import soundfile as sf


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func


@st_time
def run_pydub_conversion(array_in: NDArray, sampling_rate: int, path_to_output_file: str, bitrate: str,
                         normalized=True):
    channels = 2 if (array_in.ndim == 2 and array_in.shape[1] == 2) else 1
    if normalized:  # each item in the array should be a float in [-1, 1)
        y = np.int16(array_in * 2 ** 15)
    else:
        y = np.int16(array_in)
    song = AudioSegment(y.tobytes(), frame_rate=sampling_rate, sample_width=2, channels=channels)
    song.export(path_to_output_file, format="mp3", bitrate=bitrate)


@st_time
def run_stream_conversion(array_in: NDArray, sampling_rate: int, path_to_output_file: str, bitrate: str,
                          normalized=True):
    channels = 2 if (array_in.ndim == 2 and array_in.shape[1] == 2) else 1
    if normalized:  # each item in the array should be a float in [-1, 1)
        array_in = np.int16(array_in * 2 ** 15)
    else:
        array_in = np.int16(array_in)

    ffmpeg_process = (
        ffmpeg
        .input('pipe:', format='s16le', ac=channels, ar=sampling_rate)
        # .output(path_to_output_file, acodec='mp3', format='s16le', audio_bitrate=bitrate)
        .output(path_to_output_file, acodec='aac', audio_bitrate=bitrate)

        # .output(out_filename, acodec='pcm_s16le', ac=2, ar='44.1k')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    in_frame = (
        np
        .frombuffer(array_in, np.int16)
    )
    ffmpeg_process.stdin.write(
        in_frame
        # .astype(np.int16)
        # .tobytes()
    )
    # close pipe
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


def main(path_to_wav: str):
    # load wav audio
    audio = sf.read(path_to_wav)

    # specify output filename
    path_to_output = "/tmp/test.aac"

    # init ffmpeg process
    run_stream_conversion(audio[0], 44100, path_to_output, "128k")
    run_pydub_conversion(audio[0], 44100, path_to_output, "128k")


if __name__ == '__main__':
    # path_to_wav = "/opt/data/wav/BE6JP2000005.wav"
    path_to_wav = "/opt/data/wav/AEA040700549.wav"
    # path_to_wav = "/opt/data/wav/FR0NT1703090.wav"
    # path_to_wav = "/opt/data/wav/DEF078003170.wav"

    main(path_to_wav)
