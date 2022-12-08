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

# placeholder
def write_to_mp3(array: NDArray, sampling_rate: int, path_to_mp3: str, normalized=True):
    channels = 2 if (array.ndim == 2 and array.shape[1] == 2) else 1
    if normalized:  # each item in the array should be a float in [-1, 1)
        y = np.int16(array * 2 ** 15)
    else:
        y = np.int16(array)
    song = AudioSegment(y.tobytes(), frame_rate=sampling_rate, sample_width=2, channels=channels)
    song.export(path_to_mp3, format="mp3", bitrate="192k")


def main():
    # load wav audio
    # audio = sf.read("/opt/data/wav/BE6JP2000005.wav")
    audio = sf.read("/opt/data/wav/AEA040700549.wav")
    audio = sf.read("/opt/data/wav/FR0NT1703090.wav")

    # un-normalize array and convert to int16
    array_in = np.int16(audio[0] * 2 ** 15)

    # specify output filename
    out_filename = "/tmp/test.mp3"

    # init ffmpeg process
    run_stream_conversion(array_in, out_filename)

@st_time
def run_stream_conversion(array_in: NDArray, out_filename: str):
    ffmpeg_process = (
        ffmpeg
        .input('pipe:', format='s16le', ac=2, ar='44.1k')
        .output(out_filename, acodec='mp3', format='s16le', audio_bitrate="192k")
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


if __name__ == '__main__':
    main()
