import os
import ffmpeg
import numpy as np
from numpy._typing import NDArray
from pydub import AudioSegment
import soundfile as sf

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

    audio = sf.read("/opt/data/wav/BE6JP2000005.wav")
    # array_in = np.random.random([1,10000])
    # array_in = audio[0][:,0]
    array_in = np.int16(audio[0] * 2 ** 15)
    out_filename = "/tmp/test.wav"


    ffmpeg_process = (
        ffmpeg
        .input('pipe:', format='s16le')
        # .output(out_filename, acodec='mp3', format='s16le',audio_bitrate="192k")
        .output(out_filename, acodec='pcm_s16le')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # while True:
    in_bytes = array_in.tobytes()
    # if not in_bytes:
    #     break
    in_frame = (
        np
        .frombuffer(in_bytes, np.int16)
    )

    ffmpeg_process.stdin.write(
        in_frame
        # .astype(np.int16)
        # .tobytes()
    )

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

if __name__ == '__main__':
    main()