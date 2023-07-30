import sys
import os
import multiprocessing
import traceback
import tqdm
import logging
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa

now_dir = os.getcwd()
sys.path.append(now_dir)

from my_utils import load_audio
from slicer2 import Slicer

# Constants
DO_FORMANT = False
QUEFRENCY = 1.0
TIMBRE = 1.0
TARGET_SR = 16000

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"

# Logging
logging.basicConfig(
    filename=f"{exp_dir}/preprocess.log",
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


class PreProcess:
    def __init__(self, sr, exp_dir):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = 3.7
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir = f"{exp_dir}/1_16k_wavs"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            logger.warning(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        try:
            wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", 
                          self.sr, 
                          tmp_audio.astype(np.float32))
        except Exception as e:
            logger.error(f"Error writing file {self.gt_wavs_dir}/{idx0}_{idx1}.wav - {str(e)}")

        try:
            tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=TARGET_SR)  
            wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 
                          TARGET_SR, 
                          tmp_audio.astype(np.float32))
        except Exception as e:
            logger.error(f"Error writing file {self.wavs16k_dir}/{idx0}_{idx1}.wav - {str(e)}")

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr, DO_FORMANT, QUEFRENCY, TIMBRE)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while len(audio[int(self.sr * (self.per - self.overlap) * i):]) > self.tail * self.sr:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    tmp_audio = audio[start : start + int(self.per * self.sr)]
                tmp_audio = audio[start:]
                self.norm_write(tmp_audio, idx0, idx1)
                idx1 += 1
        except:
            logger.error(f"{path}->{traceback.format_exc()}")

    def pipeline_mp(self, infos, thread_n):
        for path, idx0 in tqdm.tqdm(
            infos, position=thread_n, leave=True, desc="thread:%s" % thread_n
        ):
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [(f"{inp_root}/{name}", idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]
            
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p], i)
            else:
                processes = [
                    multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::n_p], i))
                    for i in range(n_p)
                ]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
        except:
            logger.error(f"Failed to process. {traceback.format_exc()}")

if __name__ == "__main__":
    process = PreProcess(sr, exp_dir)
    logger.info("Starting preprocess")
    logger.info(sys.argv)
    process.pipeline_mp_inp_dir(inp_root, n_p)
    logger.info("End preprocess")