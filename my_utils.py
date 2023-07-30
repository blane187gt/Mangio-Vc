import ffmpeg
import numpy as np

import os
import sys

from shlex import quote as RQuote
import random

import csv

platform_stft_mapping = {
    'linux': 'stftpitchshift',
    'darwin': 'stftpitchshift',
    'win32': 'stftpitchshift.exe',
}

stft = platform_stft_mapping.get(sys.platform)

def CSVutil(file, rw, type, *args):
    if type == 'formanting':
        if rw == 'r':
            with open(file) as fileCSVread:
                csv_reader = list(csv.reader(fileCSVread))
                return (
                    csv_reader[0][0], csv_reader[0][1], csv_reader[0][2]
                ) if csv_reader is not None else (lambda: exec('raise ValueError("No data")'))()
        else:
            if args:
                doformnt = args[0]
            else:
                doformnt = False
            qfr = args[1] if len(args) > 1 else 1.0
            tmb = args[2] if len(args) > 2 else 1.0
            with open(file, rw, newline='') as fileCSVwrite:
                csv_writer = csv.writer(fileCSVwrite, delimiter=',')
                csv_writer.writerow([doformnt, qfr, tmb])
    elif type == 'stop':
        stop = args[0] if args else False
        with open(file, rw, newline='') as fileCSVwrite:
            csv_writer = csv.writer(fileCSVwrite, delimiter=',')
            csv_writer.writerow([stop])

def load_audio(file, sr, DoFormant, Quefrency, Timbre):
    converted = False
    DoFormant, Quefrency, Timbre = CSVutil('csvdb/formanting.csv', 'r', 'formanting')    
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        
        if (DoFormant.lower() == 'true'):
            numerator = round(random.uniform(1,4), 4)
            
            if not file.endswith(".wav"):
                
                if not os.path.isfile(f"{file_formanted}.wav"):
                    converted = True
                    #print(f"\nfile = {file}\n")
                    #print(f"\nfile_formanted = {file_formanted}\n")
                    converting = (
                        ffmpeg.input(file_formanted, threads = 0)
                        .output(f"{RQuote(file_formanted)}.wav")
                        .run(
                            cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                        )
                    )
                else:
                    pass
            file_formanted = f"{file_formanted}.wav" if not file_formanted.endswith(".wav") else file_formanted
            
            print(f" · Formanting {file_formanted}...\n")
            
            command = (
                f'{RQuote(stft)} -i "{RQuote(file_formanted)}" -q "{RQuote(Quefrency)}" '
                f'-t "{RQuote(Timbre)}" -o "{RQuote(file_formanted)}FORMANTED_{RQuote(str(numerator))}.wav"'
            )
            os.system(command)
            
            print(f" · Formanted {file_formanted}!\n")
            
            out, _ = (
                ffmpeg.input(f"{file_formanted}FORMANTED_{str(numerator)}.wav", threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )

            try: os.remove(f"{file_formanted}FORMANTED_{str(numerator)}.wav")
            except Exception as e: pass; print(f"couldn't remove formanted type of file due to {e}")
            
        else:
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")
    
    if converted:
        try: os.remove(file_formanted)
        except Exception as e: pass; print(f"Couldn't remove converted type of file due to {e}")
        converted = False
    
    return np.frombuffer(out, np.float32).flatten()
