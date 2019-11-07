import logging
import os
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
#from utils import InputsGenerator


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--unseen_speakers')  # p225,p226 example.   
    arg_p.add_argument('--extra_speakers', action='store_true')   # PhilippeRemy
    return arg_p



def main():
    args = arg_parse().parse_args()

    if args.extra_speakers:
        input_audio_dir="samples/"
    else:
        input_audio_dir="deep-speaker-data/VCTK-Corpus"

    audio_reader = AudioReader(input_audio_dir=input_audio_dir,
                               output_cache_dir="deep-speaker-data/cache",
                               sample_rate=c.AUDIO.SAMPLE_RATE)

    if args.unseen_speakers is not None:
        start_unseen = time.time()
        unseen_speakers = [x.strip() for x in args.unseen_speakers.split(',')]
        from unseen_speakers import inference_unseen_speakers
        inference_unseen_speakers(audio_reader, unseen_speakers[0], unseen_speakers[1])
        end_unseen = time.time()
        print("The time of regeneration inputs is {}".format(end_unseen-start_unseen))
        exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
