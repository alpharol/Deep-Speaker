import logging
import os
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--get_embeddings')  # p225 example.
    return arg_p




def main():
    args = arg_parse().parse_args()

    audio_reader = AudioReader(input_audio_dir="deep-speaker-data/VCTK-Corpus",
                               output_cache_dir="deep-speaker-data/cache/",
                               sample_rate=c.AUDIO.SAMPLE_RATE)

    if args.get_embeddings is not None:
        start_get = time.time()
        speaker_id = args.get_embeddings.strip()
        from unseen_speakers import inference_embeddings
        inference_embeddings(audio_reader, speaker_id)
        end_get = time.time()
        print("The time of regeneration inputs is {}".format(end_get-start_get))
        exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()