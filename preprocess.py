import logging
import os
import shutil
import time
from argparse import ArgumentParser

from audio_reader import AudioReader
from constants import c
from utils import InputsGenerator


def arg_parse():
    arg_p = ArgumentParser()
    arg_p.add_argument('--regenerate_full_cache', action='store_true')  ###是否要重新生成cache文件
    arg_p.add_argument('--update_cache', action='store_true')           ###当新的音频进来的时候，我们也需要对其进行预处理
    arg_p.add_argument('--generate_training_inputs', action='store_true')   ###生成适合输入模型的inputs
    arg_p.add_argument('--multi_threading', action='store_true')
    return arg_p


def regenerate_full_cache(audio_reader, args):
    """只有在重新训练模型的时候需要重新生成cache文件"""
    """目的：生成所有音频文件的pkl文件"""
    """以字典的格式保存起来的内容包括：文件名，音频，音强大于95%的音频部分，左端静音时长，右端静音时长"""

    cache_output_dir = "deep-speaker-data/cache"
    print('The directory containing the cache is {}.'.format(cache_output_dir))
    print('Going to wipe out and regenerate the cache in 5 seconds. Ctrl+C to kill this script.')
    time.sleep(5)
    try:
        shutil.rmtree(cache_output_dir)
    except:
        pass
    os.makedirs(cache_output_dir)
    audio_reader.build_cache()



def generate_cache_from_training_inputs(audio_reader, args):
    """只有在重新训练模型的时候需要重新inputs"""
    cache_dir = "deep-speaker-data/cache"
    inputs_generator = InputsGenerator(cache_dir=cache_dir,
                                       audio_reader=audio_reader,
                                       max_count_per_class=1000,
                                       speakers_sub_list=None,
                                       multi_threading=args.multi_threading)
    inputs_generator.start_generation()



def main():
    args = arg_parse().parse_args()

    audio_reader = AudioReader(input_audio_dir="deep-speaker-data/VCTK-Corpus/",
                               output_cache_dir="deep-speaker-data/cache",
                               sample_rate=c.AUDIO.SAMPLE_RATE,
                               multi_threading=args.multi_threading)

    if args.regenerate_full_cache:
        start_regenerate = time.time()
        regenerate_full_cache(audio_reader, args)
        end_regenerate = time.time()
        print("The time of regeneration is {}".format(end_regenerate-start_regenerate))
        exit(1)

    if args.generate_training_inputs:
        start_inputs = time.time()
        generate_cache_from_training_inputs(audio_reader, args)
        end_inputs = time.time()
        print("The time of regeneration inputs is {}".format(end_inputs-start_inputs))
        exit(1)

    if args.update_cache:
        start_update = time.time()
        audio_reader.build_new_cache()
        end_update = time.time()
        print("The time of regeneration inputs is {}".format(end_update-start_update))
        exit(1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
