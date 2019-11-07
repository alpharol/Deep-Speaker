import logging
import os
import pickle
from glob import glob
import time
import librosa
import numpy as np
from tqdm import tqdm

from utils import parallel_function

logger = logging.getLogger(__name__)

SENTENCE_ID = 'sentence_id'
SPEAKER_ID = 'speaker_id'
FILENAME = 'filename'
new_path = "samples/"

def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    # 实现的是查找文件功能，**通配符只能和recursive=True一起用
    # 具体的使用可以查看https://blog.csdn.net/weixin_43216017/article/details/100920305
    return sorted(glob(directory + pattern, recursive=True))


def read_audio_from_filename(filename, sample_rate):
    #读取音频，mono设置为True表示使用单通道，输出的另一个是采样率，这里不需要
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    #读取出来的ndarray格式为(x,)，将其reshape之后格式为(x,1)
    audio = audio.reshape(-1, 1)
    return audio, filename


def trim_silence(audio, threshold):
    """Removes silence at the beginning and end of a sample."""
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(np.array(energy > threshold))
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    audio_trim = audio[0:0]
    left_blank = audio[0:0]
    right_blank = audio[0:0]
    if indices.size:
        audio_trim = audio[indices[0]:indices[-1]]
        left_blank = audio[:indices[0]]  # slice before.
        right_blank = audio[indices[-1]:]  # slice after.
    return audio_trim, left_blank, right_blank


def extract_speaker_id(filename):
    """找到这句话的说话人"""
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


class AudioReader:
    def __init__(self, input_audio_dir,
                 output_cache_dir,
                 sample_rate,
                 multi_threading=False):
        self.audio_dir = input_audio_dir
        self.cache_dir = output_cache_dir
        self.sample_rate = sample_rate
        self.multi_threading = multi_threading
        self.cache_pkl_dir = os.path.join(self.cache_dir, 'audio_cache_pkl')
        self.pkl_filenames = find_files(self.cache_pkl_dir, pattern='/**/*.pkl')##pkl_filenames保存的是所有说话人说的所有句子的文件

        logger.info('audio_dir = {}'.format(self.audio_dir))
        logger.info('cache_dir = {}'.format(self.cache_dir))
        logger.info('sample_rate = {}'.format(sample_rate))

        speakers = set()
        self.speaker_ids_to_filename = {}   ###保存的格式是字典，{speaker_id:[这个人说的句子]}，所有人所有句子
        for pkl_filename in self.pkl_filenames:
            speaker_id = os.path.basename(pkl_filename).split('_')[0]
            if speaker_id not in self.speaker_ids_to_filename:
                self.speaker_ids_to_filename[speaker_id] = []
            self.speaker_ids_to_filename[speaker_id].append(pkl_filename)
            speakers.add(speaker_id)
        self.all_speaker_ids = sorted(speakers)    ####保存了所有说话人，字典sorted之后是列表，保存了所有说话人的id

    def load_cache(self, speakers_sub_list=None):
        cache = {}
        metadata = {}

        ### 如果speakers_sub_list是None，那么filenames返回的是所有说话人的所有句子
        ### 否则的话，那就是speakers_sub_list中包括哪个说话人，filenames中就有这个人说的所有句子
        if speakers_sub_list is None:
            filenames = self.pkl_filenames
        else:
            filenames = []
            for speaker_id in speakers_sub_list:
                filenames.extend(self.speaker_ids_to_filename[speaker_id])
        #cache中保存的是：{音频名：音频信息（音频名，音频，音强大于95%的音频部分，左端静音时长，右端静音时长）}
        for pkl_file in filenames:
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
                if FILENAME in obj:
                    cache[obj[FILENAME]] = obj
        #metadata是字典，字典的每一个元素key是speaker_id，values还是一个字典，字典的key是sentence_id
        for filename in sorted(cache):
            speaker_id = extract_speaker_id(filename)
            if speaker_id not in metadata:
                metadata[speaker_id] = {}
            sentence_id = extract_sentence_id(filename)
            if sentence_id not in metadata[speaker_id]:
                metadata[speaker_id][sentence_id] = []
            metadata[speaker_id][sentence_id] = {SPEAKER_ID: speaker_id,
                                                 SENTENCE_ID: sentence_id,
                                                 FILENAME: filename}

        # metadata # small cache <speaker_id -> sentence_id, filename> - auto generated from self.cache.
        # cache # big cache <filename, data:audio librosa, blanks.>
        return cache, metadata

    def build_cache(self):
        #将所有的音频写入pkl文件
        if not os.path.exists(self.cache_pkl_dir):
            os.makedirs(self.cache_pkl_dir)
        logger.info('Nothing found at {}. Generating all the cache now.'.format(self.cache_pkl_dir))
        logger.info('Looking for the audio dataset in {}.'.format(self.audio_dir))
        audio_files = find_files(self.audio_dir)  #找到所有音频文件，以列表的形式存储
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, '请输入正确的路径以及glob通配符查找音频文件'
        logger.info('Found {} files in total in {}.'.format(audio_files_count, self.audio_dir))
        #assert len(audio_files) != 0

        if self.multi_threading:
            num_threads = os.cpu_count()
            print("共有 {} CPU可用".format(num_threads))
            time.sleep(5)
            parallel_function(self.dump_audio_to_pkl_cache, audio_files, num_threads)
        else:
            bar = tqdm(audio_files)
            for filename in bar:
                bar.set_description(filename)
                self.dump_audio_to_pkl_cache(filename)
            bar.close()

    def build_new_cache(self):
        #将所有的音频写入pkl文件
        if not os.path.exists(self.cache_pkl_dir):
            os.makedirs(self.cache_pkl_dir)
        logger.info('Nothing found at {}. Generating all the cache now.'.format(self.cache_pkl_dir))
        logger.info('Looking for the audio dataset in {}.'.format(new_path))
        audio_files = find_files(new_path)  #找到所有音频文件，以列表的形式存储
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, '请输入正确的路径以及glob通配符查找音频文件'
        logger.info('Found {} files in total in {}.'.format(audio_files_count, self.audio_dir))
        #assert len(audio_files) != 0

        if self.multi_threading:
            num_threads = os.cpu_count()
            print("共有 {} CPU可用".format(num_threads))
            time.sleep(5)
            parallel_function(self.dump_audio_to_pkl_cache, audio_files, num_threads)
        else:
            bar = tqdm(audio_files)
            for filename in bar:
                bar.set_description(filename)
                self.dump_audio_to_pkl_cache(filename)
            bar.close()

    def dump_audio_to_pkl_cache(self, input_filename):
        try:
            cache_filename = input_filename.split('/')[-1].split('.')[0] + '_cache'
            pkl_filename = os.path.join(self.cache_pkl_dir, cache_filename) + '.pkl'

            if os.path.isfile(pkl_filename):
                logger.info('[FILE ALREADY EXISTS] {}'.format(pkl_filename))
                return

            audio, _ = read_audio_from_filename(input_filename, self.sample_rate)  ##格式是ndarray，shape是(x,1)
            energy = np.abs(audio[:, 0])   ##对其中的每一个数取绝对值
            silence_threshold = np.percentile(energy, 95)   ###取95%分位数，也就是第0.95大的数
            offsets = np.where(energy > silence_threshold)[0]
            left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
            right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate

            obj = {'audio': audio,
                   'audio_voice_only': audio[offsets[0]:offsets[-1]],
                   'left_blank_duration_ms': left_blank_duration_ms,
                   'right_blank_duration_ms': right_blank_duration_ms,
                   FILENAME: input_filename}

            with open(pkl_filename, 'wb') as f:
                pickle.dump(obj, f)                ###把对象obj保存到文件中去
                logger.info('[DUMP AUDIO] {}'.format(pkl_filename))
        except librosa.util.exceptions.ParameterError as e:
            logger.error(e)
            logger.error('[DUMP AUDIO ERROR SKIPPING FILENAME] {}'.format(input_filename))
