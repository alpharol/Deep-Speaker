import dill
import logging
import numpy as np
import os
import pickle

from constants import c
from speech_features import get_mfcc_features_390

logger = logging.getLogger(__name__)


def data_to_keras(data):
    categorical_speakers = SpeakersToCategorical(data)
    kx_train, ky_train, kx_test, ky_test = [], [], [], []
    #ky_test = []
    for speaker_id in categorical_speakers.get_speaker_ids():
        d = data[speaker_id]
        y = categorical_speakers.get_one_hot_vector(d['speaker_id'])
        for x_train_elt in data[speaker_id]['train']:
            for x_train_sub_elt in x_train_elt:
                kx_train.append(x_train_sub_elt)
                ky_train.append(y)

        for x_test_elt in data[speaker_id]['test']:
            for x_test_sub_elt in x_test_elt:
                kx_test.append(x_test_sub_elt)
                ky_test.append(y)

    kx_train = np.array(kx_train)
    kx_test = np.array(kx_test)

    ky_train = np.array(ky_train)
    ky_test = np.array(ky_test)

    return kx_train, ky_train, kx_test, ky_test, categorical_speakers


def generate_features(audio_entities, max_count, progress_bar=False):
    features = []
    count_range = range(max_count)
    if progress_bar:
        from tqdm import tqdm
        count_range = tqdm(count_range)
    for _ in count_range:
        audio_entity = np.random.choice(audio_entities)
        voice_only_signal = audio_entity['audio_voice_only']
        cuts = np.random.uniform(low=1, high=len(voice_only_signal), size=2)
        signal_to_process = voice_only_signal[int(min(cuts)):int(max(cuts))]
        features_per_conv = get_mfcc_features_390(signal_to_process, c.AUDIO.SAMPLE_RATE, max_frames=None)
        if len(features_per_conv) > 0:
            features.append(features_per_conv)
    return features


def normalize(list_matrices, mean, std):
    return [(m - mean) / std for m in list_matrices]


class InputsGenerator:

    def __init__(self, cache_dir, audio_reader, max_count_per_class=500,
                 speakers_sub_list=None, multi_threading=False):
        self.cache_dir = cache_dir
        self.audio_reader = audio_reader
        self.multi_threading = multi_threading
        self.inputs_dir = os.path.join(self.cache_dir, 'inputs')
        self.max_count_per_class = max_count_per_class
        if not os.path.exists(self.inputs_dir):
            os.makedirs(self.inputs_dir)

        self.speaker_ids = self.audio_reader.all_speaker_ids if speakers_sub_list is None else speakers_sub_list

    def start_generation(self):
        logger.info('Starting the inputs generation...')
        #前面半部分是生成每一个说话人的pkl文件
        if self.multi_threading:
            num_threads = os.cpu_count()
            logger.info('Using {} threads.'.format(num_threads))
            parallel_function(self.generate_and_dump_inputs_to_pkl, sorted(self.speaker_ids), num_threads)
        else:
            logger.info('Using only 1 thread.')
            for s in self.speaker_ids:
                self.generate_and_dump_inputs_to_pkl(s)
        #****************************************************#
        #后面半部分，其实就是把前面生成的所有说话人的pkl文件整合到一起
        from glob import glob
        logger.info('Generating the unified inputs pkl file.')
        full_inputs = {}
        for inputs_filename in glob(self.inputs_dir + '/*.pkl', recursive=True):
            with open(inputs_filename, 'rb') as r:
                inputs = pickle.load(r)
                logger.info('Read {}'.format(inputs_filename))
            full_inputs[inputs['speaker_id']] = inputs
        full_inputs_output_filename = os.path.join(self.cache_dir, 'full_inputs.pkl')
        # dill can manage with files larger than 4GB.
        # dill其实就是pickle的扩展版，就是处理大数据集
        with open(full_inputs_output_filename, 'wb') as w:
            dill.dump(obj=full_inputs, file=w)
        logger.info('[DUMP UNIFIED INPUTS] {}'.format(full_inputs_output_filename))

    def generate_and_dump_inputs_to_pkl(self, speaker_id):
        """生成每一个说话人的pkl文件，这个任务是说话人识别，需要以说话人为单位"""
        if speaker_id not in c.AUDIO.SPEAKERS_TRAINING_SET:
            """如果不training set里就不要这个说话人了"""
            logger.info('Discarding speaker for the training dataset (cf. conf.json): {}.'.format(speaker_id))
            return

        output_filename = os.path.join(self.inputs_dir, speaker_id + '.pkl')
        if os.path.isfile(output_filename):
            logger.info('Inputs file already exists: {}.'.format(output_filename))
            return

        inputs = self.generate_inputs(speaker_id)  ###这个inputs作为内容写到了inputs文件夹中speaker_id.pkl中
        with open(output_filename, 'wb') as w:
            pickle.dump(obj=inputs, file=w)
        logger.info('[DUMP INPUTS] {}'.format(output_filename))

    def generate_inputs_for_inference(self, speaker_id):
        speaker_cache, metadata = self.audio_reader.load_cache([speaker_id])
        audio_entities = list(speaker_cache.values())
        logger.info('Generating the inputs necessary for the inference (speaker is {})...'.format(speaker_id))
        logger.info('This might take a couple of minutes to complete.')
        feat = generate_features(audio_entities, self.max_count_per_class, progress_bar=False)
        mean = np.mean([np.mean(t) for t in feat])
        std = np.mean([np.std(t) for t in feat])
        feat = normalize(feat, mean, std)
        return feat

    def generate_inputs(self, speaker_id):
        """生成一个说话人pkl文件的输入"""
        from audio_reader import extract_speaker_id
        per_speaker_dict = {}
        cache, metadata = self.audio_reader.load_cache([speaker_id])

        for filename, audio_entity in cache.items():
            speaker_id_2 = extract_speaker_id(audio_entity['filename'])
            assert speaker_id_2 == speaker_id, '{} {}'.format(speaker_id_2, speaker_id)
            if speaker_id not in per_speaker_dict:
                per_speaker_dict[speaker_id] = []
            per_speaker_dict[speaker_id].append(audio_entity)
        #per_speaker_dict = per_speaker_dict

        audio_entities = per_speaker_dict[speaker_id]
        cutoff = int(len(audio_entities) * 0.8)
        #训练集中这个说话人的句子80%，这个是占句子总数的，而不是一句话的长度
        audio_entities_train = audio_entities[0:cutoff]  ##list
        audio_entities_test = audio_entities[cutoff:]

        train = generate_features(audio_entities_train, self.max_count_per_class)
        test = generate_features(audio_entities_test, self.max_count_per_class)
        logger.info('Generated {}/{} inputs for train/test for speaker {}.'.format(self.max_count_per_class,
                                                                                   self.max_count_per_class,
                                                                                   speaker_id))

        mean_train = np.mean([np.mean(t) for t in train])
        std_train = np.mean([np.std(t) for t in train])

        train = normalize(train, mean_train, std_train)
        test = normalize(test, mean_train, std_train)  ###为什么测试集在归一化的时候要输入mean_train和std_train

        inputs = {'train': train, 'test': test, 'speaker_id': speaker_id,
                  'mean_train': mean_train, 'std_train': std_train}
        return inputs


class SpeakersToCategorical:
    def __init__(self, data):
        from keras.utils import to_categorical
        self.speaker_ids = sorted(list(data.keys()))
        self.int_speaker_ids = list(range(len(self.speaker_ids)))
        self.map_speakers_to_index = dict([(k, v) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.map_index_to_speakers = dict([(v, k) for (k, v) in zip(self.speaker_ids, self.int_speaker_ids)])
        self.speaker_categories = to_categorical(self.int_speaker_ids, num_classes=len(self.speaker_ids))

    def get_speaker_from_index(self, index):
        return self.map_index_to_speakers[index]

    def get_one_hot_vector(self, speaker_id):
        index = self.map_speakers_to_index[speaker_id]
        return self.speaker_categories[index]

    def get_speaker_ids(self):
        return self.speaker_ids


def parallel_function(f, sequence, num_threads=None):
    from multiprocessing import Pool
    pool = Pool(processes=num_threads)
    result = pool.map(f, sequence)
    cleaned = [x for x in result if x is not None]
    pool.close()
    pool.join()
    return cleaned
