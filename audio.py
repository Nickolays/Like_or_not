# Работа с данными
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import tqdm
import librosa, librosa.display
import cv2, skimage
import soundfile as sf
# Нейросеть
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

class Like_or_not:
    """
    Данный класс предназначен для бинарной классификации ваших песен на те которые могут вам понравится и могут
    не понравится

    Если вы уверены, что у вас все трэки имеют одинаковую частоту дискретизации (sample_rate), то тогда лучше
    использовать библиотеку SoundFile. Эта библиотека гараздо быстрее загружает песни.
    -------------------------------------------------------------------------------------

    :param like_path: str
        Путь к файлу с треками, которые нравятся
    :param d_like_path: str
        Путь к файлу с треками, которые не нравятся
    :param sample_rate: int, default = 22050
        Частота дискретизации. Сколько измерений будем делать за 1 секунду (1 секунда / sample_rate)
    :param figsize: tuple(int, int), default = (20, 11)
        Размер изображения при сохранении. При использовании Jupyter Notebook я рекомендую увелечить
        размер, а при использовании среды разработки (например PyCharm) оставить значения по умолчанию
    
    ---------------------------------------------------------------------------------------
    methods:
        show_libraries_version  - Покажет версии библиотек
        get_name_song - Получить pd.DataFrame с директорией, названем и таргетом
        get_X_y - Получить признаки для бинарной классификации
        save_img - Преобразовать и сохранить изображения спектрограмм (для LSTM и Conv2d)

        _load_songs - Загрузка песен в класс. (self._songs)

        __calc_spec - Расчёт необходимой спектрограммы
        __save_fig - Сохранение изображения в папку
    ---------------------------------------------------------------------------------------

    ВАЖНАЯ ДЕТАЛЬ, В soundfile НЕЛЬЗЯ ВРУЧНУЮ ВЫБИРАТЬ Sample Rate

    СДЕЛАТЬ ВЕРСИЮ 3, ГДЕ ВЕСЬ ЦИКЛ БУДЕТ ВЫПОЛНЯТСЯ ДЛЯ ОДНОГО ЗНАЧЕНИЯ (open song --> get spectr --> save_photo),
    чтобы не хранить в памяти все фото
    """

    def __init__(self, like_path: str, d_like_path: str, sample_rate=22050, img_size=254):
        # Пользовательские параметры
        if not isinstance(like_path, str):
            raise TypeError('Путь к файлам должен быть в строковом формате')
        self.like_path = like_path
        if not isinstance(d_like_path, str):
            raise TypeError('Путь к файлам должен быть в строковом формате')
        self.d_like_path = d_like_path
        self.sr = sample_rate
        # Фиксация размера картинки
        self.img_size = img_size

        # Внутренние параметры
        self._paths = None  # Расположение песен
        self.target = None  # Целевая метка
        # self._songs = None  # Массив с заргруженными песнями с формой (кол-во песен, sr*duration)
        self._spectrum = None  # Выбранный пользователем спектр

    def show_libraries_version(self):
        """
        Покажет необходимые версии библиотек для работы с классом
        """
        print(f'Your pandas version is {pd.__version__}, needed: 1.2.0 and more')
        print(f'Your numpy version is {np.__version__}, needed: 1.21.0 and more')
        print(f'Your sklearn version is {sklearn.__version__}, needed: 1.0.2 and more')
        print(f'Your tqdm version is {tqdm.__version__}, needed: 4.62.0 and more')
        print(f'Your librosa version is {librosa.__version__}, needed: 0.9.0 and more')
        print(f'Your Open-CV version is {cv2.__version__}, needed: 4.5.0 and more')
        print(f'Your TensorFlow version is {tf.__version__}, needed: 2.6.0 and more')
        print(f'Your TensorFlow version is {skimage.__version__}, needed: 0.20.0 and more')

    def __len__(self):
        if self._paths is None:
            self.get_name_song()
        return len(self._paths)

    def get_name_song(self):
        """
        Функция создаст Data Frame с названиями песен, их директорией и таргетом: 1 - Нравится и 0 - не нравится

        :return: pandas.DataFrame с названием песен, директорией и их целевой меткой
        """
        # Создадим массив с названием песен
        liked = os.listdir(self.like_path)
        d_liked = os.listdir(self.d_like_path)
        names = liked + d_liked

        # Создадим названия директорий
        paths = [self.like_path] * len(liked) + [self.d_like_path] * len(d_liked)

        # Создадим целевую переменную
        labels = [1] * len(liked) + [0] * len(d_liked)

        # Объеденим всё в одну таблицу
        df = pd.DataFrame(np.array([names, paths, labels]).T, columns=['song_name', 'path', 'target'])
        df['path'] = [os.path.join(df.path[i], df.song_name[i]) for i in range(len(df))]
        df['song_name'] = df['song_name'].apply(lambda x: x[:-4])

        # Перемешаем наши данные, для классификатора
        df = df.sample(len(df))
        df.reset_index(drop=True, inplace=True)

        # Запишем в Класс
        self._paths = [path for path in df.path]
        df['target'] = df.target.astype('int8')
        self.target = df.target.values.astype(np.uint8)

        return df

    def _load_song(self, path):
        """
        Метод загрузит песни из переданных в класс директорий песни
        """
        # print(' ----- Load audio ----- ')
        # self._songs = []
        # for path in tqdm.tqdm(self._paths, total=len(self._paths)):
            # Загрузим песню
            # Librosa
            # audio, sr = librosa.load(path, sr=self.sr, offset=self.start, duration=self.duration, dtype='float32')

        # Sound File
        data, sr = sf.read(path)
            # audio = data.sum(axis=1) / 2

            # Добавить
            # self._songs.append(data)
        # Расчитаем длительность песни
        # self.duration = self._songs[0].shape[0] / self.sr

        return data

    def get_X_y(self, df=None):
        """
        Функция загрузит аудио и посчитаем по ним необходимые признаки и метки для классификации

        :param df: pandas.DataFrame, default = None
            Data Frame с названиями песен и их директорией. По умолчанию сам сгенирирует DataFrame
        :return: X, y pandas.DataFrame, который содержит признаки и pandas.Series целевую переменную
        """

        if df is None:
            df = self.get_name_song()

        # Добавим в наш ДФ новые признаки
        feats_cols = ['crossings_zero', 'rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr'] \
                     + [f'mfcc_{i}' for i in range(1, 21)]
        feats = pd.DataFrame(np.zeros((len(df), len(feats_cols))), columns=feats_cols)
        df = pd.concat([df, feats], axis=1)

        print(' ----- Calculate features ----- ')
        for i, path in tqdm.tqdm(enumerate(self._paths), total=len(self._paths)):
            # Download audio
            audio = self._load_song(path)
            # Calculate duration
            duration = audio.shape[0] / self.sr

            # Расчёт признаков
            df['crossings_zero'].iloc[i] = np.sum(librosa.zero_crossings(audio, pad=False)) / duration
            df["rmse"].iloc[i] = np.mean(librosa.feature.rms(y=audio))
            df["chroma_stft"].iloc[i] = np.mean(librosa.feature.chroma_stft(y=audio, sr=self.sr))
            df["spec_cent"].iloc[i] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sr))
            df["spec_bw"].iloc[i] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sr))
            df["rolloff"].iloc[i] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sr))
            df["zcr"].iloc[i] = np.mean(librosa.feature.zero_crossing_rate(y=audio))

            # Хорошо подходит для речи
            for x, j in zip(librosa.feature.mfcc(y=audio, sr=self.sr)[:20], range(1, 21)):
                df["mfcc_{}".format(j)].iloc[i] = np.mean(x)

        return df.drop('target', axis=1), df.target

    def __calc_spec(self, audio, spec='stft'):
        """
        Функция считает спектрограммы, на выбор: 'stft', 'cqt', 'mfcc', 'chroma_stft', 'chroma_cqt'

        :param audio: array
            Входной массив по которому будут производиться расчёты
        :param spec: str, default = 'stft'
            Выбрать необходимую спектрограмму, которую нужно сохранить:
                - 'stft'
                - 'cqt'
                - 'mfcc'
                - 'chroma_stft'
                - 'chroma_cqt'
        :return: Посчитаный спектр
        """
        hop_length = 2048

        if spec == 'stft':
            stft = librosa.stft(y=audio, n_fft=2048)
            data = librosa.power_to_db(stft ** 2, ref=np.max)

        elif spec == 'cqt':
            CQT = librosa.cqt(y=audio, sr=self.sr)
            data = librosa.power_to_db(np.abs(CQT ** 2))

        elif spec == 'mfcc':
            MFCC = librosa.feature.mfcc(y=audio, sr=self.sr)
            MFCC = sklearn.preprocessing.scale(MFCC, axis=1)
            data = librosa.power_to_db(np.abs(MFCC) ** 2)

        elif spec == 'chroma_stft':
            chromagram = librosa.feature.chroma_stft(y=audio, sr=self.sr, hop_length=hop_length)
            data = librosa.power_to_db(chromagram)

        elif spec == 'chroma_cqt':
            c_cqt = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=hop_length)
            data = librosa.power_to_db(c_cqt)

        else:
            raise KeyError("Необходимо указать в атрибуте Like_or_not._spectrum один из: ('stft', 'cqt', 'mfcc', 'chroma_stft', 'chroma_cqt' )")

        return data

    def __save_fig(self, data, out_path, spec):
        """
        Сохранение фото спектрограммы в нужную папку

        :param data: np.array
            Массив с расчитаным спектром
        :param spec: str
            Наименование спектра
        :param out_path: str
            Где сохранить изображение
        """
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if spec == 'stft':
            p = librosa.display.specshow(data=data, ax=ax, y_axis='log', cmap='Greys')
        else:
            p = librosa.display.specshow(data=data, ax=ax, cmap='Greys')

        # Сохраняем
        plt.axis('off')
        fig.savefig(out_path, bbox_inches='tight', dpi=100, pad_inches=-0.05)

        del fig, data, p

        # Переделаем размер фото под тот, который нужен
        # img = cv2.imread(out_path, 0)
        # cv2.imwrite(out_path, cv2.resize(src=img, dsize=(self.img_size, self.img_size)))
        img = skimage.io.imread(out_path)
        img = skimage.transform.resize(img.astype('uint8'), (self.img_size, self.img_size, 3))
        skimage.io.imsave(out_path, img)

        return self

    def save_img(self, out_path, spec='stft'):
        """
        Метод сохраняет изображения спектрограмм для дальнейшей обработки Свёрточной нейронной сетью.
        Если у вас уже есть сохранённые изображения свёрток, то в методе spec_2D укажите параметр 'path' путь
        к ним.

        :param out_path: str, default: None
            Выбрать папку, в которую будут сохранены изображения спектрограм. Если выбран None, то в текущей
            директории будет создана папка с именем 'Spec_pictures'.
        :param spec: str, default: 'stft'
            Выбрать необходимую спектрограмму, которую нужно сохранить:
                - 'stft'
                - 'cqt'
                - 'mfcc'
                - 'chroma_stft'
                - 'chroma_cqt'
        """
        if type(out_path) != 'str':
            TypeError('Должен быть строковый формат: "str"')

        if self._paths is None:
            self.get_name_song()

        self._spectrum = spec
        # Создать папку, если нет папки с таким - же названием
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        # Сохраним таргеты
        np.savetxt(os.path.join(out_path, 'targets.csv'), np.round(self.target), delimiter=',')

        print(' ----- Saving pictures ----- ')
        for i, path in tqdm.tqdm(enumerate(self._paths), total=len(self._paths)):
            # Audio
            audio = self._load_song(path)
            # Расчёт спектра
            data = self.__calc_spec(audio=audio, spec=self._spectrum)
            # Сохраним спектрограмму в фото
            path_for_save = os.path.join(out_path, f'{self._spectrum}_{i + 1}.png')
            self.__save_fig(data=data, out_path=path_for_save, spec=self._spectrum)

        return self

    def prepare_test(self, in_paths: list, out_path: str):
        """
        Подготовит Тестовые данные для валидации

        :param in_paths:
        :param out_path:
        :return:
        """
        # Создать папку, если нет папки с таким - же названием
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        # Запишем все пути в один файл
        liked, dliked = in_paths[0], in_paths[1]
        paths = os.listdir(liked) + os.listdir(dliked)
        # Таргеты
        targets = [1] * len(os.listdir(liked)) + [0] * len(os.listdir(dliked))

        i = 0
        for path in tqdm.tqdm(paths, total=len(paths)):
            # Путь для загрузки
            if i < len(os.listdir(liked)):
                path_to_read = os.path.join(liked, path)
            else:
                path_to_read = os.path.join(dliked, path)
            # Загрузим песню
            data, sr = sf.read(path_to_read)
            data = self.__calc_spec(audio=data, spec=self._spectrum)

            # Сохраним спектрограмму в фото
            path_for_save = os.path.join(out_path, f'{self._spectrum}_{i + 1}.png')
            self.__save_fig(data=data, out_path=path_for_save, spec=self._spectrum)

            i += 1

        np.savetxt(os.path.join(out_path, 'targets.csv'), np.array(targets).astype(np.uint8), delimiter=',')

        return np.array(targets)