# Работа с данными
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import tqdm
import librosa, librosa.display
import cv2
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
    :param start: int, default = 0
        С какой секунды открыть песню
    :param duration: int, default = None
         Длительность в секундах. Если выбран None то загрузит песню целиком
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
        spec_Conv1D - Модель Свёрточной 1D сети (анализ по времени)
        spec_Conv2D - Модель Свёрточной 2D сети (работа с фото)
        spec_LTSM - Модель рекурентной нейронной сети (принимает на вход изображение спектрограммы)
        fit - Обучение инициализированной модели
        predict - Делает предсказание обученной модели или загруженной

        _load_songs - Загрузка песен в класс. (self._songs)
        _open_img_for_CNN - Загрузка изображений

        __calc_spec - Расчёт необходимой спектрограммы
        __set_photo_shape - Сохранить в пространоство имён размеры изображения
        __save_fig - Сохранение изображения в папку
        __get_callbacks - Вернёт все callbacks для обучения модели
        __resize_img_for_LSTM - Сменит размер изображения (для снижения расхода памяти). Если выбран chroma, то
               размер будет: (n, 12) или если mfcc: (n, 20) 
    ---------------------------------------------------------------------------------------

    ВАЖНАЯ ДЕТАЛЬ, В soundfile НЕЛЬЗЯ ВРУЧНУЮ ВЫБИРАТЬ Sample Rate
    """

    def __init__(self, like_path: str, d_like_path: str, start=0, duration=None, sample_rate=22050, figsize=(16, 8)):
        # Пользовательские параметры
        if not isinstance(like_path, str):
            raise TypeError('Путь к файлам должен быть в строковом формате')
        self.like_path = like_path
        if not isinstance(d_like_path, str):
            raise TypeError('Путь к файлам должен быть в строковом формате')
        self.d_like_path = d_like_path
        self.start = start
        self.duration = duration  # УМНОЖИТЬ НА SR, Если использовать SF
        self.sr = sample_rate  # Если использовать либросу, если sf - 48000
        # Фиксация размера картинки
        self.figsize = figsize
        plt.rcParams["figure.figsize"] = self.figsize

        # Внутренние параметры
        self._paths = None  # Расположение песен
        self.target = None  # Целевая метка
        self._songs = None  # Массив с заргруженными песнями с формой (кол-во песен, sr*duration)
        self.__min_song_shape = None  # Минимальная длительность песни
        self._pictures_dir = None  # Расположение папки с изображениями спектрограм для Свёрточной сети
        self._spectrum = None  # Выбранный пользователем спектр
        self._images = None  # Загруженные изображения
        self._model = None  # Хранение модели для метода fit
        self._model_name = None  # Название модели для правильной загрузки данных в методе fit
        self.__spec_height = None  # Высота изображения
        self.__spec_width = None  # Ширина изображения
        self.__count_dirs = None  # Для создания папок с изображениями спектрограмм
        self.__fitted_model = False # Обучена ли модель

    def __str__(self):
        return "Использовать файлы в формате 'wav'. Если вы уверены, что у вас все трэки имеют одинаковую частоту " \
               "дискретизации (sample_rate), то тогда лучше использовать библиотеку SoundFile. Эта библиотека гараздо " \
               "быстрее загружает песни. В методе load_songs она уже есть, необходимо только поменять либросу на sf "

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
        self.target = df.target.values.astype('int8')

        return df

    def __len__(self):
        if self._paths is None:
            self.get_name_song
        return len(self._paths)

    def _load_songs(self):
        """
        Метод загрузит песни из переданных в класс директорий песни
        """
        if self._paths is None:
            self.get_name_song()

        print(' ----- Load audio ----- ')
        self._songs = []
        durations = []
        for path in tqdm.tqdm(self._paths, total=len(self._paths)):
            # Загрузим песню
            # Librosa
            audio, sr = librosa.load(path, sr=self.sr, offset=self.start, duration=self.duration, dtype='float32')

            # Sound File
            #             if duration is None:
            #                 data, sr = sf.read(path, start=start, stop=None)
            #             else:
            #                 data, sr = sf.read(path, start=start, stop=(duration+start)*self.sr)
            #             audio = data.sum(axis=1) / 2

            # Добавить
            durations.append(audio.shape[0])
            self._songs.append(audio)

        # self._songs = np.array(self._songs, dtype='float32')
        self.__min_song_shape = min(durations)
        del durations

        return self

    def get_X_y(self, df=None):
        """
        Функция загрузит аудио и посчитаем по ним необходимые признаки и метки для классификации

        :param df: pandas.DataFrame, default = None
            Data Frame с названиями песен и их директорией. По умолчанию сам сгенирирует DataFrame
        :return: X, y pandas.DataFrame, который содержит признаки и pandas.Series целевую переменную
        """

        if df is None:
            df = self.get_name_song()
        if self._songs is None:
            self._load_songs()

        # Добавим в наш ДФ новые признаки
        feats_cols = ['duration', 'crossings_zero', 'rmse', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr'] \
                     + [f'mfcc_{i}' for i in range(1, 21)]
        feats = pd.DataFrame(np.zeros((len(df), len(feats_cols))), columns=feats_cols)
        df = pd.concat([df, feats], axis=1)

        print(' ----- Calculate features ----- ')
        for i, audio in tqdm.tqdm(enumerate(self._songs), total=len(self._songs)):
            # Расчёт признаков
            df['duration'].iloc[i] = audio.shape[0] / self.sr
            df['crossings_zero'].iloc[i] = np.sum(librosa.zero_crossings(audio, pad=False)) / df['duration'].iloc[i]
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
            raise KeyError("Необходимо выбрать из:  ('stft', 'cqt', 'mfcc', 'chroma_stft', 'chroma_cqt' )")

        return data

    def __set_photo_shape(self, path_img:str):
        """
        Метод запомнит размеры изображений

        :param path_img: str
            Путь к изображениям
        """
        files = os.listdir(path_img)
        get_shape = cv2.imread(os.path.join(path_img, files[0]),
                               0)  # 0 обозначает, что мы открываем изображение как GRAY
        if self._model_name == 'Conv2D':
            width, height = get_shape.shape[1], get_shape.shape[0]
            # width, height = get_shape.shape[0], get_shape.shape[1]
        elif self._model_name == 'LSTM':
            if self._spectrum in ('chroma_cqt', 'chroma_stft'):
                width, height = get_shape.shape[1], 12
            elif self._spectrum == 'mfcc':
                width, height = get_shape.shape[1], 20
            else:
                width, height = get_shape.shape[1], get_shape.shape[0]

        # Сохраним размеры
        self.__spec_height = height
        self.__spec_width = width

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
        fig.savefig(out_path, bbox_inches='tight', dpi=143, pad_inches=-0.05)

        del fig, data, p

    def save_img(self, out_path=None, spec='stft'):
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

        self._spectrum = spec

        if self.__count_dirs is None:
            self.__count_dirs = 0

        # Если папка не указана, то создать новую
        if out_path is None:
            # Текущая директория
            # directori = os.getcwd()
            # Создать папку, если нет папки с таким-же названием
            if not os.path.isdir(f"Spec_pictures_{self._spectrum}"):
                pic_dir = f'Spec_pictures_{self._spectrum}'
                os.mkdir(pic_dir)
                self._pictures_dir = pic_dir
            else:
                pic_dir = f'Spec_pictures_{self._spectrum}'
                self._pictures_dir = pic_dir

        else:
            self._pictures_dir = out_path

        if self._songs is None:
            self._load_songs()

        print(' ----- Saving pictures ----- ')
        for i, audio in tqdm.tqdm(enumerate(self._songs), total=len(self._songs)):
            # Расчёт спектра
            data = self.__calc_spec(audio=audio, spec=spec)

            # Сохраним спектрограмму в фото
            path_for_save = os.path.join(self._pictures_dir, f'{self._spectrum}_{i + 1}.png')
            self.__save_fig(data=data, out_path=path_for_save, spec=self._spectrum)

        # Определим размер входного изображения
        get_shape = cv2.imread(os.path.join(self._pictures_dir, f'{self._spectrum}_{1}.png'),
                               0)  # 0 обозначает, что мы открываем изображение как GRAY
        height, width = get_shape.shape[0], get_shape.shape[1]
        # height, width = get_shape.shape[1], get_shape.shape[0]

        # Сохраним размеры
        self.__spec_height = height
        self.__spec_width = width

        return self

    def _open_img_for_CNN(self, path_img=None):
        """
        Загрузит изображения спектрограмм для Свёрточный Нейронной Сети. Для быстроты изображения загружаются в
        оттенках серого.

        :param path_img: str, default = None
            Путь к папке с изображениями
        :return: np.array
            Вернёт массив с изображениями размером (n_photo, height, weight, 1) для Свёрточной сети 2D
        """

        # ~~~~~~~~~~~~~~~ Блок проверок ~~~~~~~~~~~~~~~
        if path_img is None:
            if self._pictures_dir is None:
                try:
                    self.save_img()
                except:
                    print('Сначала необходимо вызвать метод save_img для сохранения изображений')
            else:
                path_img = self._pictures_dir

        # Названия всех сохранённых изображений
        try:
            paths_with_imgs = os.listdir(path_img)
        except:
            print('Проверьте правльность указанного пути')

        if len(paths_with_imgs) != len(self._paths):
            print(
                f'Некправильное количество изображений в папке {self._paths}. Проверьте их количество, оно должно составлять: {len(self._paths)}')

        # Запомним размеры фото
        if self.__spec_height is None:
            self.__set_photo_shape(path_img=path_img)

        # Создание нового массива, куда загрузим фото
        if self._model_name == 'Conv2D':
            images = np.zeros(shape=(len(self._paths), self.__spec_width, self.__spec_height, 1))
        elif self._model_name == 'LSTM':
            if self._spectrum in ('chroma_cqt', 'chroma_stft'):
                images = np.zeros(shape=(len(self._paths), self.__spec_width, 12))
            elif self._spectrum == ('mfcc'):
                images = np.zeros(shape=(len(self._paths), self.__spec_width, 20))
            else:
                images = np.zeros(shape=(len(self._paths), self.__spec_width, self.__spec_height))

        print(' ------ Load Images ----- ')
        # Пройдёмся по каждому и запишем в новый numpu array
        for i, img_path in tqdm.tqdm(enumerate(paths_with_imgs), total=len(self._paths)):
            img = cv2.imread(os.path.join(path_img, img_path),   # путь к файлу
                             0) / 255  # 0 обозначает, что мы открываем изображение как GRAY
            if self._model_name == 'Conv2D':
                images[i, :, :, 0] = img.T
            elif self._model_name == 'LSTM':
                if self._spectrum in ('chroma_cqt', 'chroma_stft', 'mfcc'):
                    images[i, :, :] = self.__resize_img_for_LSTM(img.T)
                else:
                    images[i, :, :, 0] = img.T

        return images

    def __resize_img_for_LSTM(self, img):
        """
        Уменьшит входное изображение, до необходимого для рекурентной сети. mfcc до(n, 20), а chroma до(n, 12)

        :param img: np.array
            Изображение размером (x, y)
        :return:
            Изображение размером (x, 12 или 20)
        """
        if self._spectrum in ('chroma_cqt', 'chroma_stft'):
            delimetr = 12
        elif self._spectrum == 'mfcc':
            delimetr = 20

        new_img = np.zeros((img.shape[0], delimetr))
        j = 0
        for i in range(5, img.shape[1], int(img.shape[1]/delimetr)+1):
            new_img[:, j] = img[:, i]
            j += 1

        return new_img

    def spec_Conv2D(self, learning_rate=0.001, paht_with_imgs=None):
        """
        Метод сгенирирует модель для 2D(photo) свёрточной нейронной сети. На вход принимает массив формой
        (batch_size, height_photo, width_photo, 1). 1 означает, что мы используем чёрно-белое изображение, а не
        RGB

        :param learning_rate: float
            Скорость обучения нейронной сети
        :param paht_with_imgs: str, default = None
            Путь к изображениям спектрограмм если они уже у вас есть
        """

        if paht_with_imgs is None:
            if self.__spec_height is None:
                self.save_img()
        else:
            self.__set_photo_shape(paht_with_imgs)
            self._pictures_dir = paht_with_imgs

        self._model_name = 'Conv2D'
        h, w = self.__spec_height, self.__spec_width

        # Создание модели
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.__spec_width, self.__spec_height, 1)))
        # Построение модели
        if h > 1400 and w > 2800:
            # Если изображение большое, то уменьшить его в 2 раза с окном 5 на 5
            model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=2))
            h, w = h/2, w/2
        #
        model.add(keras.layers.Conv2D(20, (10, 20), padding='valid', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=2))
        h, w = h/2, w/2
        #
        if h > 300 and w > 600:
            model.add(keras.layers.Conv2D(50, (10, 10), padding='valid', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(5, 10), strides=2))
            h, w = h/2, w/2
        #
        model.add(keras.layers.Conv2D(70, (5, 10), padding='valid', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(4, 2)))
        h, w = h/2, w/4
        #
        model.add(keras.layers.Conv2D(100, (10, 10), padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(100, (5, 5), padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=3))
        h, w = h/3, w/3
        #
        if h > 38 and w > 38:
            model.add(keras.layers.Conv2D(200, (3, 3), padding='valid', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=4))
        elif h < 18 and w < 18:
            model.add(keras.layers.Conv2D(220, (3, 3), padding='valid', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))
        else:
            model.add(keras.layers.Conv2D(220, (3, 3), padding='valid', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=3))
        #
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(2048, activation='relu'))
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # Компиляция модели
        print(model.summary())
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self._model = model

        return self

    def spec_Conv1D(self, learning_rate=0.001):
        """
        Метод сгенирирует свёрточную нейронную сеть для 1D массивов. На вход принимает массив формой
        (batch_size, lenght_array, 1)

        :param learning_rate: float, default = 0.001
            Скорость обучения нейронной сети
        """

        if self._songs is None:
            self._load_songs()

        # Create model
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.__min_song_shape, 1)))

        # Если песня больше 4 минут
        if self.__min_song_shape > 60 * 4 * self.sr:
            model.add(keras.layers.Conv1D(filters=5, kernel_size=120, activation='relu', padding='same'))
            model.add(keras.layers.MaxPool1D(pool_size=30, strides=2, padding='valid'))

        # Если песня больше 3 минут, но не больше 4 минут
        if 60 * 3 * self.sr < self.__min_song_shape < 60 * 4 * self.sr:
            model.add(keras.layers.Conv1D(filters=8, kernel_size=120, activation='relu', padding='same'))
            model.add(keras.layers.MaxPool1D(pool_size=30, strides=2, padding='valid'))

        # Если песня больше 2 минут
        if 60 * self.sr < self.__min_song_shape < 121 * self.sr:
            model.add(keras.layers.Conv1D(filters=10,  # Количество слоёв
                                          kernel_size=60,  # Ядро свёртки
                                          activation='relu', padding='same',
                                          input_shape=(self.__min_song_shape, 1)))
            model.add(keras.layers.MaxPool1D(pool_size=10, strides=3, padding='valid'))  # 1 323 000, 882 000
            model.add(keras.layers.Dropout(rate=0.1))

        model.add(keras.layers.Conv1D(filters=20, kernel_size=30, activation='relu', padding='same'))
        model.add(keras.layers.MaxPool1D(pool_size=10, strides=3, padding='same'))  # 661 500, 441 000, 294 000
        model.add(keras.layers.Dropout(rate=0.1))
        # Если песня больше 10 cекунд
        if self.__min_song_shape > 10 * self.sr:
            model.add(keras.layers.Conv1D(filters=30, kernel_size=30, activation='relu', padding='same'))
            model.add(keras.layers.MaxPool1D(pool_size=10, strides=3))  # 330 750,  220 500, 98 000
            model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Conv1D(filters=40, kernel_size=20, activation='relu', padding='same'))
        model.add(keras.layers.MaxPool1D(pool_size=10, strides=4))  # 165 375,  110 250, 24 500
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Conv1D(filters=60, kernel_size=15, activation='relu', padding='same'))
        model.add(keras.layers.MaxPool1D(pool_size=5, strides=5))  # 82 687, 55 125, 4 900
        model.add(keras.layers.Conv1D(filters=80, kernel_size=15, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=80, kernel_size=5, activation='relu', padding='valid'))
        model.add(keras.layers.MaxPool1D(pool_size=3, strides=5))  # 41 343, 27 652, 980
        model.add(keras.layers.Conv1D(filters=120, kernel_size=5, activation='relu', padding='valid'))
        model.add(keras.layers.MaxPool1D(pool_size=3, strides=2))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=2048, activation='relu'))
        model.add(keras.layers.Dense(units=216, activation='relu'))
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        model.summary()
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self._model = model
        self._model_name = 'Conv1D'

        return self

    def spec_LSTM(self, learning_rate=0.001, paht_with_imgs=None):
        """
        Инициализирует Рекурентную Нейронную сеть. На вход принимает массив формой
        (batch_size, height_photo, width_photo)

        :param learning_rate: float
            Скорость обучения модели
        :param paht_with_imgs: str, default = None
            Путь к изображениям спектрограмм если они уже у вас есть
        """
        # Проверка папки
        if paht_with_imgs is None:
            if self.__spec_height is None:
                self.save_img(spec='chroma_stft')
        else:
            self._pictures_dir = paht_with_imgs
        # Сохраним название модели и получим размеры
        self._model_name = 'LSTM'
        self.__set_photo_shape(self._pictures_dir)
        # Сощдание модели
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.__spec_width, self.__spec_height)))
        model.add(keras.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(keras.layers.LSTM(100, activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(0.3))
        # Применим слой к каждому временному фрагменту ввода
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(1024, activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(32, activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(8, activation='relu')))
        #
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(2048, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        print(model.summary())
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self._model = model

        return self

    def __get_callbacks(self, patience=5, verbose=1):
        """
        :param patience: int, default = 5
            Количество эпох без уменьшения лосса на валидации.
        :param verbose: int or bool, default True or 1
            Выводить информацию (True) или нет (False)
        :return:
            -Раннюю остановку для предотвращения переобучения,
            -Снижение скорости обучения, если лосс на валидации не падает
            -Сохранение лучшей модели
        """
        # Добавим защиту от переобучения и уменьшение скорости обучения, если ошибка не будет уменьшаться
        early_stopping = keras.callbacks.EarlyStopping(
            patience=patience + 2,
            min_delta=0,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=verbose,
            mode='min',
            baseline=None,
        )
        plateau = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience,
            verbose=verbose,
            mode='min',
        )
        chechpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('saved_model', 'models.hdf5'),
            monitor='val_loss', verbose=verbose, save_best_only=True,
        )

        return [early_stopping, plateau, chechpoint]

    def fit(self, epochs=50, batch_size=5, n_splits=5, device='/cpu:0', patience=5, verbose=True):
        """
        Метод предназначен для обучения выбранной модели. Обучение происходит с помощью кросс-валидации по
        n_splits фолдам. Если у вас уже есть обученная модель, напишите название папки в параметр path_img.

        :param epochs: int, default = 100
            Количество 'проходов' модели
        :param batch_size: int, default = 5
            На сколько частей делить выборку. Всё зависит от ваших мощностей, чем больше значение, тем меньше памяти требуется
        :param n_splits: int, default = 5
            Количество фолдов, на которое будет разбиваться выборка при обучении
        :param device: str, default = '/cpu:0'
            На чём вы собираетесь обучать модель:
                - '/cpu:0'
                - '/gpu:0'
        :param patience: int, default = 5
            Количество эпох, после которых уменьшиться скорость обучения, если лосс на валидационной
            выборке не улучшиться. После patience+2 модель остановиться и перейдёт к следующему фолду
        :param verbose: int or bool, default = True
            Нужно ли выводить информацию об обучении модели

        """
        from sklearn.model_selection import StratifiedKFold

        # Зададим необходимы значения
        BATCH_SIZE = batch_size
        EPOCHS = epochs

        # Разобьём выборку
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        if self._model is None:
            print('Вы не инициализировали модель')

        if self._model_name in ('Conv2D', 'LSTM'):
            X = self._open_img_for_CNN(path_img=self._pictures_dir)

        elif self._model_name == 'Conv1D':
            if self._songs is None:
                self._load_songs()
            # Сделаем все песни одной длинны
            X = [song[:self.__min_song_shape] for song in self._songs]
            X = np.array(X)

        vals_acc = []
        # Обучение
        for train_indx, test_indx in kf.split(X, self.target):
            X_train = X[train_indx]
            X_val = X[test_indx]
            y_train, y_val = self.target[train_indx], self.target[test_indx]
            with tf.device(device):
                history = self._model.fit(X_train, y_train,
                                          validation_data=(X_val, y_val),
                                          callbacks=self.__get_callbacks(patience=patience, verbose=verbose),
                                          verbose=verbose,
                                          batch_size=BATCH_SIZE,
                                          epochs=EPOCHS,
                                          )
            vals_acc.append(history.history['val_accuracy'][-1])

        print(np.mean(vals_acc))

        self.__fitted_model = True

        if self._model_name == 'Conv1D':
            del X

        return self

    def predict(self, test_path: str, saved_model=None):
        """
        Вернёт бинарное предсказание тестовых трэков, понравятся они вам или нет.
        Если у вас уже есть обученая модель вы можете загрузить её, указав путь к ней в параметре saved_model.
        Также неоходимо указать параметр класса: Like_or_not._spectrum = '1 из 5 доступных':
                - 'stft'
                - 'cqt'
                - 'mfcc'
                - 'chroma_stft'
                - 'chroma_cqt'

        :param test_path: 'str'
            Путь к папке с песнями, для которых нужно сделать прогноз
        :param saved_model: 'str'
            Если у вас уже есть обученная можель, вы можете её загрузить и сделать предсказание по ней

        :return: pandas.DataFrame
            Вернёт название трэков, а также предсказанние, понравится ли вам загруженные песни
        """
        if not isinstance(test_path, str):
            raise TypeError('Путь к файлам должен быть в строковом формате')

        if saved_model == None and self.__fitted_model == False:
            return print('Вы не обучили модель, пожалуйста выберите модель и затемвызовите метод fit(), перед методом predict()')

        # Запишем все названия
        test_audio_path = os.listdir(test_path)

        # Узнаем размер изображения. Это необходимо для задания формы массива
        if self.__spec_height is None and self._model_name != 'Conv1D':
            # Расчёт спектра
            audio, sr = librosa.load(os.path.join(test_path, test_audio_path[0]), sr=self.sr,
                                     offset=self.start, duration=self.duration, dtype='float32')
            data = self.__calc_spec(audio=audio, spec=self._spectrum)
            # Сохраним спектрограмму в фото
            dir_w_photo = 'photo_for_shape'
            path_for_save = os.path.join(dir_w_photo, 'test_photo.png')
            # Если нет папки с таким названием
            if not os.path.isdir(dir_w_photo):
                os.mkdir(dir_w_photo)
            else:
                os.remove(path_for_save)
                os.rmdir(dir_w_photo)
                os.mkdir(dir_w_photo)
            self.__save_fig(data=data, out_path=path_for_save, spec=self._spectrum)
            self.__set_photo_shape(dir_w_photo)
            os.remove(path_for_save)
            os.rmdir(dir_w_photo)

        if self._model_name == 'Conv2D':
            # Создание нового массива, куда загрузим тестовые фото
            X_test = np.zeros(shape=(len(test_audio_path), self.__spec_width, self.__spec_height, 1))
        elif self._model_name == 'LSTM':
            # Для сокращения размера
            if self._spectrum in ('chroma_cqt', 'chroma_stft'):
                X_test = np.zeros(shape=(len(test_audio_path), self.__spec_width, 12))
            elif self._spectrum == 'mfcc':
                X_test = np.zeros(shape=(len(test_audio_path), self.__spec_width, 20))
            else:
                X_test = np.zeros(shape=(len(test_audio_path), self.__spec_width, self.__spec_height))
        elif self._model_name == 'Conv1D':
            X_test = np.zeros((len(test_audio_path), self.__min_song_shape, 1))
        else:
            raise Exception("Вы не указали модель. Пожалуйста выберите вашу обученную модель из (LSTM, "
                  "Conv1D, Conv2D) и укажите ёё в параметре Like_or_not._model_name = your_model_name")

        # Создание папки для тестовых изображений
        i = 0
        if not os.path.isdir(f'Test_spec_{self._spectrum}'):
            test_dir = f'Test_spec_{self._spectrum}'
            os.mkdir(test_dir)
        else:
            test_dir = f'Test_spec_{self._spectrum}'

        print(' ----- Load test audio ----- ')
        for path in tqdm.tqdm(test_audio_path, total=len(test_audio_path)):
            # Загрузим песню
            # Librosa
            path_to_open = os.path.join(test_path, path)
            test_audio, sr = librosa.load(path_to_open, sr=self.sr, offset=self.start, duration=self.duration, dtype='float32')
            # --- Создание признаков ---
            # Какую модель используем
            if self._model_name in ('Conv2D', 'LSTM'):
                # Сохранить фото
                path_for_save = os.path.join(test_dir, (self._spectrum + f'_{i}.png'))
                data = self.__calc_spec(test_audio, self._spectrum)
                self.__save_fig(data=data, out_path=path_for_save, spec=self._spectrum)
                # Откроем и запишем фото
                img = cv2.imread(path_for_save, 0) / 255  # 0 обозначает, что мы открываем изображение как GRAY
                if self._model == 'Conv2D':
                    X_test[i, :, :, 0] = img.T
                elif self._model == 'LSTM':
                    #if self._spectrum in ('chroma_cqt', 'chroma_stft'):
                    X_test[i, :, :] = self.__resize_img_for_LSTM(img.T)
                    # elif self._spectrum == 'mfcc':


            elif self._model_name == 'Conv1D':
                # Добавить аудио в список
                if test_audio.shape[0] < self.__min_song_shape and self.__min_song_shape is not None:
                    print(f'Файл {test_audio_path[i]} имеет меньшую длительность песни, чем на обучающей выборке.'
                          f'{test_audio.shape[0] / self.sr}, а на обучающей {self.__min_song_shape / self.sr}')
                    X_test[i, :test_audio.shape[0], 0]
                elif self.__min_song_shape is None:
                    min_duration_test = int(input('Пожалуйста введите минимальную длительность трека'))
                    X_test[i, :min_duration_test, 0] = test_audio[:min_duration_test]
                else:
                    X_test[i, :, 0] = test_audio[:self.__min_song_shape]

            i += 1

        if saved_model is None:
            predict = self._model.predict(X_test)
        else:
            new_model = keras.models.load_model(saved_model)
            predict = new_model.predict(X_test)

        pred_df = pd.DataFrame(test_audio_path, columns=['Name'])
        pred_df['Predict'] = np.round(predict)
        return pred_df
