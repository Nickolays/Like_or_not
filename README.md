<h1 align="center">Like or not</h1>

Бинарная классификация для ваших песен на те что могут понравится и не понравится
____

В современном мире существует большое количество музыки и всю её прослушать не представляется возможным. Но что делать когда из всего этого многообразия хочется 
найти ту песню, которая может вам понравится. Для таких случаев пригодится файл audio.py. Он поможет вам не пропустить потенциально хорошие песни. Я думаю многие слышали о том
что для каждого человека существует свой, определённый набор звуков, которые ему нравятся.

Большой проблемой может показаться скачивание файлов, однако есть решение этой проблемы. Вот несколько ссылок для скачивания песен из интернета:
- [Скачать музыку с сайта](https://stackoverflow.com/questions/68808045/how-can-i-download-music-files-from-websites-using-python)
- [Скачать музыку с YouTube](https://www.geeksforgeeks.org/download-video-in-mp3-format-using-pytube/)
- [Скачать музыку с Spotify](https://habr.com/ru/post/582170/)
- [Смена формата mp3 на wav](https://www.geeksforgeeks.org/convert-mp3-to-wav-using-python/)

Я старался упроситить интерфейс для пользователя. На вход класс принимает 2 обязательных параметра () и 4 необязательных ().

```python
like_or_not = audio.Like_or_not()
```

В классе реализованны следующие методы:
- show_libraries_version  - Покажет версии библиотек
- get_name_song - Получить pd.DataFrame с директорией, названем и таргетом
- get_X_y - Получить признаки для бинарной классификации
- save_img - Преобразовать и сохранить изображения спектрограмм (для LSTM и Conv2d)
- spec_Conv1D - Модель Свёрточной 1D сети (анализ по времени). Если параметр duration=None, то отмасштабирует все треки к длине самого маленького
- spec_Conv2D - Модель Свёрточной 2D сети (работа с фото)
- spec_LTSM - Модель рекурентной нейронной сети (принимает на вход изображение спектрограммы)
- fit - Обучение инициализированной модели
- predict - Делает предсказание обученной модели или загруженной



| LEFT | CENTER | RIGHT |
|-------|:-----:|-------:|
| Слева | Центр | Справа |
| текст | текст | текста |
