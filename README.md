# Тестовое задание в MTech

## Описание проекта

Данный проект осуществляет анализ данных с целью проверки двух гипотез:

1. Мужчины пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще женщин.
2. Работники старше 35 лет (age) пропускают в течение года более 2 рабочих дней (work_days) по болезни значимо чаще своих более молодых коллег.

Данные для анализа содержатся в файле "М.Тех_Данные_к_ТЗ_DS".

## Структура проекта

Проект состоит из двух основных частей:

1. Jupyter Notebook: Файл в формате Jupyter Notebook содержит код для проведения статистического анализа, проверки гипотез и визуализации результатов. Весь код аккуратно оформлен, содержит графики и подробные комментарии по логике решения.

2. Streamlit Dashboard: Дашборд, созданный с использованием библиотеки Streamlit, предоставляет простой интерфейс для загрузки данных, задания параметров и отображения результатов проверки гипотез. Дашборд также включает в себя графики распределений и подробные объяснения критериев проверки.

## Как использовать проект

### Jupyter Notebook

Для ознакомления с анализом данных и проверкой гипотез можно использовать Jupyter Notebook. Запустите файл "mtech_chapanov.ipynb" в среде Jupyter Notebook, следуя шагам и комментариям внутри файла.

### Docker

Для удобства развертывания проекта включен Dockerfile. Следуйте инструкциям по сборке и запуску Docker-контейнера для быстрого развертывания проекта.

##### Установка
Проследуйсте инструкциям по ссылке ниже
    ```
    https://docs.docker.com/engine/install/
    ```

##### Запуск
Для запуска необходимо создать контейнер
    ```
    docker build -t your_image_name .
    ```

Затем запустить его.
    ```
    docker run -p 8501:8501 your_image_name
    ```


Откройте веб-браузер и перейдите по адресу, указанному в выводе команды. Вас встретит интерфейс дашборда, где вы сможете загрузить данные, задать параметры и увидеть результаты проверки гипотез.

## Структура репозитория

1. `mtech_chapanov.ipynb`: Jupyter Notebook с кодом анализа данных и проверки гипотез.
2. `requirements.txt`: Файл с перечислением необходимых библиотек.
3. `Dockerfile`: Файл для сборки Docker-контейнера.
4. `README.md`: Описание проекта.
5. `data/`: Папка, содержащая файл с данными "М.Тех_Данные_к_ТЗ_DS".
6. `dashboard.py`: Python-скрипт с кодом для создания Streamlit Dashboard.
7. `М.Тех_Данные_к_ТЗ_DS.csv`: Данные для анализа.


## Контакты

Если у вас есть вопросы или предложения по улучшению проекта, свяжитесь со мной по следующим контактам:

- **Email:** abchapanov@gmail.com
- **GitHub:** [GitHub профиль](https://github.com/itsresearcher)

Благодарю за внимание к проекту!
