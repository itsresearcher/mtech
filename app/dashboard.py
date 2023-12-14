import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import chi2_contingency

def analyze_data(data, age_threshold, work_days_threshold):
    
    older_than_age_threshold_data = data[data['age'] > age_threshold].reset_index(drop=True)
    younger_than_age_threshold_data = data[data['age'] <= age_threshold].reset_index(drop=True)
    
    
    st.write(f"Описательная статистика для тех кто младше {age_threshold} включительно")
    st.write(younger_than_age_threshold_data.describe())

    st.write(f"Описательная статистика для тех кто старше {age_threshold}")    
    st.write(older_than_age_threshold_data.describe())

    fig, ax = plt.subplots()
    sns.histplot(data=data, x='work_days', hue='sex', bins=20, kde=True)
    plt.title('Гистограмма пропущенных дней по полу')
    plt.xlabel('Пропущенные дни')
    plt.ylabel('Частота')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(data=data, x='work_days', hue='age_group', bins=20, kde=True)
    plt.title('Гистограмма пропущенных дней в зависимости от возрастной группы')
    plt.xlabel('Пропущенные дни')
    plt.ylabel('Частота')
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(x='sex', y='work_days', data=data)
    plt.title('Ящик с усами пропущенных дней по полу')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x='age_group', y='work_days', data=data)
    plt.title('Ящик с усами пропущенных дней по возрасту')
    st.pyplot(fig)


    st.write("Проведем тестирования гипотез")
    st.write('Z-test:')
    st.write(z_test_for_sex(data))
    st.write("Теперь проведем такой же тест, только в отношении возраста")
    st.write(z_test_for_age(data))
    st.write("сделаем еще пару тестов, используя хи квадрат")

    st.write('Относительно пола')
    contingency_table = pd.crosstab(data['sex'],data['work_days']>work_days_threshold)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    st.write("Chi-Square Statistic:", chi2_stat)
    st.write("P-value:", p_value)
    st.write("Degrees of Freedom:", dof)
    st.write("Expected Frequencies Table:")
    st.write(expected)
    alpha = 0.05
    if p_value < alpha:
        st.write(f"Отклоняем нулевую гипотезу в пользу альтернативной: мужчины пропускают более {work_days_threshold} дней чаще, чем женщины.")
    else:
        st.write("Нет статистически значимых доказательств того, что мужчины пропускают более {work_days_threshold} дней чаще, чем женщины.")

    
    st.write('Относительно возраста')
    contingency = pd.DataFrame([[36, 21], [115, 65]], index=['>=35', '<35'], columns=['True', 'False'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
    st.write("Chi-Square Statistic:", chi2_stat)
    st.write("P-value:", p_value)
    st.write("Degrees of Freedom:", dof)
    st.write("Expected Frequencies Table:")
    st.write(expected)
    alpha = 0.05
    if p_value < alpha:
        st.write("Отклоняем нулевую гипотезу в пользу альтернативной: старшие работники пропускают более {work_days_threshold} дней чаще, чем молодые.")
    else:
        st.write("Нет статистически значимых доказательств того, что старшие работники пропускают более {work_days_threshold} дней чаще, чем молодые.")

def normality_test(data):
    stat, p_value = shapiro(data)

    st.write(f"Статистика теста: {stat}")
    st.write(f"P-value: {p_value}")

    if p_value > 0.05:
        st.write("Распределение не отличается от нормального (гипотеза о нормальности принимается).")
    else:
        st.write("Есть статистически значимые отличия от нормального распределения (гипотеза о нормальности отвергается).")

def z_test_for_sex(data):
    contingency_gender = pd.crosstab(data['sex'], data['work_days'] > work_days_threshold)
    prop_women = contingency_gender.iloc[0, 1] / contingency_gender.iloc[0].sum()
    z_stat, p_value = sm.stats.proportions_ztest(count=contingency_gender.iloc[1, 1],      
                                             nobs=contingency_gender.iloc[1].sum(),  
                                             alternative='larger',                   
                                             value=prop_women)                       

    st.write(f"Z-статистика: {z_stat}")
    st.write(f"p-значение: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        st.write(f"Отклоняем нулевую гипотезу в пользу альтернативной: мужчины пропускают более {work_days_threshold} дней чаще, чем женщины.")
    else:
        st.write("Нет статистически значимых доказательств того, что мужчины пропускают более {work_days_threshold} дней чаще, чем женщины.")

def z_test_for_age(data):

    contingency_age = pd.crosstab(data['age'], data['work_days'] > work_days_threshold)
    prop_young = len(data[(data['age'] <= age_threshold) & (data['work_days'] > work_days_threshold)]) / len(data[(data['age'] <= age_threshold)])

    z_stat_age, p_value_age = sm.stats.proportions_ztest(count=len(data[(data['age'] > age_threshold) & (data['work_days'] > work_days_threshold)]),    # количество старших работников, пропускающих более 2 дней
                                                 nobs=len(data[(data['age'] > age_threshold)]),         # общее количество старших работников
                                                 alternative='larger',                      # односторонняя альтернатива (больше)
                                                 value=prop_young)                          # доля молодых работников, берущих больничный более 2 дней

    st.write(f"Z-статистика для возраста: {z_stat_age}")
    st.write(f"p-значение для возраста: {p_value_age}")

    alpha_age = 0.05
    if p_value_age < alpha_age:
        st.write("Отклоняем нулевую гипотезу в пользу альтернативной: старшие работники пропускают более {work_days_threshold} дней чаще, чем молодые.")
    else:
        st.write("Нет статистически значимых доказательств того, что старшие работники пропускают более {work_days_threshold} дней чаще, чем молодые.")

# Заголовок дашборда
st.title("Проверка гипотез и анализ данных работников")
st.write("Для удобства, оставлены все методы, на которые влияет ползунок")
# Загрузка данных
data = pd.read_csv("app/М.Тех_Данные_к_ТЗ_DS.csv", encoding='cp1251')
data[['work_days', 'age', 'sex']] = data['Количество больничных дней,"Возраст","Пол"'].str.split(',', expand=True)
del data['Количество больничных дней,"Возраст","Пол"']

data['work_days'] = pd.to_numeric(data['work_days'])
data['age'] = pd.to_numeric(data['age'])
data['encoded'] = data['sex'].apply(lambda x: 1 if x == '"М"' else 0)


# Виджеты для ввода параметров
age_threshold = st.slider("Выберите возрастной порог:", min_value=20, max_value=60, value=35)
work_days_threshold = st.slider("Выберите порог отсутствия по болезни:", min_value=0, max_value=10, value=2)

data['age_group'] = data['age'].apply(lambda x: ' younger' if x <= age_threshold else 'older')

if st.button("Обновить результаты"):
    analyze_data(data, age_threshold, work_days_threshold)
