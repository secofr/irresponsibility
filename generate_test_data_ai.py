import csv
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class RealisticDataGenerator:
    def __init__(self):
        self.female_names = [
            "Александра", "Алина", "Алиса", "Алла", "Анастасия", "Ангелина", "Анна", "Антонина",
            "Валентина", "Валерия", "Варвара", "Василиса", "Вера", "Вероника", "Виктория",
            "Галина", "Дарья", "Диана", "Ева", "Евгения", "Екатерина", "Елена", "Елизавета",
            "Жанна", "Злата", "Ирина", "Карина", "Кира", "Ксения", "Лариса", "Лидия", "Любовь",
            "Людмила", "Маргарита", "Марина", "Мария", "Надежда", "Наталья", "Нина", "Оксана",
            "Олеся", "Ольга", "Полина", "Раиса", "Регина", "Светлана", "София", "Тамара",
            "Татьяна", "Ульяна", "Юлия", "Яна"
        ]
        
        # Профессиональные группы с разными паттернами работы
        self.profession_patterns = {
            'office_worker': {'work_hours_range': (8, 10), 'home_hours_range': (12, 14), 'work_irresponsibility_range': (-20, 20)},
            'healthcare': {'work_hours_range': (10, 14), 'home_hours_range': (8, 12), 'work_irresponsibility_range': (-30, 10)},
            'teacher': {'work_hours_range': (6, 9), 'home_hours_range': (13, 16), 'work_irresponsibility_range': (-10, 30)},
            'creative': {'work_hours_range': (4, 12), 'home_hours_range': (10, 18), 'work_irresponsibility_range': (-50, 50)},
            'remote_worker': {'work_hours_range': (6, 8), 'home_hours_range': (14, 16), 'work_irresponsibility_range': (-40, 40)}
        }
        
        # Инициализация ML моделей
        self.work_model = None
        self.home_model = None
        self.scaler = StandardScaler()
        
    def create_synthetic_correlations(self):
        """Создает синтетические корреляции между признаками"""
        # Сильная отрицательная корреляция: больше работы -> меньше дома
        work_home_corr = -0.7
        
        # Умеренные корреляции с безответственностью
        work_irr_corr = 0.4  # Больше работы -> выше безответственность на работе
        home_irr_corr = -0.3  # Больше дома -> ниже безответственность дома
        
        return work_home_corr, work_irr_corr, home_irr_corr
    
    def generate_realistic_patterns(self, num_samples):
        """Генерирует реалистичные паттерны данных с использованием ML подходов"""
        
        # Создаем базовые нормальные распределения
        work_hours = np.clip(np.random.normal(8, 3, num_samples), 0, 24)
        
        # Создаем коррелированные данные для времени дома
        work_home_corr, work_irr_corr, home_irr_corr = self.create_synthetic_correlations()
        
        # Время дома коррелировано с временем работы (отрицательно)
        home_hours = np.clip(24 - work_hours + np.random.normal(0, 4, num_samples), 0, 24)
        
        # Генерируем безответственность на основе паттернов
        work_irresponsibility = []
        home_irresponsibility = []
        
        for i in range(num_samples):
            # Безответственность на работе зависит от часов работы
            work_irr = work_hours[i] * 2 + np.random.normal(0, 20)
            work_irr = np.clip(work_irr, -100, 100)
            work_irresponsibility.append(work_irr)
            
            # Безответственность дома обратно зависит от времени дома
            home_irr = -home_hours[i] * 1.5 + np.random.normal(0, 25)
            home_irr = np.clip(home_irr, -100, 100)
            home_irresponsibility.append(home_irr)
        
        return work_hours, home_hours, work_irresponsibility, home_irresponsibility
    
    def apply_profession_clusters(self, data, num_clusters=5):
        """Применяет кластеризацию для создания профессиональных групп"""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        return clusters
    
    def add_realistic_noise(self, data, noise_level=0.1):
        """Добавляет реалистичный шум к данным"""
        noisy_data = data + np.random.normal(0, noise_level * np.std(data), len(data))
        return np.clip(noisy_data, 0, 24) if np.max(data) <= 24 else np.clip(noisy_data, -100, 100)
    
    def generate_name_distribution(self, num_samples):
        """Генерирует распределение имен с реалистичной частотой"""
        # Создаем веса для каждого имени (более популярные имена имеют больший вес)
        name_weights = []
        
        # Распределение весов: некоторые имена встречаются чаще
        base_weights = {
            'Анна': 15, 'Мария': 14, 'Елена': 13, 'Ольга': 12, 'Наталья': 12,
            'Ирина': 11, 'Светлана': 10, 'Татьяна': 10, 'Екатерина': 9, 'Юлия': 9,
            'Анастасия': 8, 'Дарья': 8, 'Виктория': 7, 'Александра': 7, 'Полина': 6,
            'Ксения': 6, 'Валентина': 5, 'Людмила': 5, 'Галина': 5, 'Маргарита': 4,
            'Вера': 4, 'Любовь': 4, 'Нина': 4, 'Злата': 3, 'Алина': 3,
            'Алиса': 3, 'Алла': 3, 'Ангелина': 3, 'Антонина': 2, 'Валерия': 2,
            'Варвара': 2, 'Василиса': 2, 'Вероника': 2, 'Диана': 2, 'Ева': 2,
            'Евгения': 2, 'Елизавета': 2, 'Жанна': 2, 'Карина': 2, 'Кира': 2,
            'Лариса': 2, 'Лидия': 2, 'Марина': 2, 'Надежда': 2, 'Оксана': 2,
            'Олеся': 2, 'Раиса': 2, 'Регина': 2, 'София': 2, 'Тамара': 2,
            'Ульяна': 2, 'Яна': 2
        }
        
        # Создаем список весов в том же порядке, что и имена
        for name in self.female_names:
            name_weights.append(base_weights.get(name, 2))  # По умолчанию вес 2
        
        # Нормализуем веса
        total_weight = sum(name_weights)
        probabilities = [w/total_weight for w in name_weights]
        
        names = np.random.choice(self.female_names, size=num_samples, p=probabilities)
        return names
    
    def create_realistic_dataset(self, num_rows=15000):
        """Создает реалистичный набор данных"""
        
        num_rows = min(num_rows, 15000)
        
        print("Генерация реалистичных тестовых данных...")
        print("Создание корреляций и паттернов...")
        
        # Генерируем реалистичные паттерны
        work_hours, home_hours, work_irr, home_irr = self.generate_realistic_patterns(num_rows)
        
        # Добавляем профессиональные кластеры
        features = np.column_stack([work_hours, home_hours, work_irr, home_irr])
        clusters = self.apply_profession_clusters(features)
        
        # Применяем паттерны профессиональных групп
        for i, cluster in enumerate(clusters):
            pattern_key = list(self.profession_patterns.keys())[cluster % len(self.profession_patterns)]
            pattern = self.profession_patterns[pattern_key]
            
            # Корректируем данные согласно паттерну профессии
            work_hours[i] = np.clip(work_hours[i] * 0.7 + np.random.uniform(*pattern['work_hours_range']) * 0.3, 0, 24)
            home_hours[i] = np.clip(home_hours[i] * 0.7 + np.random.uniform(*pattern['home_hours_range']) * 0.3, 0, 24)
            work_irr[i] = np.clip(work_irr[i] * 0.6 + np.random.uniform(*pattern['work_irresponsibility_range']) * 0.4, -100, 100)
        
        # Генерируем имена с реалистичным распределением
        print("Генерация распределения имен...")
        names = self.generate_name_distribution(num_rows)
        
        # Создаем DataFrame
        data = []
        for i in range(num_rows):
            data.append({
                'Имя': names[i],
                'Часы работы': round(float(work_hours[i]), 2),
                'Процент безответственности (работа)': round(float(work_irr[i]), 2),
                'Часы дома': round(float(home_hours[i]), 2),
                'Процент безответственности (дом)': round(float(home_irr[i]), 2)
            })
            
            if (i + 1) % 1000 == 0:
                print(f"Обработано {i + 1} строк...")
        
        df = pd.DataFrame(data)
        
        # Добавляем финальную корректировку для реалистичности
        df = self.final_data_adjustment(df)
        
        return df
    
    def final_data_adjustment(self, df):
        """Финальная корректировка данных для большей реалистичности"""
        
        # Убеждаемся, что сумма часов работы и дома не превышает 24 (с небольшим допуском)
        total_hours = df['Часы работы'] + df['Часы дома']
        excess_mask = total_hours > 24
        
        # Корректируем превышающие значения
        if excess_mask.any():
            scale_factor = 24 / total_hours[excess_mask]
            df.loc[excess_mask, 'Часы работы'] *= scale_factor
            df.loc[excess_mask, 'Часы дома'] *= scale_factor
        
        # Добавляем реалистичные выбросы (5% данных)
        outlier_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
        
        for idx in outlier_indices:
            # Выбросы по времени работы (очень много или очень мало)
            if random.random() < 0.5:
                df.at[idx, 'Часы работы'] = random.choice([0, 1, 2, 22, 23, 24])
                df.at[idx, 'Часы дома'] = 24 - df.at[idx, 'Часы работы'] + random.uniform(-2, 2)
            # Выбросы по безответственности
            else:
                df.at[idx, 'Процент безответственности (работа)'] = random.choice([-100, 100])
                df.at[idx, 'Процент безответственности (дом)'] = random.choice([-100, 100])
        
        return df
    
    def save_to_csv(self, df, filename):
        """Сохраняет DataFrame в CSV файл"""
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"Данные сохранены в файл: {filename}")
        
    def analyze_dataset(self, df):
        """Анализирует сгенерированный набор данных"""
        print("\n" + "="*50)
        print("АНАЛИЗ СГЕНЕРИРОВАННЫХ ДАННЫХ")
        print("="*50)
        
        print(f"Общее количество записей: {len(df)}")
        print(f"Уникальных имен: {df['Имя'].nunique()}")
        
        print("\nСТАТИСТИКА ПО ЧАСАМ:")
        print(f"Часы работы: {df['Часы работы'].mean():.2f} ± {df['Часы работы'].std():.2f}")
        print(f"Часы дома: {df['Часы дома'].mean():.2f} ± {df['Часы дома'].std():.2f}")
        
        print("\nСТАТИСТИКА ПО БЕЗОТВЕТСТВЕННОСТИ:")
        print(f"Безответственность на работе: {df['Процент безответственности (работа)'].mean():.2f} ± {df['Процент безответственности (работа)'].std():.2f}")
        print(f"Безответственность дома: {df['Процент безответственности (дом)'].mean():.2f} ± {df['Процент безответственности (дом)'].std():.2f}")
        
        print("\nКОРРЕЛЯЦИИ:")
        correlation_matrix = df[['Часы работы', 'Часы дома', 'Процент безответственности (работа)', 'Процент безответственности (дом)']].corr()
        print("Корреляция работа-дом:", round(correlation_matrix.loc['Часы работы', 'Часы дома'], 3))
        print("Корреляция работа-безответственность_работа:", round(correlation_matrix.loc['Часы работы', 'Процент безответственности (работа)'], 3))
        print("Корреляция дом-безответственность_дом:", round(correlation_matrix.loc['Часы дома', 'Процент безответственности (дом)'], 3))
        
        print("\nТОП-10 САМЫХ ЧАСТЫХ ИМЕН:")
        name_counts = df['Имя'].value_counts().head(10)
        for name, count in name_counts.items():
            print(f"  {name}: {count} записей")

def main():
    """Основная функция программы"""
    generator = RealisticDataGenerator()
    
    # Генерируем данные
    df = generator.create_realistic_dataset(15000)
    
    # Сохраняем в CSV
    generator.save_to_csv(df, "test_data2.csv")
    
    # Анализируем данные
    generator.analyze_dataset(df)
    
    print("\nГенерация завершена успешно!")
    print("Файл 'test_data2.csv' готов к использованию.")

if __name__ == "__main__":
    main()