import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# Настройка шрифтов для корректного отображения кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

def create_corrected_plot(csv_file):
    """
    Строит исправленный XY plot с ограничениями по оси Y от -24 до +24
    """
    
    # Читаем данные из CSV
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Подсчитываем частоту имен
    name_counts = {}
    for name in df['Имя']:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    # Размеры точек на основе частоты имен (минимальный размер 20, максимальный 200)
    point_sizes = [min(200, max(20, name_counts[name] * 15)) for name in df['Имя']]
    
    # ВРЕМЯ ДОМА - положительные значения (0 до 24)
    home_scatter = ax.scatter(df['Процент безответственности (дом)'], 
                             df['Часы дома'],  # Положительные значения
                             s=point_sizes, 
                             alpha=0.6,
                             c=df['Часы дома'], 
                             cmap='viridis',
                             label='Время дома',
                             edgecolors='black', 
                             linewidth=0.5)
    
    # ВРЕМЯ НА РАБОТЕ - отрицательные значения (-24 до 0)
    work_scatter = ax.scatter(df['Процент безответственности (работа)'], 
                             -df['Часы работы'],  # Отрицательные значения
                             s=point_sizes, 
                             alpha=0.6,
                             c=df['Часы работы'], 
                             cmap='plasma',
                             label='Время на работе',
                             edgecolors='black', 
                             linewidth=0.5)
    
    # Настраиваем график
    ax.set_xlabel('Процент безответственности', fontsize=14, fontweight='bold')
    ax.set_ylabel('Время (часы)', fontsize=14, fontweight='bold')
    ax.set_title('Распределение безответственности', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Устанавливаем ограничения по оси Y
    ax.set_ylim(-24, 24)
    
    # Добавляем горизонтальные линии для разделения областей
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axhline(y=12, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-12, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Добавляем вертикальную линию нулевой безответственности
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
               label='Нулевая безответственность')
    
    # Добавляем подписи областей
    ax.text(0.02, 0.85, 'ВРЕМЯ ДОМА', transform=ax.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(0.02, 0.15, 'ВРЕМЯ НА РАБОТЕ', transform=ax.transAxes, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Настраиваем сетку
    ax.grid(True, alpha=0.3)
    
    # Настраиваем деления на оси Y
    y_ticks = [-24, -18, -12, -6, 0, 6, 12, 18, 24]
    y_labels = ['24', '18', '12', '6', '0', '6', '12', '18', '24']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    
    # Настраиваем деления на оси X
    x_ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=12)
    
    # Добавляем аннотации для топ-5 самых частых имен
    top_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for name, count in top_names:
        name_data = df[df['Имя'] == name]
        if len(name_data) > 0:
            # Берем первую точку для аннотации (можно выбрать медианные значения)
            median_home_x = name_data['Процент безответственности (дом)'].median()
            median_home_y = name_data['Часы дома'].median()
            median_work_x = name_data['Процент безответственности (работа)'].median()
            median_work_y = -name_data['Часы работы'].median()
            
            # Аннотация для времени дома
            ax.annotate(f'{name}', 
                       xy=(median_home_x, median_home_y), 
                       xytext=(10, 10),
                       textcoords='offset points', 
                       fontsize=9, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))
            
            # Аннотация для времени работы
            ax.annotate(f'{name}', 
                       xy=(median_work_x, median_work_y), 
                       xytext=(10, -15),
                       textcoords='offset points', 
                       fontsize=9, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
    
    # Добавляем легенду
    ax.legend(loc='upper right', fontsize=12)
    
    # Добавляем цветовые шкалы
    cbar_home = plt.colorbar(home_scatter, ax=ax, orientation='vertical', pad=0.1)
    cbar_home.set_label('Часы дома', rotation=270, labelpad=15, fontsize=12)
    
    cbar_work = plt.colorbar(work_scatter, ax=ax, orientation='vertical', pad=0.15)
    cbar_work.set_label('Часы работы', rotation=270, labelpad=15, fontsize=12)
    
    # Добавляем информационную панель
    info_text = f"Всего записей: {len(df)}\nУникальных имен: {len(name_counts)}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('corrected_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Выводим статистику
    print("Статистика данных:")
    print(f"Всего записей: {len(df)}")
    print(f"Уникальных имен: {len(name_counts)}")
    print("\nТоп-5 самых частых имен:")
    for i, (name, count) in enumerate(top_names, 1):
        print(f"{i}. {name}: {count} упоминаний")
    
    print(f"\nДиапазон времени дома: {df['Часы дома'].min():.1f} - {df['Часы дома'].max():.1f} часов")
    print(f"Диапазон времени работы: {df['Часы работы'].min():.1f} - {df['Часы работы'].max():.1f} часов")
    print(f"Диапазон безответственности дома: {df['Процент безответственности (дом)'].min():.1f} - {df['Процент безответственности (дом)'].max():.1f}%")
    print(f"Диапазон безответственности работы: {df['Процент безответственности (работа)'].min():.1f} - {df['Процент безответственности (работа)'].max():.1f}%")

# Основная программа
if __name__ == "__main__":
    # Генерируем данные если файл не существует
    try:
        pd.read_csv("test_data2.csv", encoding='utf-8')
        print("Файл test_data2.csv найден. Загружаем данные...")
    except FileNotFoundError:
        print("Файл test_data2.csv не найден. Генерируем данные...")
        # Импортируем и запускаем функцию генерации
        import sys
        import os
        # Добавляем текущую директорию в путь для импорта
        sys.path.append(os.path.dirname(__file__))
        from generate_test_data import generate_test_data
        generate_test_data("test_data.csv", 15000)
    
    # Строим исправленный график
    print("Строим исправленный график с ограничениями -24 до +24...")
    create_corrected_plot("test_data2.csv")