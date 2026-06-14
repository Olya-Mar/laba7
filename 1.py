import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Настройка шрифтов
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title('Лифтовый дашборд (Вариант №45)')
        root.geometry('1100x750')
        root.configure(bg='#000000')
        root.resizable(True, True)

        self.df_raw = None
        self.df_work = None
        self.current_chart = "line"

        self.fig = plt.figure(figsize=(9, 5), dpi=100, facecolor='#0a0a0a')
        self.canvas = None

        # Переменные для фильтров
        self.lift_var = tk.StringVar(value="Все")
        self.risk_var = tk.StringVar(value="Все")
        self.time_var = tk.StringVar(value="Все")

        self.create_widgets()
        self.load_data()

    # ============================================================
    # ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
    # ============================================================

    def load_data(self):
        try:
            self.df_raw = pd.read_csv('data.csv')
            self.df_raw.columns = ['ts', 'elev_id', 'floors', 'door_t', 'vib', 'load']

            # Временные признаки
            self.df_raw['date'] = pd.to_datetime(self.df_raw['ts'], unit='s')
            self.df_raw['weekday'] = self.df_raw['date'].dt.dayofweek
            self.df_raw['hour'] = self.df_raw['date'].dt.hour

            # Категория риска (на основе вибрации)
            risk = []
            for v in self.df_raw['vib']:
                if v <= 2:
                    risk.append('Низкий')
                elif v <= 5:
                    risk.append('Средний')
                else:
                    risk.append('Высокий')
            self.df_raw['risk_category'] = risk

            # Время суток
            tod = []
            for h in self.df_raw['hour']:
                if 6 <= h < 12:
                    tod.append('Утро')
                elif 12 <= h < 18:
                    tod.append('День')
                elif 18 <= h < 22:
                    tod.append('Вечер')
                else:
                    tod.append('Ночь')
            self.df_raw['time_of_day'] = tod

            # Категория загрузки
            load_cat = []
            for l in self.df_raw['load']:
                if l < 30:
                    load_cat.append('Низкая')
                elif l < 70:
                    load_cat.append('Средняя')
                else:
                    load_cat.append('Высокая')
            self.df_raw['load_category'] = load_cat

            # Очистка данных
            self.df_raw['floors'] = np.where(self.df_raw['floors'] < 0, 0, self.df_raw['floors'])
            self.df_raw['door_t'] = np.clip(self.df_raw['door_t'], 0.5, None)
            self.df_raw['load'] = np.clip(self.df_raw['load'], 0, 100)

            # Замена NaN
            for col in ['door_t', 'vib', 'load']:
                if self.df_raw[col].isna().any():
                    self.df_raw[col] = self.df_raw[col].fillna(self.df_raw[col].median())

            self.df_work = self.df_raw.copy()
            self.update_status(f"Загружено записей: {len(self.df_raw):,} | Лифтов: {len(self.df_raw['elev_id'].unique())}")
            self.update_filters()
            self.plot_line()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить data.csv\n{str(e)}")

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def update_filters(self):
        if self.df_raw is not None:
            self.lift_combo['values'] = ["Все"] + sorted(self.df_raw['elev_id'].unique().tolist())
            self.risk_combo['values'] = ["Все"] + sorted(self.df_raw['risk_category'].unique().tolist())
            self.time_combo['values'] = ["Все", "Утро", "День", "Вечер", "Ночь"]

    # ============================================================
    # ФИЛЬТРАЦИЯ ДАННЫХ
    # ============================================================

    def preprocess_data(self):
        self.df_work = self.df_raw.copy()
        if self.lift_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['elev_id'] == int(self.lift_var.get())]
        if self.risk_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['risk_category'] == self.risk_var.get()]
        if self.time_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['time_of_day'] == self.time_var.get()]
        return self.df_work

    def apply_filters(self):
        self.refresh_data()

    # ============================================================
    # ФУНКЦИИ ОТРИСОВКИ ГРАФИКОВ
    # ============================================================

    def clear_figure(self):
        self.fig.clear()

    def plot_line(self):
        """Линейный график: динамика вибрации"""
        self.current_chart = "line"
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            grouped = data.groupby('date')['vib'].mean().reset_index()
            ax.plot(grouped['date'], grouped['vib'], marker='o', linewidth=2,
                    markersize=4, color='#00FFFF', markerfacecolor='#00FFFF')
            ax.set_xlabel('Дата', color='#c0c0c0')
            ax.set_ylabel('Средняя вибрация (мм/с)', color='#c0c0c0')
            ax.set_title('Динамика вибрации лифтов', color='#00FFFF')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='#c0c0c0')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, color='#c0c0c0')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_bar(self):
        """Столбчатая диаграмма: среднее время дверей по категориям риска"""
        self.current_chart = "bar"
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            grouped = data.groupby('risk_category')['door_t'].mean().reset_index()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            bars = ax.bar(grouped['risk_category'], grouped['door_t'], color=colors, edgecolor='#00FFFF')

            for bar, val in zip(bars, grouped['door_t']):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='#c0c0c0')

            ax.set_xlabel('Категория риска', color='#c0c0c0')
            ax.set_ylabel('Среднее время дверей (с)', color='#c0c0c0')
            ax.set_title('Среднее время открытия дверей по категориям риска', color='#00FFFF')
            ax.tick_params(colors='#c0c0c0')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, color='#c0c0c0')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_scatter(self):
        """Точечная диаграмма: зависимость вибрации от загрузки"""
        self.current_chart = "scatter"
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            scatter = ax.scatter(data['load'], data['vib'], c=data['door_t'],
                                 cmap='plasma', alpha=0.6, s=30)
            ax.set_xlabel('Загрузка (%)', color='#c0c0c0')
            ax.set_ylabel('Вибрация (мм/с)', color='#c0c0c0')
            ax.set_title('Зависимость вибрации от загрузки', color='#00FFFF')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Время дверей (с)', color='#c0c0c0')
            cbar.ax.yaxis.set_tick_params(color='#c0c0c0')
            plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#c0c0c0')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='#c0c0c0')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, color='#c0c0c0')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def plot_heatmap(self):
        """Тепловая карта: корреляция признаков"""
        self.current_chart = "heatmap"
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            numeric_cols = ['door_t', 'vib', 'load', 'floors', 'hour', 'weekday']
            corr = data[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', ax=ax, square=True)
            ax.set_title('Матрица корреляции признаков', color='#00FFFF')
            ax.tick_params(colors='#c0c0c0')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#c0c0c0')
            plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#c0c0c0')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, color='#c0c0c0')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ============================================================
    # ЭКСПОРТ И ОБНОВЛЕНИЕ
    # ============================================================

    def export_plot(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                  filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Экспорт", f"График сохранён: {filepath}")

    def refresh_data(self):
        if self.current_chart == "line":
            self.plot_line()
        elif self.current_chart == "bar":
            self.plot_bar()
        elif self.current_chart == "scatter":
            self.plot_scatter()
        elif self.current_chart == "heatmap":
            self.plot_heatmap()

    # ============================================================
    # ИНТЕРФЕЙС TKINTER
    # ============================================================

    def create_widgets(self):
        main = tk.Frame(self.root, bg='#000000')
        main.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        # Заголовок
        tk.Label(main, text='ЛИФТОВЫЙ ДАШБОРД (Вариант №45)',
                 font=('Courier New', 14, 'bold'), fg='#00FFFF', bg='#000000').pack(pady=10)

        # === ПАНЕЛЬ ТИПА ГРАФИКА ===
        btn_frame = tk.LabelFrame(main, text='Тип графика', font=('Courier New', 9),
                                  fg='#00FFFF', bg='#0a0a0a')
        btn_frame.pack(fill=tk.X, pady=5)

        inner = tk.Frame(btn_frame, bg='#0a0a0a')
        inner.pack(pady=8)

        def set_line():
            self.current_chart = "line"
            self.plot_line()

        def set_bar():
            self.current_chart = "bar"
            self.plot_bar()

        def set_scatter():
            self.current_chart = "scatter"
            self.plot_scatter()

        def set_heatmap():
            self.current_chart = "heatmap"
            self.plot_heatmap()

        btn_style = {'font': ('Courier New', 9), 'bg': '#1a1a1a', 'fg': '#c0c0c0',
                     'activebackground': '#00FFFF', 'activeforeground': '#000000',
                     'relief': tk.RAISED, 'bd': 1}

        tk.Button(inner, text='Линейный', command=set_line, width=12, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Столбчатый', command=set_bar, width=12, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Точечный', command=set_scatter, width=12, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Тепловая карта', command=set_heatmap, width=14, **btn_style).pack(side=tk.LEFT, padx=4)

        tk.Button(inner, text='Экспорт', command=self.export_plot, width=10, **btn_style).pack(side=tk.RIGHT, padx=4)
        tk.Button(inner, text='Обновить', command=self.refresh_data, width=10, **btn_style).pack(side=tk.RIGHT, padx=4)

        # === ПАНЕЛЬ ФИЛЬТРОВ ===
        filter_frame = tk.LabelFrame(main, text='Фильтры', font=('Courier New', 9),
                                     fg='#00FFFF', bg='#0a0a0a')
        filter_frame.pack(fill=tk.X, pady=5)

        filter_inner = tk.Frame(filter_frame, bg='#0a0a0a')
        filter_inner.pack(pady=8, padx=10)

        label_style = {'font': ('Courier New', 9), 'fg': '#c0c0c0', 'bg': '#0a0a0a'}

        tk.Label(filter_inner, text='Лифт:', **label_style).pack(side=tk.LEFT, padx=5)
        self.lift_combo = ttk.Combobox(filter_inner, textvariable=self.lift_var, width=10, state="readonly")
        self.lift_combo.pack(side=tk.LEFT, padx=5)
        self.lift_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_filters())

        tk.Label(filter_inner, text='Риск:', **label_style).pack(side=tk.LEFT, padx=5)
        self.risk_combo = ttk.Combobox(filter_inner, textvariable=self.risk_var, width=12, state="readonly")
        self.risk_combo.pack(side=tk.LEFT, padx=5)
        self.risk_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_filters())

        tk.Label(filter_inner, text='Время суток:', **label_style).pack(side=tk.LEFT, padx=5)
        self.time_combo = ttk.Combobox(filter_inner, textvariable=self.time_var, width=12, state="readonly")
        self.time_combo.pack(side=tk.LEFT, padx=5)
        self.time_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_filters())

        tk.Button(filter_inner, text='Применить', command=self.apply_filters, width=10,
                  font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0',
                  activebackground='#00FFFF', activeforeground='#000000').pack(side=tk.RIGHT, padx=5)

        # === ОБЛАСТЬ ГРАФИКА ===
        plot_frame = tk.LabelFrame(main, text='График', font=('Courier New', 9),
                                   fg='#00FFFF', bg='#0a0a0a')
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === СТРОКА СТАТУСА ===
        status_frame = tk.Frame(main, bg='#000000')
        status_frame.pack(fill=tk.X, pady=5)

        self.status_label = tk.Label(status_frame, text='Загрузка...',
                                     font=('Courier New', 8), fg='#00FFFF', bg='#000000')
        self.status_label.pack(side=tk.LEFT)


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == '__main__':
    main()