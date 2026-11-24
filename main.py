import os
import atexit
import shutil
import tempfile
import pandas as pd
import numpy as np
import re
from flask import Flask, request, render_template, session, send_file
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
from io import BytesIO

# =============================================================================
# Инициализация приложения
# =============================================================================
app = Flask(__name__)

# Временная директория для загружаемых файлов — существует только на время
# работы приложения и автоматически удаляется при завершении процесса.
TEMP_DIR = tempfile.mkdtemp(prefix='ytgraph_')
app.config['UPLOAD_FOLDER'] = TEMP_DIR


def _cleanup_temp_dir():
    """Удаляет временную директорию с загруженными файлами."""
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


atexit.register(_cleanup_temp_dir)

app.secret_key = 'your-very-secret-key-change-in-production'


def cleanup_uploaded_files(*filepaths):
    """Немедленно удаляет указанные временные файлы после обработки."""
    for fp in filepaths:
        try:
            if fp and os.path.exists(fp):
                os.remove(fp)
        except OSError:
            pass

# =============================================================================
# Цветовая палитра (30 цветов)
# =============================================================================
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#ad494a',
    '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd',
    '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#e7f482',
    '#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'
]


# =============================================================================
# Безопасное вычисление функций
# =============================================================================
def safe_eval(expr, variables):
    """Безопасное вычисление математических выражений"""
    try:
        # Создаём безопасное окружение
        safe_dict = {
            "__builtins__": {},
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": abs,
            "pi": np.pi,
            "e": np.e,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
        }

        # Добавляем переменные пользователя
        safe_dict.update(variables)

        # Заменяем ^ на **
        expr = expr.replace('^', '**')

        # Вычисляем
        result = eval(expr, {"__builtins__": {}}, safe_dict)

        # Преобразуем в numpy массив
        result = np.array(result, dtype=np.float64)

        # Заменяем inf и nan на np.nan
        result = np.where(np.isinf(result) | np.isnan(result), np.nan, result)

        return result

    except Exception as e:
        raise ValueError(f"Ошибка в формуле: {e}")


# =============================================================================
# Автоопределение типа функции
# =============================================================================
def detect_function_type(expr1, expr2=''):
    """Автоматически определяет тип функции по выражению"""
    expr1_lower = expr1.lower() if expr1 else ''
    expr2_lower = expr2.lower() if expr2 else ''
    expr_combined = expr1_lower + ' ' + expr2_lower

    # Полярная: содержит theta или θ
    if re.search(r'\b(theta|θ)\b', expr_combined):
        return 'polar'

    # Параметрическая: используется t в выражениях
    if re.search(r'\bt\b', expr1_lower) and re.search(r'\bt\b', expr_combined):
        return 'parametric'

    # Проверка на неявную: содержит и x, и y в одном выражении
    has_x = bool(re.search(r'\bx\b', expr1_lower))
    has_y = bool(re.search(r'\by\b', expr1_lower))

    if has_x and has_y:
        return 'implicit'

    # Проверка на x = f(y): содержит y, но не содержит x
    if has_y and not has_x:
        return 'x_y'

    # По умолчанию: y = f(x)
    return 'y_x'


# =============================================================================
# Построение функций разных типов
# =============================================================================
def plot_function_type(func_type, expr1, expr2, x_range, y_range, t_range, color, name):
    """Построение функций разных типов"""
    try:
        if func_type == 'y_x':
            # y = f(x)
            x_vals = np.linspace(x_range[0], x_range[1], 1000)
            y_vals = safe_eval(expr1, {'x': x_vals})
            return go.Scatter(x=x_vals, y=y_vals, mode='lines', name=name,
                              line=dict(color=color, width=2))

        elif func_type == 'x_y':
            # x = f(y)
            y_vals = np.linspace(y_range[0], y_range[1], 1000)
            x_vals = safe_eval(expr1, {'y': y_vals})
            return go.Scatter(x=x_vals, y=y_vals, mode='lines', name=name,
                              line=dict(color=color, width=2))

        elif func_type == 'parametric':
            # x = f(t), y = g(t)
            t_vals = np.linspace(t_range[0], t_range[1], 1000)
            x_vals = safe_eval(expr1, {'t': t_vals})
            y_vals = safe_eval(expr2, {'t': t_vals}) if expr2 else np.zeros_like(t_vals)
            return go.Scatter(x=x_vals, y=y_vals, mode='lines', name=name,
                              line=dict(color=color, width=2))

        elif func_type == 'polar':
            # r = f(θ)
            theta_vals = np.linspace(0, 2 * np.pi, 1000)
            r_vals = safe_eval(expr1, {'theta': theta_vals})
            x_vals = r_vals * np.cos(theta_vals)
            y_vals = r_vals * np.sin(theta_vals)
            return go.Scatter(x=x_vals, y=y_vals, mode='lines', name=name,
                              line=dict(color=color, width=2))

        elif func_type == 'implicit':
            # f(x,y) = 0
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            y_vals = np.linspace(y_range[0], y_range[1], 500)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = safe_eval(expr1, {'x': X, 'y': Y})
            return go.Contour(x=x_vals, y=y_vals, z=Z, contours=dict(start=0, end=0, size=0.001),
                              colorscale=[[0, color], [1, color]], name=name,
                              line=dict(width=2))

        else:
            raise ValueError(f"Неизвестный тип функции: {func_type}")

    except Exception as e:
        raise ValueError(f"Ошибка построения {name}: {str(e)}")


# =============================================================================
# Автомасштабирование графика
# =============================================================================
def auto_scale(traces, x_range, y_range):
    """Автоматическая настройка масштаба графика"""
    all_x = []
    all_y = []

    for trace in traces:
        if hasattr(trace, 'x') and trace.x is not None:
            vals = np.array(trace.x)
            all_x.extend(vals[np.isfinite(vals)])
        if hasattr(trace, 'y') and trace.y is not None:
            vals = np.array(trace.y)
            all_y.extend(vals[np.isfinite(vals)])

    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
        x_range = [x_min - x_margin, x_max + x_margin]

    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
        y_range = [y_min - y_margin, y_max + y_margin]

    return x_range, y_range


# =============================================================================
# Функции аппроксимации
# =============================================================================
def calculate_approximation(x_vals, y_vals, degree=1, approx_type='polynomial'):
    """Вычисляет аппроксимацию данных"""
    try:
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            raise ValueError("Недостаточно данных для аппроксимации")

        if approx_type == 'logarithmic':
            if np.any(x <= 0):
                x_shifted = x - np.min(x) + 1e-10
            else:
                x_shifted = x

            log_x = np.log(x_shifted)
            A = np.vstack([log_x, np.ones(len(log_x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]

            x_dense = np.linspace(min(x), max(x), 500)
            x_dense_shifted = x_dense - np.min(x) + 1e-10 if np.any(x <= 0) else x_dense
            y_approx = a * np.log(x_dense_shifted) + b

            return x_dense, y_approx, f"y = {a:.3f}·ln(x) + {b:.3f}"

        elif approx_type == 'polynomial':
            if len(x) < degree + 1:
                degree = max(1, len(x) - 1)

            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)

            x_dense = np.linspace(min(x), max(x), 500)
            y_approx = poly(x_dense)

            return x_dense, y_approx, f"Полином {degree}°"

        else:
            raise ValueError(f"Неизвестный тип аппроксимации: {approx_type}")

    except Exception as e:
        raise ValueError(f"Ошибка аппроксимации: {str(e)}")


# =============================================================================
# Примеры функций для галереи
# =============================================================================
def get_function_examples():
    """Возвращает словарь с примерами всех типов функций"""
    return {
        'y_x': [
            {'expr': 'sin(x)', 'desc': 'Синусоида'},
            {'expr': 'x**2', 'desc': 'Парабола'},
            {'expr': 'exp(-x)*sin(x)', 'desc': 'Затухающие колебания'},
            {'expr': 'sqrt(abs(x))', 'desc': 'Корень модуля'},
            {'expr': 'x**3 - x', 'desc': 'Кубическая'},
        ],
        'x_y': [
            {'expr': 'y**2', 'desc': 'Парабола →'},
            {'expr': 'sin(y)', 'desc': 'Синус (повёрнут)'},
            {'expr': 'sqrt(abs(y))', 'desc': 'Корень по Y'},
            {'expr': 'y**3 - y', 'desc': 'Кубическая'},
        ],
        'parametric': [
            {'expr1': 'cos(t)', 'expr2': 'sin(t)', 'desc': 'Окружность'},
            {'expr1': 't*cos(t)', 'expr2': 't*sin(t)', 'desc': 'Спираль'},
            {'expr1': 'cos(3*t)', 'expr2': 'sin(2*t)', 'desc': 'Лиссажу'},
            {'expr1': 't - sin(t)', 'expr2': '1 - cos(t)', 'desc': 'Циклоида'},
        ],
        'polar': [
            {'expr': '1', 'desc': 'Окружность r=1'},
            {'expr': 'theta', 'desc': 'Спираль Архимеда'},
            {'expr': 'sin(3*theta)', 'desc': 'Трёхлепестковая роза'},
            {'expr': '1 + cos(theta)', 'desc': 'Кардиоида'},
            {'expr': 'exp(cos(theta)) - 2*cos(4*theta) + sin(theta/12)**5', 'desc': 'Бабочка'},
        ],
        'implicit': [
            {'expr': 'x**2 + y**2 - 1', 'desc': 'Окружность'},
            {'expr': 'x**2/4 + y**2 - 1', 'desc': 'Эллипс'},
            {'expr': 'y**2 - x**3 + x', 'desc': 'Эллиптическая'},
            {'expr': 'sin(x) - cos(y)', 'desc': 'sin(x)=cos(y)'},
            {'expr': '(x**2 + y**2 - 1)**3 - x**2*y**3', 'desc': 'Сердце ❤️'},
        ]
    }


# =============================================================================
# Маршруты приложения
# =============================================================================

@app.route('/', methods=['GET', 'POST'])
def index_page():
    plot_html = None
    error = None
    files_count = 0

    if request.method == 'POST':
        files = []
        connect_lines = []
        approx_types = []
        approx_degrees = []

        for i in range(1, 21):
            f = request.files.get(f'file{i}')
            if f and f.filename.endswith('.xlsx'):
                files.append(f)
                files_count += 1
                connect_lines.append(request.form.get(f'connect{i}') == 'on')
                approx_types.append(request.form.get(f'approx_type{i}', 'none'))
                deg = request.form.get(f'approx_degree{i}', '').strip()
                approx_degrees.append(int(deg) if deg else 1)
            elif f:
                error = "Все файлы должны иметь расширение .xlsx"
                break

        if not error:
            if not files:
                error = "Загрузите хотя бы один .xlsx файл"
            else:
                fig = go.Figure()
                success_count = 0
                x_axis_title = "X"
                y_axis_title = "Y"
                titles_set = False

                for idx, file in enumerate(files):
                    if idx >= 20:
                        break
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)

                        df = pd.read_excel(filepath, header=0)
                        # Данные уже в памяти — временный файл больше не нужен
                        cleanup_uploaded_files(filepath)

                        if df.shape[1] < 2:
                            error = f"Файл {filename}: нужно ≥2 столбцов"
                            continue

                        col_names = df.columns.tolist()
                        x_name = str(col_names[0]).strip()
                        y_name = str(col_names[1]).strip()

                        if "Unnamed" in x_name or x_name == "nan":
                            x_name = ""
                        if "Unnamed" in y_name or y_name == "nan":
                            y_name = ""

                        df_clean = df.dropna(subset=[df.columns[0], df.columns[1]])
                        if df_clean.empty:
                            continue

                        if not titles_set:
                            x_axis_title = x_name if x_name else "X"
                            y_axis_title = y_name if y_name else "Y"
                            titles_set = True

                        mode = 'lines+markers' if connect_lines[idx] else 'markers'
                        color_idx = idx % len(COLOR_PALETTE)
                        fig.add_trace(go.Scatter(
                            x=df_clean[df.columns[0]],
                            y=df_clean[df.columns[1]],
                            mode=mode,
                            name=filename,
                            line=dict(color=COLOR_PALETTE[color_idx]),
                            marker=dict(color=COLOR_PALETTE[color_idx], size=6)
                        ))

                        approx_type = approx_types[idx]
                        if approx_type != 'none':
                            try:
                                degree = approx_degrees[idx] if approx_type == 'polynomial' else 1
                                x_dense, y_approx, equation = calculate_approximation(
                                    df_clean[df.columns[0]],
                                    df_clean[df.columns[1]],
                                    degree=degree,
                                    approx_type=approx_type
                                )
                                label = "Лог." if approx_type == 'logarithmic' else f"Полином {degree}°"
                                fig.add_trace(go.Scatter(
                                    x=x_dense,
                                    y=y_approx,
                                    mode='lines',
                                    name=f"{label}: {filename}",
                                    line=dict(color=COLOR_PALETTE[color_idx], width=2, dash='dash'),
                                    opacity=0.7
                                ))
                            except Exception as approx_err:
                                error = f"⚠️ Аппроксимация для {filename}: {str(approx_err)}"

                        success_count += 1

                    except Exception as e:
                        error = f"Ошибка при обработке {file.filename}: {str(e)}"
                        continue

                if success_count == 0:
                    error = error or "Не удалось обработать ни один файл"
                else:
                    user_title = request.form.get('plot_title', '').strip()
                    if not user_title:
                        user_title = "График по загруженным файлам"

                    fig.update_layout(
                        title=user_title,
                        xaxis_title=x_axis_title,
                        yaxis_title=y_axis_title,
                        hovermode='x unified',
                        template='plotly_white',
                        legend_title="Файлы"
                    )
                    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('index.html', plot_html=plot_html, error=error, files_count=files_count)


@app.route('/analytical', methods=['GET', 'POST'])
def analytical():
    plot_html = None
    error = None
    files_count = 0
    functions_count = 0

    if request.method == 'GET':
        return render_template('analytical.html', plot_html=plot_html, error=error,
                               files_count=files_count, functions_count=functions_count,
                               color_palette=COLOR_PALETTE, function_examples=get_function_examples())

    files = []
    approx_types = []
    approx_degrees = []

    for i in range(1, 11):
        f = request.files.get(f'file{i}')
        if f and f.filename.endswith('.xlsx'):
            files.append(f)
            files_count += 1
            approx_types.append(request.form.get(f'approx_type{i}', 'none'))
            deg = request.form.get(f'approx_degree{i}', '').strip()
            approx_degrees.append(int(deg) if deg else 1)
        elif f:
            error = "Все файлы должны быть .xlsx"
            break

    functions = []
    func_expr2_list = []
    func_colors = []

    for i in range(1, 11):
        func = request.form.get(f'func{i}', '').strip()
        expr2 = request.form.get(f'func_expr2_{i}', '').strip()
        fcolor = request.form.get(f'func_color{i}', COLOR_PALETTE[(i - 1) % len(COLOR_PALETTE)])
        if func:
            functions.append(func)
            func_expr2_list.append(expr2)
            func_colors.append(fcolor)

    functions_count = len(functions)

    if not files and not functions:
        error = "Загрузите файлы или введите хотя бы одну функцию"

    if not error:
        fig = go.Figure()
        traces = []
        x_axis_title = "X"
        y_axis_title = "Y"
        titles_set = False
        all_x = []

        for idx, file in enumerate(files):
            if idx >= 10:
                break
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                df = pd.read_excel(filepath, header=0)
                # Данные уже в памяти — временный файл больше не нужен
                cleanup_uploaded_files(filepath)

                if df.shape[1] < 2:
                    error = f"Файл {filename}: нужно ≥2 столбцов"
                    continue

                col_names = df.columns.tolist()
                x_name = str(col_names[0]).strip()
                y_name = str(col_names[1]).strip()

                if "Unnamed" in x_name or x_name == "nan":
                    x_name = ""
                if "Unnamed" in y_name or y_name == "nan":
                    y_name = ""

                df_clean = df.dropna(subset=[df.columns[0], df.columns[1]])
                if df_clean.empty:
                    continue

                if not titles_set:
                    x_axis_title = x_name if x_name else "X"
                    y_axis_title = y_name if y_name else "Y"
                    titles_set = True

                all_x.extend(df_clean[df.columns[0]].tolist())
                color_idx = idx % len(COLOR_PALETTE)

                traces.append(go.Scatter(
                    x=df_clean[df.columns[0]],
                    y=df_clean[df.columns[1]],
                    mode='markers',
                    name=f"📁 Файл {idx + 1}: {filename}",
                    marker=dict(color=COLOR_PALETTE[color_idx], size=6)
                ))

                approx_type = approx_types[idx]
                if approx_type != 'none':
                    try:
                        degree = approx_degrees[idx] if approx_type == 'polynomial' else 1
                        x_dense, y_approx, equation = calculate_approximation(
                            df_clean[df.columns[0]],
                            df_clean[df.columns[1]],
                            degree=degree,
                            approx_type=approx_type
                        )
                        label = "📈 Лог." if approx_type == 'logarithmic' else f"📈 Полином {degree}°"
                        traces.append(go.Scatter(
                            x=x_dense,
                            y=y_approx,
                            mode='lines',
                            name=f"{label}: {filename}",
                            line=dict(color=COLOR_PALETTE[color_idx], width=2, dash='dash'),
                            opacity=0.8
                        ))
                    except Exception as approx_err:
                        error = f"⚠️ Аппроксимация для {filename}: {str(approx_err)}"

            except Exception as e:
                error = f"Ошибка файла {file.filename}: {str(e)}"
                continue

        x_range = [float(request.form.get('x_min', -10)), float(request.form.get('x_max', 10))]
        y_range = [float(request.form.get('y_min', -10)), float(request.form.get('y_max', 10))]
        t_range = [float(request.form.get('t_min', 0)), float(request.form.get('t_max', 6.28))]

        for idx, (expr1, expr2, fcolor) in enumerate(zip(functions, func_expr2_list, func_colors)):
            if idx >= 10:
                break

            func_type = detect_function_type(expr1, expr2)

            try:
                # 🔥 Показываем саму функцию в легенде (обрезаем если длинная)
                func_name = expr1[:30] + ('...' if len(expr1) > 30 else '')
                trace = plot_function_type(
                    func_type, expr1, expr2, x_range, y_range, t_range,
                    fcolor, func_name
                )
                traces.append(trace)
            except Exception as e:
                error = f"⚠️ Функция {idx + 1} ({func_type}): {str(e)}"
                continue

        if traces:
            x_range, y_range = auto_scale(traces, x_range, y_range)
            for trace in traces:
                fig.add_trace(trace)

            user_title = request.form.get('plot_title', '').strip()
            if not user_title:
                user_title = "Аналитический анализ"

            fig.update_layout(
                title=user_title,
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                hovermode='x unified',
                template='plotly_white',
                legend_title="Данные и функции",
                showlegend=True,
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range)
            )
            plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('analytical.html', plot_html=plot_html, error=error,
                           files_count=files_count, functions_count=functions_count,
                           color_palette=COLOR_PALETTE, function_examples=get_function_examples())


@app.route('/functions_only', methods=['GET', 'POST'])
def functions_only():
    plot_html = None
    error = None
    functions_count = 0

    if request.method == 'GET':
        return render_template('functions_only.html', plot_html=plot_html, error=error,
                               functions_count=functions_count, color_palette=COLOR_PALETTE,
                               function_examples=get_function_examples())

    if request.method == 'POST':
        functions = []
        func_expr2_list = []
        func_colors = []

        for i in range(1, 51):
            func = request.form.get(f'func{i}', '').strip()
            expr2 = request.form.get(f'func_expr2_{i}', '').strip()
            fcolor = request.form.get(f'func_color{i}', COLOR_PALETTE[(i - 1) % len(COLOR_PALETTE)])
            if func:
                functions.append(func)
                func_expr2_list.append(expr2)
                func_colors.append(fcolor)

        if not functions:
            error = "Введите хотя бы одну функцию"
        else:
            functions_count = len(functions)
            fig = go.Figure()
            traces = []

            x_min = float(request.form.get('x_min', -10))
            x_max = float(request.form.get('x_max', 10))
            y_min = float(request.form.get('y_min', -10))
            y_max = float(request.form.get('y_max', 10))
            t_min = float(request.form.get('t_min', 0))
            t_max = float(request.form.get('t_max', 6.28))

            x_range = [x_min, x_max]
            y_range = [y_min, y_max]
            t_range = [t_min, t_max]

            for idx, (expr1, expr2, fcolor) in enumerate(zip(functions, func_expr2_list, func_colors)):
                func_type = detect_function_type(expr1, expr2)

                try:
                    # 🔥 Показываем саму функцию в легенде (обрезаем если длинная)
                    func_name = expr1[:30] + ('...' if len(expr1) > 30 else '')
                    trace = plot_function_type(
                        func_type, expr1, expr2, x_range, y_range, t_range,
                        fcolor, func_name
                    )
                    traces.append(trace)
                except Exception as e:
                    error = f"⚠️ Функция {idx + 1}: {str(e)}"
                    continue

            if traces:
                x_range, y_range = auto_scale(traces, x_range, y_range)
                for trace in traces:
                    fig.add_trace(trace)

                user_title = request.form.get('plot_title', '').strip()
                if not user_title:
                    user_title = "График функций"

                fig.update_layout(
                    title=user_title,
                    xaxis_title="X",
                    yaxis_title="Y",
                    hovermode='x unified',
                    template='plotly_white',
                    legend_title="Функции",
                    showlegend=True,
                    xaxis=dict(range=x_range),
                    yaxis=dict(range=y_range),
                    width=1200,
                    height=800
                )
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('functions_only.html', plot_html=plot_html, error=error,
                           functions_count=functions_count, color_palette=COLOR_PALETTE,
                           function_examples=get_function_examples())


@app.route('/instruktion')
def instruktion():
    return render_template('instruktion.html')


@app.route('/manual', methods=['GET', 'POST'])
def manual_input():
    plot_html = None
    error = None

    if request.method == 'POST':
        x_title = request.form.get('x_title', '').strip() or "X"
        y_title = request.form.get('y_title', '').strip() or "Y"
        points_text = request.form.get('points', '').strip()
        connect_line = request.form.get('connect_line') == 'on'
        approx_type = request.form.get('approx_type', 'none')
        approx_degree = request.form.get('approx_degree', '').strip()
        approx_degree = int(approx_degree) if approx_degree else 1
        plot_title = request.form.get('plot_title', '').strip() or "График по введённым данным"

        if not points_text:
            error = "Пожалуйста, введите хотя бы одну точку в формате 'x, y'."
        else:
            x_vals = []
            y_vals = []
            lines = points_text.strip().splitlines()
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) != 2:
                        raise ValueError("Должно быть ровно два значения, разделённых запятой")
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    x_vals.append(x)
                    y_vals.append(y)
                except Exception as e:
                    error = f"Ошибка в строке {i}: '{line}' — {str(e)}"
                    break

            if not error and len(x_vals) == 0:
                error = "Не удалось извлечь ни одной корректной точки."

            if not error:
                session['manual_data'] = {
                    'x_title': x_title,
                    'y_title': y_title,
                    'x_vals': x_vals,
                    'y_vals': y_vals,
                    'plot_title': plot_title
                }

                fig = go.Figure()
                mode = 'lines+markers' if connect_line else 'markers'
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode=mode,
                    name="Введённые данные",
                    line=dict(color='blue'),
                    marker=dict(color='blue', size=6)
                ))

                if approx_type != 'none' and len(x_vals) >= 2:
                    try:
                        x_dense, y_approx, equation = calculate_approximation(
                            x_vals, y_vals,
                            degree=approx_degree if approx_type == 'polynomial' else 1,
                            approx_type=approx_type
                        )
                        label = "Логарифмическая" if approx_type == 'logarithmic' else f"Полином {approx_degree}°"
                        fig.add_trace(go.Scatter(
                            x=x_dense,
                            y=y_approx,
                            mode='lines',
                            name=f"📈 {label}",
                            line=dict(color='red', width=2, dash='dash'),
                            opacity=0.7
                        ))
                    except Exception as approx_err:
                        error = f"⚠️ Ошибка аппроксимации: {str(approx_err)}"

                fig.update_layout(
                    title=plot_title,
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    hovermode='x unified',
                    template='plotly_white',
                    legend_title="Данные"
                )
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('manual.html', plot_html=plot_html, error=error)


@app.route('/download_manual_data')
def download_manual_data():
    data = session.get('manual_data')
    if not data:
        return "Нет данных для скачивания. Сначала введите данные на странице /manual.", 400

    df = pd.DataFrame({
        data['x_title']: data['x_vals'],
        data['y_title']: data['y_vals']
    })

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Данные')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='manual_graph_data.xlsx'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)