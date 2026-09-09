import sys
import os
import pandas as pd
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QSpinBox,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QShortcut,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QKeySequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class DataLoaderThread(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            logging.debug(f"Начало загрузки файла: {self.file_path}")
            df = pd.read_csv(self.file_path)

            if "query" not in df.columns or "text" not in df.columns:
                error_msg = "csv должен содержать колонки query и text"
                logging.error(error_msg)
                self.error.emit(error_msg)
                return

            if "is_similar" not in df.columns:
                df["is_similar"] = -1
            else:
                df["is_similar"] = df["is_similar"].fillna(-1).astype(int)

            logging.info(f"Файл загружен, записей: {len(df)}")
            self.finished.emit(df)

        except Exception as e:
            error_msg = f"Ошибка загрузки: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


class STSAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_loaded = False
        self.df = None
        self.current_index = 0
        self.output_file = "labelled_sts_data.csv"
        self.unsaved_changes = False
        self.query_font_size = 22
        self.text_font_size = 22

        self.setup_ui()
        self.setup_shortcuts()
        self.update_fonts()
        self.update_ui()

        logging.info("Приложение инициализировано")

    def setup_ui(self):
        self.setWindowTitle("Разметка данных Semantic Similarity")
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        left_panel = QWidget()
        left_panel.setMinimumWidth(600)
        left_layout = QVBoxLayout(left_panel)

        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("📁 Загрузить csv")
        self.save_btn = QPushButton("💾 Сохранить")
        self.delete_btn = QPushButton("🗑️ Удалить строку")
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addStretch()
        left_layout.addLayout(control_layout)

        self.file_info = QLabel("Файл не загружен")
        left_layout.addWidget(self.file_info)

        stats_layout = QHBoxLayout()
        self.stats_total = QLabel("Всего: 0")
        self.stats_labeled = QLabel("Размечено: 0")
        self.stats_remaining = QLabel("Осталось: 0")
        stats_layout.addWidget(self.stats_total)
        stats_layout.addWidget(self.stats_labeled)
        stats_layout.addWidget(self.stats_remaining)
        stats_layout.addStretch()
        left_layout.addLayout(stats_layout)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)

        texts_group = QGroupBox("Тексты для сравнения")
        texts_layout = QVBoxLayout(texts_group)

        query_layout = QVBoxLayout()
        query_layout.addWidget(QLabel("Query:"))
        self.query_display = QTextEdit()
        self.query_display.setReadOnly(True)
        self.query_display.setMaximumHeight(180)
        query_layout.addWidget(self.query_display)

        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Text:"))
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(180)
        text_layout.addWidget(self.text_display)

        texts_layout.addLayout(query_layout)
        texts_layout.addLayout(text_layout)
        left_layout.addWidget(texts_group)

        annotation_group = QGroupBox("Разметка похожести (Semantic Similarity)")
        annotation_layout = QVBoxLayout(annotation_group)

        similarity_layout = QHBoxLayout()
        self.btn_similar = QPushButton("✅ Похожи")
        self.btn_not_similar = QPushButton("❌ Не похожи")
        self.btn_skip = QPushButton("⏭️ Пропустить")

        self.btn_similar.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold;"
        )
        self.btn_not_similar.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold;"
        )
        self.btn_skip.setStyleSheet("background-color: #95a5a6; color: white;")

        similarity_layout.addWidget(self.btn_similar)
        similarity_layout.addWidget(self.btn_not_similar)
        similarity_layout.addWidget(self.btn_skip)
        annotation_layout.addLayout(similarity_layout)

        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Текущая разметка:"))
        self.current_label = QLabel("Не размечено")
        self.current_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        current_layout.addWidget(self.current_label)
        current_layout.addStretch()
        annotation_layout.addLayout(current_layout)
        left_layout.addWidget(annotation_group)

        nav_group = QGroupBox("Навигация")
        nav_layout = QVBoxLayout(nav_group)

        self.position_label = QLabel("Запись 0 из 0")
        self.position_label.setStyleSheet("font-weight: bold;")
        nav_layout.addWidget(self.position_label)

        nav_buttons_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⬅️ Назад")
        self.btn_next = QPushButton("Вперед ➡️")
        self.btn_next_unlabeled = QPushButton("➡️ След. неразмеченный")
        nav_buttons_layout.addWidget(self.btn_prev)
        nav_buttons_layout.addWidget(self.btn_next)
        nav_buttons_layout.addWidget(self.btn_next_unlabeled)
        nav_layout.addLayout(nav_buttons_layout)

        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Перейти к:"))
        self.jump_input = QSpinBox()
        self.jump_input.setMinimum(1)
        self.jump_input.setMaximum(1)
        self.jump_btn = QPushButton("Перейти")
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_btn)
        jump_layout.addStretch()
        nav_layout.addLayout(jump_layout)
        left_layout.addWidget(nav_group)

        left_layout.addStretch()

        right_panel = QWidget()
        right_panel.setMinimumWidth(500)
        right_layout = QVBoxLayout(right_panel)

        table_group = QGroupBox("Таблица данных")
        table_layout = QVBoxLayout(table_group)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["Query", "Text", "Similar"])
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSortingEnabled(False)

        table_layout.addWidget(self.data_table)
        right_layout.addWidget(table_group)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        main_layout.addWidget(splitter)

        self.connect_signals()

    def connect_signals(self):
        try:
            self.load_btn.clicked.connect(self.load_data)
            self.save_btn.clicked.connect(self.save_data)
            self.delete_btn.clicked.connect(self.delete_current_record)

            self.btn_similar.clicked.connect(lambda: self.annotate_current(1))
            self.btn_not_similar.clicked.connect(lambda: self.annotate_current(0))
            self.btn_skip.clicked.connect(self.skip_current)

            self.btn_prev.clicked.connect(self.previous_record)
            self.btn_next.clicked.connect(self.next_record)
            self.btn_next_unlabeled.clicked.connect(self.next_unlabeled)

            self.jump_btn.clicked.connect(self.jump_to_record)
            self.data_table.itemSelectionChanged.connect(self.table_selection_changed)
            logging.debug("Сигналы подключены")
        except Exception as e:
            logging.error(f"Ошибка при подключении сигналов: {e}", exc_info=True)

    def setup_shortcuts(self):
        try:
            QShortcut(QKeySequence("Right"), self).activated.connect(self.next_record)
            QShortcut(QKeySequence("Left"), self).activated.connect(
                self.previous_record
            )
            QShortcut(QKeySequence("Space"), self).activated.connect(
                self.next_unlabeled
            )
            QShortcut(QKeySequence("1"), self).activated.connect(
                lambda: self.annotate_current(1)
            )
            QShortcut(QKeySequence("2"), self).activated.connect(
                lambda: self.annotate_current(0)
            )
            QShortcut(QKeySequence("S"), self).activated.connect(self.skip_current)
            QShortcut(QKeySequence("Delete"), self).activated.connect(
                self.delete_current_record
            )
            logging.debug("Горячие клавиши настроены")
        except Exception as e:
            logging.error(f"Ошибка при настройке горячих клавиш: {e}", exc_info=True)

    def update_fonts(self):
        try:
            query_font = QFont("Arial", self.query_font_size)
            self.query_display.setFont(query_font)
            text_font = QFont("Arial", self.text_font_size)
            self.text_display.setFont(text_font)
            app_font = QFont("Arial", 14)
            self.setFont(app_font)
        except Exception as e:
            logging.error(f"Ошибка при обновлении шрифтов: {e}", exc_info=True)

    def load_data(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите csv файл", "", "csv файлы (*.csv)"
            )

            if not file_path:
                return

            self.load_btn.setEnabled(False)
            self.load_btn.setText("Загрузка...")
            QApplication.processEvents()

            self.loader_thread = DataLoaderThread(file_path)
            self.loader_thread.finished.connect(self.on_data_loaded)
            self.loader_thread.error.connect(self.on_load_error)
            self.loader_thread.start()
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            self.load_btn.setEnabled(True)
            self.load_btn.setText("📁 Загрузить csv")

    def on_data_loaded(self, df):
        try:
            self.df = df
            self.current_index = 0
            self.unsaved_changes = False
            self.data_loaded = True

            max_records = len(self.df)
            self.jump_input.setMaximum(max_records)

            self.file_info.setText(f"Загружено: {len(self.df)} записей")

            self.load_btn.setEnabled(True)
            self.load_btn.setText("📁 Загрузить csv")

            self.update_ui()
            logging.debug(f"Данные загружены в интерфейс")
        except Exception as e:
            logging.error(
                f"Ошибка при обработке загруженных данных: {e}", exc_info=True
            )

    def on_load_error(self, error_msg):
        try:
            QMessageBox.critical(self, "Ошибка", error_msg)
            self.df = None
            self.data_loaded = False

            self.load_btn.setEnabled(True)
            self.load_btn.setText("📁 Загрузить csv")
        except Exception as e:
            logging.error(f"Ошибка при обработке: {e}", exc_info=True)

    def update_ui(self):
        try:
            if not self.data_loaded or self.df is None:
                self.clear_ui()
                return

            total = len(self.df)
            labeled = (self.df["is_similar"] != -1).sum()
            remaining = total - labeled

            self.stats_total.setText(f"Всего: {total}")
            self.stats_labeled.setText(f"Размечено: {labeled}")
            self.stats_remaining.setText(f"Осталось: {remaining}")

            progress = int((labeled / total) * 100) if total > 0 else 0
            self.progress.setValue(progress)

            self.display_current_record()
            self.update_buttons_state()

            QTimer.singleShot(10, self.update_table)
        except Exception as e:
            logging.error(f"Ошибка при обновлении UI: {e}", exc_info=True)

    def update_table(self):
        try:
            if not self.data_loaded:
                return

            self.data_table.blockSignals(True)

            try:
                display_rows = min(100, len(self.df))
                self.data_table.setRowCount(display_rows)

                start_idx = max(0, self.current_index - 50)

                for i in range(display_rows):
                    actual_idx = start_idx + i
                    if actual_idx >= len(self.df):
                        break

                    row = self.df.iloc[actual_idx]

                    query_text = str(row["query"])
                    if len(query_text) > 80:
                        query_text = query_text[:80] + "..."

                    text_text = str(row["text"])
                    if len(text_text) > 80:
                        text_text = text_text[:80] + "..."

                    self.data_table.setItem(i, 0, QTableWidgetItem(query_text))
                    self.data_table.setItem(i, 1, QTableWidgetItem(text_text))

                    similar = row["is_similar"]
                    if similar == -1:
                        similar_text = "NoLabel"
                        color = QColor(200, 200, 200)
                    elif similar == 1:
                        similar_text = "Sim"
                        color = QColor(144, 238, 144)
                    else:
                        similar_text = "Diff"
                        color = QColor(255, 182, 193)

                    similar_item = QTableWidgetItem(similar_text)
                    similar_item.setBackground(color)
                    self.data_table.setItem(i, 2, similar_item)

                header = self.data_table.horizontalHeader()
                header.setSectionResizeMode(0, QHeaderView.Stretch)
                header.setSectionResizeMode(1, QHeaderView.Stretch)
                header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

                table_row = self.current_index - start_idx
                if 0 <= table_row < self.data_table.rowCount():
                    self.data_table.selectRow(table_row)

            finally:
                self.data_table.blockSignals(False)
        except Exception as e:
            logging.error(f"Ошибка при обновлении таблицы: {e}", exc_info=True)

    def clear_ui(self):
        try:
            self.query_display.clear()
            self.text_display.clear()
            self.position_label.setText("Запись 0 из 0")
            self.current_label.setText("Не размечено")
            self.data_table.setRowCount(0)

            self.stats_total.setText("Всего: 0")
            self.stats_labeled.setText("Размечено: 0")
            self.stats_remaining.setText("Осталось: 0")
            self.progress.setValue(0)
        except Exception as e:
            logging.error(f"Ошибка при очистке UI: {e}", exc_info=True)

    def display_current_record(self):
        try:
            if not self.data_loaded:
                return

            record = self.df.iloc[self.current_index]

            self.query_display.setText(str(record["query"]))
            self.text_display.setText(str(record["text"]))

            self.position_label.setText(
                f"Запись {self.current_index + 1} из {len(self.df)}"
            )
            self.jump_input.setValue(self.current_index + 1)

            similar = record["is_similar"]
            if similar == -1:
                self.current_label.setText("Не размечено")
                self.current_label.setStyleSheet(
                    "color: gray; font-weight: bold; font-size: 14px;"
                )
            else:
                similar_text = "Похожи" if similar == 1 else "Не похожи"
                self.current_label.setText(similar_text)

                if similar == 1:
                    self.current_label.setStyleSheet(
                        "color: green; font-weight: bold; font-size: 14px;"
                    )
                else:
                    self.current_label.setStyleSheet(
                        "color: red; font-weight: bold; font-size: 14px;"
                    )
        except Exception as e:
            logging.error(f"Ошибка при отображении текущей записи: {e}", exc_info=True)

    def update_buttons_state(self):
        try:
            has_data = self.data_loaded and self.df is not None
            self.save_btn.setEnabled(has_data and self.unsaved_changes)
            self.delete_btn.setEnabled(has_data)
            self.btn_prev.setEnabled(has_data and self.current_index > 0)
            self.btn_next.setEnabled(has_data and self.current_index < len(self.df) - 1)
            self.btn_similar.setEnabled(has_data)
            self.btn_not_similar.setEnabled(has_data)
            self.btn_skip.setEnabled(has_data)
            self.btn_next_unlabeled.setEnabled(has_data)
        except Exception as e:
            logging.error(f"Ошибка при обновлении состояния кнопок: {e}", exc_info=True)

    def annotate_current(self, is_similar):
        try:
            if not self.data_loaded:
                return

            current_idx = self.current_index
            self.df.at[current_idx, "is_similar"] = is_similar
            self.unsaved_changes = True

            logging.debug(f"Запись {current_idx} размечена: similar={is_similar}")

            self.update_buttons_state()
            self.display_current_record()

            if current_idx < len(self.df) - 1:
                self.current_index = current_idx + 1
                QTimer.singleShot(50, self.update_ui)
            else:
                QTimer.singleShot(50, self.update_ui)
        except Exception as e:
            logging.error(f"Ошибка при разметке записи: {e}", exc_info=True)

    def skip_current(self):
        try:
            if not self.data_loaded:
                return

            if self.current_index < len(self.df) - 1:
                self.current_index += 1
                QTimer.singleShot(50, self.update_ui)
        except Exception as e:
            logging.error(f"Ошибка при пропуске записи: {e}", exc_info=True)

    def previous_record(self):
        try:
            if not self.data_loaded:
                return

            if self.current_index > 0:
                self.current_index -= 1
                QTimer.singleShot(50, self.update_ui)
        except Exception as e:
            logging.error(
                f"Ошибка при переходе к предыдущей записи: {e}", exc_info=True
            )

    def next_record(self):
        try:
            if not self.data_loaded:
                return

            if self.current_index < len(self.df) - 1:
                self.current_index += 1
                QTimer.singleShot(50, self.update_ui)
        except Exception as e:
            logging.error(f"Ошибка при переходе к следующей записи: {e}", exc_info=True)

    def next_unlabeled(self):
        try:
            if not self.data_loaded:
                return

            for i in range(self.current_index + 1, len(self.df)):
                if self.df.iloc[i]["is_similar"] == -1:
                    self.current_index = i
                    QTimer.singleShot(50, self.update_ui)
                    return

            for i in range(0, self.current_index):
                if self.df.iloc[i]["is_similar"] == -1:
                    self.current_index = i
                    QTimer.singleShot(50, self.update_ui)
                    return

            QMessageBox.information(self, "Информация", "Все записи размечены!")
        except Exception as e:
            logging.error(
                f"Ошибка при поиске следующей неразмеченной записи: {e}", exc_info=True
            )

    def jump_to_record(self):
        try:
            if not self.data_loaded:
                return

            target_index = self.jump_input.value() - 1

            if 0 <= target_index < len(self.df):
                self.current_index = target_index
                QTimer.singleShot(50, self.update_ui)
        except Exception as e:
            logging.error(f"Ошибка при переходе к записи: {e}", exc_info=True)

    def table_selection_changed(self):
        try:
            if not self.data_loaded or self.data_table.signalsBlocked():
                return

            selected = self.data_table.selectedItems()
            if not selected:
                return

            row = selected[0].row()
            start_idx = max(0, self.current_index - 50)
            real_index = start_idx + row

            if real_index < len(self.df) and real_index != self.current_index:
                self.current_index = real_index
                self.display_current_record()
        except Exception as e:
            logging.error(
                f"Ошибка при изменении выделения в таблице: {e}", exc_info=True
            )

    def delete_current_record(self):
        try:
            if not self.data_loaded:
                return

            current_idx = self.current_index
            record_query = self.df.iloc[current_idx]["query"]
            display_query = (
                record_query[:100] + "..." if len(record_query) > 100 else record_query
            )

            reply = QMessageBox.question(
                self,
                "Подтверждение удаления",
                f"Удалить текущую запись {current_idx + 1}?\n\nQuery: {display_query}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                original_index = current_idx

                self.df = self.df.drop(original_index).reset_index(drop=True)
                self.unsaved_changes = True

                if len(self.df) == 0:
                    self.data_loaded = False
                    self.df = None
                    self.current_index = 0
                else:
                    if original_index >= len(self.df):
                        self.current_index = len(self.df) - 1

                self.jump_input.setMaximum(len(self.df) if self.df is not None else 1)
                self.update_ui()

                logging.info(f"Запись {original_index} удалена")

        except Exception as e:
            logging.error(f"Ошибка при удалении записи: {e}", exc_info=True)

    def save_data(self):
        try:
            if not self.data_loaded:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения")
                return False

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить данные",
                self.output_file,
                "CSV файлы (*.csv);;Все файлы (*)",
            )

            if not file_path:
                return False

            self.df.to_csv(file_path, index=False, encoding="utf-8")
            self.output_file = file_path
            self.unsaved_changes = False
            self.update_buttons_state()
            logging.info(f"Данные сохранены в файл: {file_path}")
            QMessageBox.information(self, "Успех", f"Данные сохранены в {file_path}")
            return True

        except Exception as e:
            error_msg = f"Ошибка сохранения: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Ошибка", error_msg)
            return False

    def closeEvent(self, event):
        try:
            if self.unsaved_changes:
                reply = QMessageBox.question(
                    self,
                    "Несохраненные изменения",
                    "Сохранить изменения перед выходом?",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                )

                if reply == QMessageBox.Save:
                    if self.save_data():
                        event.accept()
                    else:
                        event.ignore()
                elif reply == QMessageBox.Discard:
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()
            logging.info("Приложение закрыто")
        except Exception as e:
            logging.error(f"Ошибка при закрытии приложения: {e}", exc_info=True)
            event.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        window = STSAnnotator()
        window.show()

        logging.debug("Приложение запущено")
        sys.exit(app.exec_())
    except Exception as e:
        logging.critical(
            f"Критическая ошибка при запуске приложения: {e}", exc_info=True
        )