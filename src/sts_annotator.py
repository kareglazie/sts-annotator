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
            df = pd.read_csv(self.file_path, skipinitialspace=True)

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
        self.current_file_path = None
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
        self.setGeometry(80, 60, 1100, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("📁 Загрузить csv")
        self.save_as_btn = QPushButton("📝 Сохранить как")
        self.save_btn = QPushButton("💾 Сохранить")
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.save_as_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.file_info = QLabel("Файл не загружен")
        main_layout.addWidget(self.file_info)

        stats_layout = QHBoxLayout()
        self.stats_total = QLabel("Всего: 0")
        self.stats_labeled = QLabel("Размечено: 0")
        self.stats_remaining = QLabel("Осталось: 0")
        stats_layout.addWidget(self.stats_total)
        stats_layout.addWidget(self.stats_labeled)
        stats_layout.addWidget(self.stats_remaining)
        stats_layout.addStretch()
        main_layout.addLayout(stats_layout)

        self.progress = QProgressBar()
        main_layout.addWidget(self.progress)

        nav_layout = QHBoxLayout()

        self.position_label = QLabel("Запись 0 из 0")
        self.position_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        nav_layout.addWidget(self.position_label)

        nav_layout.addSpacing(20)

        self.btn_prev = QPushButton("<< Предыдущий")
        self.btn_next = QPushButton("Следующий >>")
        self.btn_next_unlabeled = QPushButton("⏭ Следующий неразмеченный")

        nav_btn_style = "padding: 6px 12px; min-height: 30px;"
        self.btn_prev.setStyleSheet(nav_btn_style)
        self.btn_next.setStyleSheet(nav_btn_style)
        self.btn_next_unlabeled.setStyleSheet(nav_btn_style)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_next_unlabeled)

        nav_layout.addSpacing(20)

        nav_layout.addWidget(QLabel("Перейти к:"))
        self.jump_input = QSpinBox()
        self.jump_input.setMinimum(1)
        self.jump_input.setMaximum(1)
        self.jump_input.setMaximumWidth(120)
        self.jump_btn = QPushButton("Перейти")
        nav_layout.addWidget(self.jump_input)
        nav_layout.addWidget(self.jump_btn)

        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)

        current_layout = QHBoxLayout()
        current_label_title = QLabel("Текущая разметка:")
        current_label_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        current_layout.addWidget(current_label_title)

        self.current_label = QLabel("Не размечено")
        self.current_label.setStyleSheet(
            "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: gray;"
        )
        current_layout.addWidget(self.current_label)
        current_layout.addStretch()
        main_layout.addLayout(current_layout)

        query_layout = QVBoxLayout()
        query_layout.addWidget(QLabel("Query:"))
        self.query_display = QTextEdit()
        self.query_display.setReadOnly(True)
        self.query_display.setMaximumHeight(180)
        query_layout.addWidget(self.query_display)
        main_layout.addLayout(query_layout)

        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Text:"))
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(300)
        text_layout.addWidget(self.text_display)
        main_layout.addLayout(text_layout, stretch=1)

        annotation_layout = QHBoxLayout()

        self.btn_not_similar = QPushButton("Не похожи")
        self.btn_partial = QPushButton("Похожи частично")
        self.btn_similar = QPushButton("Похожи")
        self.btn_skip = QPushButton("⏭ Пропустить")
        self.delete_btn = QPushButton("🗑 Удалить строку")

        common_btn_style = (
            "padding: 10px 16px;"
            "min-height: 30px;"
            "font-weight: bold;"
        )
        self.btn_not_similar.setStyleSheet(
            "background-color: #e74c3c; color: white; " + common_btn_style
        )
        self.btn_partial.setStyleSheet(
            "background-color: #f39c12; color: white; " + common_btn_style
        )
        self.btn_similar.setStyleSheet(
            "background-color: #2ecc71; color: white; " + common_btn_style
        )
        self.btn_skip.setStyleSheet(
            "background-color: #95a5a6; color: white; " + common_btn_style
        )
        self.delete_btn.setStyleSheet(common_btn_style)

        btn_font = QFont("Arial", 14, QFont.Bold)
        self.btn_not_similar.setFont(btn_font)
        self.btn_partial.setFont(btn_font)
        self.btn_similar.setFont(btn_font)
        self.btn_skip.setFont(btn_font)
        self.delete_btn.setFont(btn_font)

        self.btn_not_similar.setToolTip("Не похожи (клавиша 1)")
        self.btn_partial.setToolTip("Похожи частично (клавиша 2)")
        self.btn_similar.setToolTip("Похожи (клавиша 3)")

        annotation_layout.addWidget(self.btn_not_similar)
        annotation_layout.addWidget(self.btn_partial)
        annotation_layout.addWidget(self.btn_similar)
        annotation_layout.addWidget(self.btn_skip)
        annotation_layout.addWidget(self.delete_btn)

        main_layout.addLayout(annotation_layout)

        self.connect_signals()

    def connect_signals(self):
        try:
            self.load_btn.clicked.connect(self.load_data)
            self.save_btn.clicked.connect(self.save_data)
            self.save_as_btn.clicked.connect(self.save_data_as)
            self.delete_btn.clicked.connect(self.delete_current_record)

            self.btn_not_similar.clicked.connect(lambda: self.annotate_current(1))
            self.btn_partial.clicked.connect(lambda: self.annotate_current(2))
            self.btn_similar.clicked.connect(lambda: self.annotate_current(3))
            self.btn_skip.clicked.connect(self.skip_current)

            self.btn_prev.clicked.connect(self.previous_record)
            self.btn_next.clicked.connect(self.next_record)
            self.btn_next_unlabeled.clicked.connect(self.next_unlabeled)

            self.jump_btn.clicked.connect(self.jump_to_record)
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
                lambda: self.annotate_current(2)
            )
            QShortcut(QKeySequence("3"), self).activated.connect(
                lambda: self.annotate_current(3)
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
            self.loader_thread.finished.connect(
                lambda df: self.on_data_loaded(df, file_path)
            )
            self.loader_thread.error.connect(self.on_load_error)
            self.loader_thread.start()
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            self.load_btn.setEnabled(True)
            self.load_btn.setText("📁 Загрузить csv")

    def on_data_loaded(self, df, file_path=None):
        try:
            self.df = df
            self.current_index = 0
            self.unsaved_changes = False
            self.data_loaded = True
            if file_path:
                self.current_file_path = file_path

            max_records = len(self.df)
            self.jump_input.setMaximum(max_records)

            if self.current_file_path:
                self.file_info.setText(
                    f"Файл: {self.current_file_path} | Записей: {len(self.df)}"
                )
            else:
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
        except Exception as e:
            logging.error(f"Ошибка при обновлении UI: {e}", exc_info=True)

    def clear_ui(self):
        try:
            self.query_display.clear()
            self.text_display.clear()
            self.position_label.setText("Запись 0 из 0")
            self.current_label.setText("Не размечено")
            self.current_label.setStyleSheet(
                "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: gray;"
            )

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
                    "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: gray;"
                )
            elif similar == 1:
                self.current_label.setText("Не похожи")
                self.current_label.setStyleSheet(
                    "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: red;"
                )
            elif similar == 2:
                self.current_label.setText("Похожи частично")
                self.current_label.setStyleSheet(
                    "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: #f39c12;"
                )
            else:
                self.current_label.setText("Похожи")
                self.current_label.setStyleSheet(
                    "font-family: 'Segoe UI', Arial; font-size: 26px; font-weight: bold; color: green;"
                )
        except Exception as e:
            logging.error(f"Ошибка при отображении текущей записи: {e}", exc_info=True)

    def update_buttons_state(self):
        try:
            has_data = self.data_loaded and self.df is not None
            self.save_btn.setEnabled(has_data and self.unsaved_changes)
            self.save_as_btn.setEnabled(has_data)
            self.delete_btn.setEnabled(has_data)
            self.btn_prev.setEnabled(has_data and self.current_index > 0)
            self.btn_next.setEnabled(has_data and self.current_index < len(self.df) - 1)
            self.btn_not_similar.setEnabled(has_data)
            self.btn_partial.setEnabled(has_data)
            self.btn_similar.setEnabled(has_data)
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

            logging.debug(f"Запись {current_idx} размечена: is_similar={is_similar}")

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

            if not self.current_file_path:
                return self.save_data_as()

            self.df.to_csv(self.current_file_path, index=False, encoding="utf-8")
            self.unsaved_changes = False
            self.update_buttons_state()
            logging.info(f"Данные сохранены в файл: {self.current_file_path}")
            QMessageBox.information(
                self, "Успех", f"Данные сохранены в {self.current_file_path}"
            )
            return True

        except Exception as e:
            error_msg = f"Ошибка сохранения: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Ошибка", error_msg)
            return False

    def save_data_as(self):
        try:
            if not self.data_loaded:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения")
                return False

            default_name = (
                self.current_file_path if self.current_file_path else "labelled_sts_data.csv"
            )

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить как",
                default_name,
                "CSV файлы (*.csv);;Все файлы (*)",
            )

            if not file_path:
                return False

            self.df.to_csv(file_path, index=False, encoding="utf-8")
            self.current_file_path = file_path
            self.unsaved_changes = False
            self.update_buttons_state()
            self.file_info.setText(
                f"Файл: {self.current_file_path} | Записей: {len(self.df)}"
            )
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