"""TutorAndQuizBot desktop interface.

Run from the project folder:
    uv run python -m ui.main
"""

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer, QRunnable, QThreadPool, QObject
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget,
)

try:
    from agents import StudyAssistant
except ImportError:
    from agents.agents import StudyAssistant

MAX_FILE_SIZE = 50 * 1024 * 1024

# The colors and appearance of the application.
STYLE = """
QMainWindow, QWidget { background: #111111; color: #ededed; font: 13px 'Segoe UI'; }
QMenuBar { background: #171717; color: #ddd; border-bottom: 1px solid #333; padding: 3px; }
QWidget#menuControls { background: transparent; }
QLineEdit#command { background: #0b0b0b; color: #dedede; font: 12px Consolas; border: 0; padding: 5px 8px; }
QFrame#consolePane { background: #0b0b0b; border: 0; }
QMenuBar::item { padding: 5px 11px; }
QMenuBar::item:selected, QMenu::item:selected { background: #383838; }
QMenu { background: #1d1d1d; color: #eee; border: 1px solid #444; padding: 5px; }
QMenu::item { padding: 7px 25px; }
QFrame#header { border-bottom: 1px solid #303030; background: #151515; }
QFrame#chat { border: 0; background: #111111; }
QFrame#messageUser { background: #343434; border-radius: 9px; }
QFrame#messageUser QLabel, QFrame#messageAgent QLabel { background: transparent; border: none; }
QFrame#messageAgent { background: #202020; border: 1px solid #333; border-radius: 9px; }
QPushButton { background: #282828; color: #eee; border: 1px solid #424242; border-radius: 6px; padding: 8px 13px; }
QPushButton:hover { background: #373737; }
QPushButton:pressed { background: #484848; }
QPushButton:disabled { color: #777; background: #202020; border-color: #303030; }
QPushButton#selected { background: #e9e9e9; color: #111; border-color: #e9e9e9; }
QPushButton#send { background: #dcdcdc; color: #121212; font-weight: bold; }
QPlainTextEdit { background: #1b1b1b; color: #eee; border: 1px solid #3b3b3b; border-radius: 7px; padding: 8px; selection-background-color: #676767; }
QPlainTextEdit#console { background: #0b0b0b; color: #bababa; border: 0; border-radius: 0; font: 12px Consolas; }
QProgressBar { background: #303030; border: 0; border-radius: 3px; min-height: 6px; max-height: 6px; }
QProgressBar::chunk { background: #dadada; border-radius: 3px; }
QScrollArea { border: 0; background: #111; }
QScrollArea > QWidget > QWidget { background: #111; }
QSplitter::handle { background: #333; }
QSplitter::handle:hover { background: #777; }
QStatusBar { background: #181818; color: #aaa; border-top: 1px solid #303030; }
"""


class TaskSignals(QObject):
    finished = Signal(object, str, object)
    failed = Signal(object, str, str)


class AgentTask(QRunnable):
    """Run Ollama and document-ingestion work without freezing the GUI."""

    def __init__(self, header, function):
        super().__init__()
        self.header = header
        self.function = function
        self.signals = TaskSignals()

    def run(self):
        try:
            result = self.function()
            self.signals.finished.emit(self, self.header, result)
        except Exception as exc:
            self.signals.failed.emit(self, self.header, str(exc))


class MessageInput(QPlainTextEdit):
    # The Enter key sends the message. Shift + Enter adds a new line.
    send_requested = Signal()

    def keyPressEvent(self, event):
        is_enter = event.key() == Qt.Key.Key_Return
        is_enter = is_enter or event.key() == Qt.Key.Key_Enter
        shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier

        if is_enter and not shift_pressed:
            self.send_requested.emit()
            event.accept()
            return

        super().keyPressEvent(event)


class ChatPanel(QFrame):
    # The main window listens for this message.
    message_sent = Signal(str)

    def __init__(self, bot_name):
        super().__init__()
        self.bot_name = bot_name
        self.history = []
        self.setObjectName("chat")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 13)
        layout.setSpacing(10)

        title = QLabel(bot_name)
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        # Scrollable area for previous messages.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        container = QWidget()
        self.message_layout = QVBoxLayout(container)
        self.message_layout.setContentsMargins(5, 6, 5, 6)
        self.message_layout.setSpacing(11)
        self.message_layout.addStretch()
        self.scroll_area.setWidget(container)
        layout.addWidget(self.scroll_area, 1)

        # Keep the existing message box and Send button.
        input_row = QHBoxLayout()
        self.message_input = MessageInput()
        self.message_input.setPlaceholderText("Message...")
        self.message_input.setFixedHeight(66)
        self.message_input.send_requested.connect(self.send_message)
        input_row.addWidget(self.message_input, 1)

        send_button = QPushButton("Send")
        send_button.setObjectName("send")
        send_button.clicked.connect(self.send_message)
        input_row.addWidget(send_button, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(input_row)

    def add_message(self, speaker, message):
        # A user message sits on the right. A bot message sits on the left.
        row = QHBoxLayout()
        bubble = QFrame()

        if speaker == "user":
            bubble.setObjectName("messageUser")
            name = "You"
        else:
            bubble.setObjectName("messageAgent")
            name = self.bot_name

        bubble.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 10, 12, 10)

        name_label = QLabel(name)
        name_label.setStyleSheet(
            "background: transparent; border: none; color: #b8b8b8; font-size: 11px; font-weight: bold;"
        )
        bubble_layout.addWidget(name_label)

        message_label = QLabel(message)
        message_label.setTextFormat(Qt.TextFormat.PlainText)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        message_label.setStyleSheet("background: transparent; border: none;")
        bubble_layout.addWidget(message_label)

        # Keep messages wide enough to read, but not wider than the chat.
        chat_width = self.scroll_area.viewport().width()
        max_width = min(620, int(chat_width * 0.78))
        max_width = max(130, max_width)

        text_width = 0
        for line in message.splitlines():
            line_width = message_label.fontMetrics().horizontalAdvance(line)
            text_width = max(text_width, line_width)

        name_width = name_label.fontMetrics().horizontalAdvance(name)
        bubble_width = max(130, text_width + 32, name_width + 32)
        bubble_width = min(max_width, bubble_width)

        bubble.setFixedWidth(bubble_width)
        message_label.setFixedWidth(bubble_width - 24)

        if speaker == "user":
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()

        self.message_layout.insertLayout(self.message_layout.count() - 1, row)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def send_message(self):
        message = self.message_input.toPlainText().strip()
        if not message:
            return

        self.message_input.clear()
        self.history.append(("user", message))
        self.add_message("user", message)

        self.message_sent.emit(message)


class MainWindow(QMainWindow):
    # Connect your teammate's listener to these three signals.
    # Each emits (header, payload).
    tutor_event = Signal(tuple)
    quiz_event = Signal(tuple)
    file_event = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TutorAndQuizBot")
        self.resize(1220, 790)
        self.setMinimumSize(740, 480)

        self.upload_queue = []
        self.processing = False
        self.worker = None
        self.thread = None
        self.current_filename = ""
        self.files_sent = 0
        self.assistant = StudyAssistant()
        self.thread_pool = QThreadPool.globalInstance()

        self.make_window()
        self.make_menus()

        # Route the GUI's three events directly to the backend commands.
        self.tutor_event.connect(self.handle_tutor_event)
        self.quiz_event.connect(self.handle_quiz_event)
        self.file_event.connect(self.handle_file_event)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("Ready")
        self.write_log("Ready")

    def make_window(self):
        # The top part always contains Tutor and Quiz side by side.
        self.tutor = ChatPanel("Tutor")
        self.quiz = ChatPanel("Quiz")
        self.tutor.message_sent.connect(self.send_tutor)
        self.quiz.message_sent.connect(self.send_quiz)

        self.chat_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.chat_splitter.setChildrenCollapsible(False)
        self.chat_splitter.setHandleWidth(7)
        self.chat_splitter.addWidget(self.tutor)
        self.chat_splitter.addWidget(self.quiz)
        self.chat_splitter.setSizes([600, 600])

        # The bottom part is a console whose height can be dragged.
        self.console_panel = QFrame()
        self.console_panel.setObjectName("consolePane")
        console_layout = QVBoxLayout(self.console_panel)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)

        console_heading = QHBoxLayout()
        console_heading.setContentsMargins(12, 5, 12, 5)
        console_heading.addWidget(QLabel("Terminal"))
        console_heading.addStretch()

        self.progress_label = QLabel("")
        console_heading.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(110)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        console_heading.addWidget(self.progress_bar)
        console_layout.addLayout(console_heading)

        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(600)
        console_layout.addWidget(self.console, 1)

        self.command_input = QLineEdit()
        self.command_input.setObjectName("command")
        self.command_input.setPlaceholderText(">")
        self.command_input.returnPressed.connect(self.run_command)
        console_layout.addWidget(self.command_input)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setChildrenCollapsible(False)
        self.vertical_splitter.setHandleWidth(5)
        self.vertical_splitter.addWidget(self.chat_splitter)
        self.vertical_splitter.addWidget(self.console_panel)
        self.vertical_splitter.setSizes([600, 170])

        self.setCentralWidget(self.vertical_splitter)

    def make_menus(self):
        file_menu = self.menuBar().addMenu("File")

        upload_action = QAction("Upload files...", self)
        upload_action.setShortcut("Ctrl+O")
        upload_action.triggered.connect(self.upload_files)
        file_menu.addAction(upload_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("View")
        self.terminal_action = QAction("Show terminal", self)
        self.terminal_action.setCheckable(True)
        self.terminal_action.setChecked(True)
        self.terminal_action.triggered.connect(self.console_panel.setVisible)
        view_menu.addAction(self.terminal_action)

    def write_log(self, message):
        time = datetime.now().strftime("%H:%M:%S")
        self.console.appendPlainText("[" + time + "] " + message)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def run_command(self):
        command = self.command_input.text().strip()
        self.command_input.clear()

        if not command:
            return

        command_lower = command.lower()
        if command_lower == "clear" or command_lower == "cls":
            self.console.clear()
        elif command_lower == "help":
            self.write_log("Commands: clear, cls, help, status")
        elif command_lower == "status":
            self.write_log("Files sent: " + str(self.files_sent))
        else:
            self.write_log("Unknown command: " + command)

    def send_tutor(self, message):
        # Send the Tutor text to the listener.
        self.tutor_event.emit(("t", message))
        self.write_log("Tutor: message emitted")

    def send_quiz(self, message):
        # Send the Quiz text to the listener.
        self.quiz_event.emit(("q", message))
        self.write_log("Quiz: message emitted")

    def upload_files(self):
        paths, selected_filter = QFileDialog.getOpenFileNames(
            self, "Upload files", "", "Documents (*.pdf *.docx *.txt)"
        )
        if not paths:
            return

        # Each file contains its original filename and raw bytes.
        # The backend is responsible for saving and indexing files.
        fileslist = []

        for file_path in paths:
            path = Path(file_path)
            filename = path.name

            if path.suffix.lower() not in [".pdf", ".docx", ".txt"]:
                self.write_log(filename + ": unsupported file type")
                continue

            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    self.write_log(filename + ": exceeds 50 MB")
                    continue

                file_bytes = path.read_bytes()
            except OSError:
                self.write_log(filename + ": could not read file")
                continue

            fileslist.append((filename, file_bytes))
            self.write_log(filename + ": ready")

        if not fileslist:
            return

        # One file event contains all the selected valid files.
        self.file_event.emit(("f", fileslist))
        self.files_sent += len(fileslist)
        self.write_log("File event emitted: " + str(len(fileslist)) + " file(s)")

    def start_agent_task(self, header, function):
        task = AgentTask(header, function)
        task.signals.finished.connect(self._task_finished)
        task.signals.failed.connect(self._task_failed)
        self.thread_pool.start(task)

    def _task_finished(self, task, header, result):
        self.receive_response(header, result)

    def _task_failed(self, task, header, error):
        self.receive_error(header, error)

    def handle_tutor_event(self, event):
        header, message = event
        self.write_log("Tutor: processing request")
        self.start_agent_task(
            header, lambda: self.assistant.ask_tutor(message)
        )

    def handle_quiz_event(self, event):
        header, message = event
        self.write_log("Quiz: processing request")
        self.start_agent_task(
            header, lambda: self.assistant.ask_quiz(message)
        )

    def handle_file_event(self, event):
        header, fileslist = event
        upload_dir = Path(__file__).resolve().parents[1] / "uploaded_documents"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for filename, file_bytes in fileslist:
            # Use only the basename so an uploaded filename cannot escape
            # the upload directory.
            safe_name = Path(filename).name
            destination = upload_dir / safe_name
            try:
                destination.write_bytes(file_bytes)
                saved_paths.append(str(destination))
                self.write_log(f"{safe_name}: saved; starting ingestion")
            except OSError as exc:
                self.write_log(f"{safe_name}: could not save ({exc})")

        if saved_paths:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText("Indexing documents...")
            self.progress_bar.show()
            self.start_agent_task(
                header, lambda: self.assistant.add_files(saved_paths)
            )

    def receive_response(self, header, payload):
        # This slot is invoked on the Qt UI thread by the worker's signal.
        if header == "t":
            message = str(payload)
            self.tutor.history.append(("assistant", message))
            self.tutor.add_message("assistant", message)
            self.write_log("Tutor: response received")
        elif header == "q":
            message = str(payload)
            self.quiz.history.append(("assistant", message))
            self.quiz.add_message("assistant", message)
            self.write_log("Quiz: response received")
        elif header == "f":
            if payload is None:
                self.write_log("Files ingested")
            else:
                for status in payload:
                    self.write_log(str(status))
            self.progress_label.setText("")
            self.progress_bar.hide()

    def receive_error(self, header, error):
        self.write_log(f"{header}: error: {error}")
        if header == "t":
            self.tutor.add_message("assistant", f"Error: {error}")
        elif header == "q":
            self.quiz.add_message("assistant", f"Error: {error}")
        elif header == "f":
            self.progress_label.setText("")
            self.progress_bar.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
