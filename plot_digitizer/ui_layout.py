"""Layouts that work in logical pixels, including small and scaled screens."""

from PySide6.QtCore import QEvent, QObject, QRect, QSize, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLayout, QLineEdit, QHBoxLayout, QGridLayout, QScrollArea,
    QSizePolicy, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

try:
    from .ui_appearance import apply_window_appearance
except ImportError:
    from ui_appearance import apply_window_appearance


class FlowLayout(QLayout):
    """Wrap a row of controls without shrinking their text or click targets."""

    def __init__(self):
        super().__init__()
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(8)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), measure_only=True)

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rectangle):
        super().setGeometry(rectangle)
        self._arrange(rectangle, measure_only=False)

    def _arrange(self, rectangle, *, measure_only):
        x, y, row_height = rectangle.x(), rectangle.y(), 0
        width = max(1, rectangle.width())
        spacing = self.spacing()
        for item in self.items:
            if item.isEmpty():
                continue
            item_width = min(width, item.sizeHint().width())
            if x > rectangle.x() and x + item_width > rectangle.x() + width:
                x = rectangle.x()
                y += row_height + spacing
                row_height = 0
            item_height = item.heightForWidth(item_width) if item.hasHeightForWidth() else item.sizeHint().height()
            item_height = max(item.minimumSize().height(), item_height)
            if not measure_only:
                item.setGeometry(QRect(x, y, item_width, item_height))
            x += item_width + spacing
            row_height = max(row_height, item_height)
        return y + row_height - rectangle.y()


def wrap_control_rows(body):
    """Keep label/field pairs together when long analysis rows wrap."""
    for row in body.findChildren(QHBoxLayout):
        if any(row.itemAt(index).layout() for index in range(row.count())):
            continue
        widgets = []
        while row.count():
            item = row.takeAt(0)
            if item.widget():
                widgets.append(item.widget())
        flow = FlowLayout()
        row.addLayout(flow)
        index = 0
        while index < len(widgets):
            widget = widgets[index]
            if isinstance(widget, QLabel):
                widget.setWordWrap(True)
            if (
                isinstance(widget, QLabel) and index + 1 < len(widgets)
                and isinstance(widgets[index + 1], (QAbstractSpinBox, QComboBox, QLineEdit))
            ):
                pair = QWidget(body)
                form = QFormLayout(pair)
                form.setContentsMargins(0, 0, 0, 0)
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                form.addRow(widget, widgets[index + 1])
                flow.addWidget(pair)
                index += 2
            else:
                flow.addWidget(widget)
                index += 1


def wrap_form_grids(body):
    """Let wide form grids reflow as intact label/field pairs."""
    for grid in body.findChildren(QGridLayout):
        columns = grid.columnCount()
        if columns < 4:
            continue
        if not any(isinstance(grid.itemAt(index).widget(), QLabel) for index in range(grid.count())):
            continue
        positions = [grid.getItemPosition(index) for index in range(grid.count())]
        if any(row_span != 1
               for _row, _column, row_span, column_span in positions):
            continue
        items = []
        while grid.count():
            position = grid.getItemPosition(0)
            items.append((position, grid.takeAt(0)))
        items.sort(key=lambda value: value[0][:2])
        for column in range(columns):
            grid.setColumnStretch(column, 0)
        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        grid.addLayout(stack, 0, 0, 1, columns)
        current_row = None
        fields = None
        index = 0
        while index < len(items):
            (row, column, _row_span, column_span), item = items[index]
            if row != current_row:
                current_row, fields = row, None
            if column_span == columns:
                if item.layout():
                    nested = item.layout()
                    nested.setParent(None)
                    stack.addLayout(nested)
                elif item.widget():
                    stack.addWidget(item.widget())
                index += 1
            elif (
                isinstance(item.widget(), QLabel) and index + 1 < len(items)
                and items[index + 1][0][:2] == (row, column + 1)
            ):
                if fields is None:
                    fields = FlowLayout()
                    stack.addLayout(fields)
                field = items[index + 1][1]
                pair = QWidget()
                form = QFormLayout(pair)
                form.setContentsMargins(0, 0, 0, 0)
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
                item.widget().setWordWrap(True)
                if field.layout():
                    nested = field.layout()
                    nested.setParent(None)
                    form.addRow(item.widget(), nested)
                else:
                    form.addRow(item.widget(), field.widget())
                fields.addWidget(pair)
                index += 2
            else:
                if item.widget():
                    if fields is None:
                        fields = FlowLayout()
                        stack.addLayout(fields)
                    fields.addWidget(item.widget())
                elif item.layout():
                    nested = item.layout()
                    nested.setParent(None)
                    stack.addLayout(nested)
                index += 1


class PanelScrollArea(QScrollArea):
    """Let a panel scroll instead of imposing its contents' minimum height."""

    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setWidget(widget)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def sizeHint(self):
        hint = self.widget().sizeHint()
        return QSize(hint.width() + 24, min(320, hint.height() + 8))

    def minimumSizeHint(self):
        return QSize(0, 100)


def fit_to_screen(window):
    """Fit to the current monitor's available logical workspace."""
    screen = window.screen()
    if screen is None:
        return
    bounds = screen.availableGeometry().adjusted(16, 16, -16, -16)
    window.resize(min(window.width(), bounds.width()), min(window.height(), bounds.height()))
    frame = window.frameGeometry()
    x = max(bounds.left(), min(frame.left(), bounds.right() - frame.width() + 1))
    y = max(bounds.top(), min(frame.top(), bounds.bottom() - frame.height() + 1))
    window.move(x, y)


class AdaptiveDialog(QDialog):
    """Scroll long dialog bodies while keeping standard action buttons visible."""

    adaptive_scroll = True
    adaptive_wrap_rows = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        apply_window_appearance(self)

    @staticmethod
    def _take_buttons(layout):
        buttons = []
        index = 0
        while index < layout.count():
            item = layout.itemAt(index)
            if isinstance(item.widget(), QDialogButtonBox):
                buttons.append(layout.takeAt(index).widget())
                continue
            if item.layout():
                buttons.extend(AdaptiveDialog._take_buttons(item.layout()))
            index += 1
        return buttons

    def showEvent(self, event):
        if not getattr(self, "_adaptive_prepared", False):
            self._adaptive_prepared = True
            for form in self.findChildren(QFormLayout):
                form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            layout = self.layout()
            if self.adaptive_scroll and isinstance(layout, QVBoxLayout):
                body = QWidget(self)
                body_layout = QVBoxLayout(body)
                body_layout.setContentsMargins(0, 0, 0, 0)
                footer = self._take_buttons(layout)
                while layout.count():
                    stretch = layout.stretch(0)
                    item = layout.takeAt(0)
                    if item.widget():
                        body_layout.addWidget(item.widget(), stretch)
                    elif item.layout():
                        nested = item.layout()
                        nested.setParent(None)
                        body_layout.addLayout(nested, stretch)
                    else:
                        body_layout.addItem(item)
                if self.adaptive_wrap_rows:
                    wrap_form_grids(body)
                    wrap_control_rows(body)
                self.body_scroll = PanelScrollArea(body, self)
                layout.addWidget(self.body_scroll, 1)
                for buttons in footer:
                    layout.addWidget(buttons)
            if layout:
                layout.setSizeConstraint(QLayout.SetNoConstraint)
            self.setMinimumSize(320, 240)
        super().showEvent(event)
        if self.adaptive_scroll and hasattr(self, "body_scroll") and self.screen():
            margins = self.layout().contentsMargins()
            needed_width = (
                self.body_scroll.widget().minimumSizeHint().width()
                + margins.left() + margins.right()
                + self.body_scroll.verticalScrollBar().sizeHint().width()
                + 2 * self.body_scroll.frameWidth()
            )
            available_width = self.screen().availableGeometry().width() - 32
            self.resize(max(self.width(), min(needed_width, available_width)), self.height())
        fit_to_screen(self)
        QTimer.singleShot(0, lambda: fit_to_screen(self))
        handle = self.windowHandle()
        if handle and not getattr(self, "_adaptive_screen_connected", False):
            handle.screenChanged.connect(lambda _screen: fit_to_screen(self))
            self._adaptive_screen_connected = True


class AdaptivePanes(QObject):
    """Keep a splitter on wide windows and expose every pane in compact tabs."""

    modeChanged = Signal(bool)

    def __init__(self, host, layout, splitter, panes, titles, wide_width):
        super().__init__(host)
        self.host, self.splitter = host, splitter
        self.panes, self.titles = panes, titles
        self.collapsible = [splitter.isCollapsible(index) for index in range(len(panes))]
        self.wide_width = wide_width
        self.compact = False
        self.saved_sizes = []
        self.stack = QStackedWidget(host)
        self.stack.setMinimumSize(0, 0)
        item = layout.replaceWidget(splitter, self.stack)
        del item
        self.stack.addWidget(splitter)
        self.tabs = QTabWidget(self.stack)
        self.stack.addWidget(self.tabs)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_layout)
        host.installEventFilter(self)
        self.timer.start(0)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Resize, QEvent.Show, QEvent.FontChange):
            self.timer.start(0)
        return False

    def update_layout(self):
        content_width = self.host.contentsRect().width()
        needed = max(self.wide_width, sum(max(p.minimumWidth(), p.minimumSizeHint().width())
                                         for p in self.panes) + 40)
        compact = content_width < needed
        if compact == self.compact:
            return
        self.compact = compact
        if compact:
            self.saved_sizes = self.splitter.sizes()
            for pane, title in zip(self.panes, self.titles):
                pane.setProperty("i18n_tab_key", title)
                self.tabs.addTab(pane, title)
            self.stack.setCurrentWidget(self.tabs)
        else:
            for index, pane in enumerate(self.panes):
                self.tabs.removeTab(self.tabs.indexOf(pane))
                self.splitter.addWidget(pane)
                self.splitter.setCollapsible(index, self.collapsible[index])
                pane.show()
            self.stack.setCurrentWidget(self.splitter)
            if self.saved_sizes:
                self.splitter.setSizes(self.saved_sizes)
        self.modeChanged.emit(compact)
