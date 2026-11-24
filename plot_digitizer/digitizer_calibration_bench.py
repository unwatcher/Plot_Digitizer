"""Editable axis anchors shared by the digitizer bench and OCR review."""

import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemDelegate, QAbstractItemView, QApplication, QCheckBox, QComboBox, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSizePolicy, QStyledItemDelegate, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

try:
    from .axis_calibration import calibration_coefficients, parse_axis_number
    from .ui_layout import FlowLayout
except ImportError:  # flat offline archive
    from axis_calibration import calibration_coefficients, parse_axis_number
    from ui_layout import FlowLayout


CALIBRATION_LABELS = {
    "X axis": "X 轴", "Y axis": "Y 轴",
    "Calibration bench": "校准工作台",
    "Original OCR label": "OCR 原始识别文本",
    "Use all": "全部使用",
    "Include or exclude every tick on this axis.": "勾选或取消勾选此轴的所有刻度。",
    "Check at least two ticks in the Use column.": "请在使用列勾选至少两个刻度。",
    "Click a value to edit it. Select a tick to place or nudge it on the image.": "单击数值即可编辑。选择刻度后，可在图中定位或微调。",
    "Add tick": "添加刻度", "Remove tick": "移除刻度", "Place on image": "在图中定位",
    "Nudge −1 px": "微调 −1 像素", "Nudge +1 px": "微调 +1 像素",
    "Move along the selected axis. Alt + an arrow key moves the tick one image pixel in that direction.": "沿所选坐标轴移动。Alt + 方向键可沿对应方向移动一个图像像素。",
    "Apply calibration": "应用校准", "Revert edits": "还原修改",
    "Zoom to selected tick": "放大所选刻度",
    "Click the precise tick position on the scan.": "请在原图中单击准确的刻度位置。",
    "The tick must lie inside the image.": "刻度必须位于图像内。",
    "Enter a numeric value.": "请输入数值。",
    "Enter valid pixel coordinates inside the image.": "请输入图像内的有效像素坐标。",
    "Each axis needs at least two calibration points.": "每个坐标轴至少需要两个校准点。",
    "Use distinct tick positions and numeric values for each axis.": "每个坐标轴的刻度位置和数值必须不同。",
    "Each calibration label must use a distinct tick position.": "每个校准标签必须对应不同的刻度位置。",
    "Axis values must increase or decrease consistently along the ticks.": "沿刻度方向，轴数值必须单调递增或递减。",
    "Logarithmic axis labels must be positive.": "对数坐标轴的刻度值必须为正数。",
    "Calibration coordinates and values must be finite.": "校准坐标及其数值必须为有限数。",
    "The selected ticks do not define a valid calibration.": "所选刻度无法构成有效的校准。",
}


def _number_text(number):
    if number is None or number == "":
        return ""
    if isinstance(number, (float, np.floating)) and abs(number) < 1e12 and number.is_integer():
        return str(int(number))
    return str(number)


class _TickDelegate(QStyledItemDelegate):
    """Validate while typing so Apply is available without a second click."""

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            editor.setProperty("originalTickText", index.data(Qt.EditRole))
            editor.textEdited.connect(lambda _text: self.commitData.emit(editor))
        return editor

    def setEditorData(self, editor, index):
        if isinstance(editor, QLineEdit) and editor.text() == index.data(Qt.EditRole):
            # A live commit can echo back through dataChanged. Keep the caret.
            return
        super().setEditorData(editor, index)

    def eventFilter(self, editor, event):
        if (event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape
                and isinstance(editor, QLineEdit)):
            editor.setText(editor.property("originalTickText"))
            self.commitData.emit(editor)
        return super().eventFilter(editor, event)


class _UseAllCheckBox(QCheckBox):
    def nextCheckState(self):
        self.setCheckState(Qt.Unchecked if self.checkState() == Qt.Checked else Qt.Checked)


class CalibrationBench(QWidget):
    """Stage edits locally. Only the owner applies the two validated fits."""

    draftChanged = Signal()
    applyRequested = Signal()
    pickRequested = Signal()

    def __init__(self, translate, parent=None):
        super().__init__(parent)
        self._tr = translate
        self._baseline = {}
        self._loading = True
        self.image_shape = None
        self.dirty = False
        self.valid = False
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        body.addWidget(self.hint)
        self.tabs = QTabWidget()
        self.tables, self.scales, self.scale_labels, self.use_all_checks = {}, {}, {}, {}
        for axis in ("x", "y"):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(3, 4, 3, 4)
            scale_row = FlowLayout()
            self.scale_labels[axis] = label = QLabel()
            scale_row.addWidget(label)
            self.scales[axis] = scale = QComboBox()
            for key in ("linear", "log"):
                scale.addItem("", key)
            label.setBuddy(scale)
            scale.currentIndexChanged.connect(self._edited)
            scale_row.addWidget(scale)
            self.use_all_checks[axis] = use_all = _UseAllCheckBox()
            use_all.setTristate(True)
            use_all.clicked.connect(lambda checked, key=axis: self._set_all_used(key, checked))
            scale_row.addWidget(use_all)
            layout.addLayout(scale_row)
            self.tables[axis] = table = QTableWidget(0, 5)
            table.setItemDelegate(_TickDelegate(table))
            table.setColumnHidden(2 if axis == "x" else 1, True)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.horizontalHeader().setMinimumSectionSize(24)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            table.setMinimumWidth(0)
            table.setMaximumHeight(195)
            table.itemChanged.connect(self._edited)
            table.currentCellChanged.connect(self._selection_changed)
            table.cellClicked.connect(lambda row, col, key=axis: self._edit_clicked(key, row, col))
            layout.addWidget(table, 1)
            self.tabs.addTab(page, axis.upper())
        body.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self._selection_changed)

        tools = FlowLayout()
        self.add_button = QPushButton()
        self.remove_button = QPushButton()
        self.pick_button = QPushButton()
        self.minus_button = QPushButton()
        self.plus_button = QPushButton()
        self.add_button.clicked.connect(self._add_tick)
        self.remove_button.clicked.connect(self._remove_tick)
        self.pick_button.clicked.connect(self.pickRequested)
        self.minus_button.clicked.connect(lambda: self._nudge_axis(-1))
        self.plus_button.clicked.connect(lambda: self._nudge_axis(1))
        for button in (self.add_button, self.remove_button, self.pick_button, self.minus_button, self.plus_button):
            button.setAutoDefault(False)
            tools.addWidget(button)
        body.addLayout(tools)
        self.error = QLabel()
        self.error.setWordWrap(True)
        body.addWidget(self.error)
        footer = FlowLayout()
        self.apply_button = QPushButton()
        self.revert_button = QPushButton()
        self.apply_button.setObjectName("primary")
        self.apply_button.setAutoDefault(False)
        self.apply_button.clicked.connect(self.applyRequested)
        self.revert_button.setAutoDefault(False)
        self.revert_button.clicked.connect(self.revert)
        footer.addWidget(self.apply_button)
        footer.addWidget(self.revert_button)
        body.addLayout(footer)
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            shortcut = QShortcut(QKeySequence("Alt+" + key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda x=dx, y=dy: self.nudge(x, y))
        self.set_language()
        self.load_axes({}, None)

    def set_language(self):
        self.hint.setText(self._tr("Click a value to edit it. Select a tick to place or nudge it on the image."))
        for axis, caption in (("x", "X axis"), ("y", "Y axis")):
            i = 0 if axis == "x" else 1
            self.tabs.setTabText(i, self._tr(caption))
            self.tables[axis].setAccessibleName(self._tr(caption))
            self.tables[axis].setHorizontalHeaderLabels(
                [self._tr(key) for key in ("Use", "Pixel X", "Pixel Y", "Value", "OCR")])
            self.tables[axis].horizontalHeaderItem(4).setToolTip(self._tr("Original OCR label"))
            self.use_all_checks[axis].setText(self._tr("Use all"))
            self.use_all_checks[axis].setToolTip(self._tr("Include or exclude every tick on this axis."))
            self.scale_labels[axis].setText(self._tr("Axis Scale"))
            for index, key in enumerate(("Linear", "Logarithmic")):
                self.scales[axis].setItemText(index, self._tr(key))
        for button, key in ((self.add_button, "Add tick"), (self.remove_button, "Remove tick"),
                            (self.pick_button, "Place on image"), (self.minus_button, "Nudge −1 px"),
                            (self.plus_button, "Nudge +1 px"), (self.apply_button, "Apply calibration"),
                            (self.revert_button, "Revert edits")):
            button.setText(self._tr(key))
        tip = self._tr("Move along the selected axis. Alt + an arrow key moves the tick one image pixel in that direction.")
        for button in (self.minus_button, self.plus_button):
            button.setToolTip(tip)
        if not self._loading:
            self._validate()

    def load_axes(self, axes, image_shape):
        """Load applied axes or raw OCR proposals, retaining unselected suggestions."""
        axes = axes or {}
        self.image_shape = tuple(image_shape[:2]) if image_shape is not None else None
        self._baseline = {}
        for axis in ("x", "y"):
            source = axes.get(axis, {})
            points = source.get("points", [])
            accepted = len(points) if not source.get("scale_conflict") else 0
            rows = []
            for index, point in enumerate([*points, *source.get("rejected", [])]):
                x, y = point["pixel"]
                rows.append((index < accepted, _number_text(x), _number_text(y),
                             _number_text(point.get("value")), str(point.get("text", ""))))
            self._baseline[axis] = (source.get("scale", "linear"), tuple(rows))
        self.setEnabled(bool(self.image_shape and min(self.image_shape) > 0))
        self.revert()

    def revert(self):
        self._restore(self)

    def accept_draft(self, source=None):
        """Keep the applied draft, including OCR labels and excluded alternatives."""
        source = self if source is None else source
        self._baseline = {axis: source._snapshot(axis) for axis in self.tables}
        if source is self:
            self._edited()
        else:
            self.image_shape = source.image_shape
            self.setEnabled(bool(self.image_shape and min(self.image_shape) > 0))
            self._restore(source)

    def _restore(self, source):
        current_rows = {axis: table.currentRow() for axis, table in source.tables.items()}
        tab = source.tabs.currentIndex()
        self._loading = True
        for axis in ("x", "y"):
            table, combo = self.tables[axis], self.scales[axis]
            scale, rows = self._baseline[axis]
            combo.setCurrentIndex(combo.findData(scale))
            picked = max(0, min(current_rows[axis], len(rows)-1))
            table.setRowCount(0)
            table.setRowCount(len(rows))
            for row, fields in enumerate(rows):
                self._write_row(axis, row, fields)
            if rows:
                table.setCurrentCell(picked, 3)
            table.setColumnHidden(4, not any(fields[4] for fields in rows))
        self.tabs.setCurrentIndex(tab)
        self._loading = False
        self._edited()

    def _write_row(self, axis, row, fields):
        checked, *values = fields
        table = self.tables[axis]
        table.setVerticalHeaderItem(row, QTableWidgetItem(f"{axis.upper()}{row+1}"))
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        use.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        table.setItem(row, 0, use)
        for column, value in enumerate(values, 1):
            item = QTableWidgetItem(value)
            if column == 4:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            else:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, column, item)

    def selection(self):
        axis = "x" if self.tabs.currentIndex() == 0 else "y"
        row = self.tables[axis].currentRow()
        return (axis, row) if row >= 0 else None

    def select_tick(self, axis, row):
        if axis in self.tables and 0 <= row < self.tables[axis].rowCount():
            self.tabs.setCurrentIndex(0 if axis == "x" else 1)
            self.tables[axis].setCurrentCell(row, 3)

    def _edit_clicked(self, axis, row, column):
        if column in (1, 2, 3):
            self.tables[axis].editItem(self.tables[axis].item(row, column))

    def _set_all_used(self, axis, checked):
        table = self.tables[axis]
        with QSignalBlocker(table):
            for row in range(table.rowCount()):
                table.item(row, 0).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._edited()

    def _add_tick(self):
        if self.image_shape is None:
            return
        axis = "x" if self.tabs.currentIndex() == 0 else "y"
        table = self.tables[axis]
        cross_column = 2 if axis == "x" else 1
        prev = table.item(table.currentRow(), cross_column)
        cross = prev.text() if prev else str(self.image_shape[0] - 1 if axis == "x" else 0)
        x, y = ("", cross) if axis == "x" else (cross, "")
        row = table.rowCount()
        with QSignalBlocker(table):
            table.insertRow(row)
            self._write_row(axis, row, (True, x, y, "", ""))
            table.setCurrentCell(row, 1 if axis == "x" else 2)
        self._edited()
        table.setFocus()

    def _remove_tick(self):
        selected = self.selection()
        if selected is None:
            return
        axis, row = selected
        table = self.tables[axis]
        with QSignalBlocker(table):
            table.removeRow(row)
            for remaining in range(row, table.rowCount()):
                table.verticalHeaderItem(remaining).setText(f"{axis.upper()}{remaining+1}")
            if table.rowCount():
                table.setCurrentCell(min(row, table.rowCount()-1), 3)
        self._edited()

    def _inside(self, x, y):
        return (self.image_shape is not None and 0 <= x < self.image_shape[1]
                and 0 <= y < self.image_shape[0])

    def set_pixel(self, px, py):
        selected = self.selection()
        if selected is None:
            return False
        if px is None or py is None or not self._inside(px, py):
            self.error.setText(self._tr("The tick must lie inside the image."))
            self.error.show()
            return False
        axis, row = selected
        table = self.tables[axis]
        with QSignalBlocker(table):
            table.item(row, 1).setText(_number_text(px))
            table.item(row, 2).setText(_number_text(py))
        self._edited()
        return True

    def _nudge_axis(self, step):
        dx, dy = (step, 0) if self.tabs.currentIndex() == 0 else (0, step)
        self.nudge(dx, dy)

    def nudge(self, dx, dy):
        selected = self.selection()
        if selected is None or self.image_shape is None:
            return
        axis, row = selected
        table = self.tables[axis]
        try:
            x = float(table.item(row, 1).text())
            y = float(table.item(row, 2).text())
        except ValueError:
            return
        if not self._inside(x, y):
            return
        h, w = self.image_shape
        self.set_pixel(max(0, min(w-1, x+dx)), max(0, min(h-1, y+dy)))

    def preview_points(self, include_selected_excluded=False):
        """Source positions indexed by axis, optionally showing an excluded selection."""
        result = {"x": [], "y": []}
        selected = self.selection()
        for axis, table in self.tables.items():
            for row in range(table.rowCount()):
                checked = table.item(row, 0).checkState() == Qt.Checked
                is_selected = selected == (axis, row)
                if not checked and not (include_selected_excluded and is_selected):
                    continue
                try:
                    x = float(table.item(row, 1).text())
                    y = float(table.item(row, 2).text())
                except ValueError:
                    continue
                if not self._inside(x, y):
                    continue
                value = parse_axis_number(table.item(row, 3).text())
                result[axis].append({"row": row, "pixel": [x, y], "value": value,
                                     "selected": is_selected, "checked": checked})
        return result

    def _snapshot(self, axis):
        table = self.tables[axis]
        rows = tuple((table.item(row, 0).checkState() == Qt.Checked,
                      *(table.item(row, col).text() for col in range(1, 5)))
                     for row in range(table.rowCount()))
        return self.scales[axis].currentData(), rows

    def _edited(self):
        if self._loading:
            return
        snapshots = {axis: self._snapshot(axis) for axis in self.tables}
        self.dirty = snapshots != self._baseline
        for axis, (_, rows) in snapshots.items():
            use_all = self.use_all_checks[axis]
            used = sum(row[0] for row in rows)
            state = Qt.Unchecked if used == 0 else Qt.Checked if used == len(rows) else Qt.PartiallyChecked
            use_all.setEnabled(bool(rows))
            use_all.setCheckState(state)
        self._validate()
        self._selection_changed()

    def _selection_changed(self):
        if self._loading:
            return
        selected = self.selection() is not None
        for button in (self.remove_button, self.pick_button, self.minus_button, self.plus_button):
            button.setEnabled(selected)
        self.draftChanged.emit()

    def _validate(self):
        try:
            self._checked_axes()
        except ValueError as error:
            self.valid = False
            self.error.setText(str(error) if self.dirty or any(t.rowCount() for t in self.tables.values()) else "")
        else:
            self.valid = True
            self.error.clear()
        self.error.setVisible(bool(self.error.text()))
        self.revert_button.setEnabled(self.dirty)
        self.apply_button.setEnabled(self.valid and self.dirty)

    def _checked_axes(self):
        axes = {}
        for axis, table in self.tables.items():
            points = []
            for row in range(table.rowCount()):
                if table.item(row, 0).checkState() != Qt.Checked:
                    continue
                try:
                    x, y = (float(table.item(row, col).text()) for col in (1, 2))
                    if not self._inside(x, y):
                        raise ValueError()
                except ValueError:
                    raise ValueError(f'{axis.upper()}{row+1} — {self._tr("Enter valid pixel coordinates inside the image.")}') from None
                value = parse_axis_number(table.item(row, 3).text())
                if value is None:
                    raise ValueError(f'{axis.upper()}{row+1} — {self._tr("Enter a numeric value.")}')
                points.append({"pixel": [x, y], "value": value})
            if len(points) < 2:
                raise ValueError(f'{axis.upper()} — {self._tr("Check at least two ticks in the Use column.")}')
            scale = self.scales[axis].currentData()
            try:
                calibration_coefficients(points, axis, scale)
            except ValueError as error:
                raise ValueError(f"{axis.upper()} — {self._tr(str(error))}") from None
            axes[axis] = {"points": points, "scale": scale}
        return axes

    def reviewed_axes(self):
        editor = QApplication.focusWidget()
        if isinstance(editor, QLineEdit):
            for table in self.tables.values():
                if table.isAncestorOf(editor) and table.state() == QAbstractItemView.EditingState:
                    table.commitData(editor)
                    table.closeEditor(editor, QAbstractItemDelegate.NoHint)
                    break
        return self._checked_axes()
