"""Keep a calibrated trace, its numerical points and the original scan in register."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)


def curve_breaks(pixels):
    """Break a sampled curve at missing columns or a return to an earlier X."""
    steps = np.diff(np.asarray(pixels, dtype=float)[:, 0])
    forward = steps[steps > 0]
    typical = np.median(forward) if forward.size else np.inf
    return np.flatnonzero((steps <= 0) | (steps > 1.5 * typical)) + 1


class _SourceDetail(QWidget):
    """Display a source-pixel neighborhood at fixed magnification, even at an image edge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self._buffer = None
        self._image = QImage()
        self.point = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        return QSize(200, 95)

    def set_source(self, source):
        if source is self.source:
            return
        self.source = source
        self._image = QImage()
        self._buffer = None
        self.point = None
        if source is not None:
            array = np.asarray(source)
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            channels = 1 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else 0
            formats = {1: QImage.Format_Grayscale8, 3: QImage.Format_RGB888, 4: QImage.Format_RGBA8888}
            if channels in formats and array.size:
                if array.dtype != np.uint8:
                    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1:
                        array = array * 255
                    array = np.clip(array, 0, 255).astype(np.uint8)
                self._buffer = np.ascontiguousarray(array)
                height, width = array.shape[:2]
                self._image = QImage(self._buffer.data, width, height, self._buffer.strides[0], formats[channels])
        self.update()

    def set_point(self, pixel):
        self.point = None
        if pixel is not None and not self._image.isNull():
            x, y = pixel
            if 0 <= x < self._image.width() and 0 <= y < self._image.height():
                self.point = (x, y)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        bounds = QRectF(self.rect())
        painter.fillRect(bounds, self.palette().brush(QPalette.AlternateBase))
        if self.point is None:
            return
        zoom = 4
        center = bounds.center()
        # Integer image coordinates identify pixel centers. Qt's image rectangles
        # start at the *edge* of the first pixel.
        x, y = self.point
        source = QRectF(x + 0.5 - bounds.width() / (2 * zoom),
                        y + 0.5 - bounds.height() / (2 * zoom),
                        bounds.width() / zoom, bounds.height() / zoom)
        visible = source.intersected(QRectF(self._image.rect()))
        target = QRectF((visible.x() - source.x()) * zoom, (visible.y() - source.y()) * zoom,
                        visible.width() * zoom, visible.height() * zoom)
        painter.drawImage(target, self._image, visible)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        for color, thickness in (("#ffffff", 4), ("#914100", 2)):
            painter.setPen(QPen(QColor(color), thickness))
            painter.drawEllipse(center, 6, 6)


class LinkedReview(QWidget):
    """Review the same source point through the graph, table, or keyboard selection."""

    pointSelected = Signal(int, int)

    def __init__(self, translate, parent=None):
        super().__init__(parent)
        self._tr = translate
        self._datasets = []
        self._offsets = []
        self._external_active = None
        self._selected_dataset = None
        self._selected_pixel = None
        self.selection = None
        self._scale = "linear"
        self._drawn_dataset = None
        self._trace = np.empty((0, 2))
        self._plottable = np.zeros(0, dtype=bool)
        self._marker = None
        self._line = None
        self._compact = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.controls = QGridLayout()
        self.controls.setColumnStretch(1, 1)
        self.heading = QLabel()
        font = self.heading.font()
        font.setBold(True)
        self.heading.setFont(font)
        self.dataset_picker = QComboBox()
        self.dataset_picker.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dataset_picker.setMinimumContentsLength(3)
        self.dataset_picker.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.point_label = QLabel()
        self.point_number = QSpinBox()
        self.point_number.setKeyboardTracking(False)
        self.point_label.setBuddy(self.point_number)
        self.point_control = QWidget()
        point_layout = QHBoxLayout(self.point_control)
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.addWidget(self.point_label)
        point_layout.addWidget(self.point_number)
        self.controls.addWidget(self.heading, 0, 0)
        self.controls.addWidget(self.dataset_picker, 0, 1)
        self.controls.addWidget(self.point_control, 0, 2)
        layout.addLayout(self.controls)

        self.figure = Figure(figsize=(5.4, 2.4), layout="constrained")
        self.axes = self.figure.add_subplot(111)
        self.trace_canvas = FigureCanvasQTAgg(self.figure)
        self.trace_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.trace_canvas.setMinimumSize(0, 0)
        self.trace_canvas.mpl_connect("button_press_event", self._pick_trace)
        self.table = QTableWidget(0, 3)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.detail_panel = QWidget()
        self.detail_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        detail_layout = QVBoxLayout(self.detail_panel)
        detail_layout.setContentsMargins(4, 0, 2, 0)
        self.detail_title = QLabel()
        self.source_detail = _SourceDetail()
        self.source_position = QLabel()
        self.source_position.setWordWrap(True)
        detail_layout.addWidget(self.detail_title)
        detail_layout.addWidget(self.source_detail, 1)
        detail_layout.addWidget(self.source_position)

        self.details_splitter = QSplitter(Qt.Horizontal)
        self.details_splitter.setChildrenCollapsible(False)
        self.details_splitter.addWidget(self.table)
        self.details_splitter.addWidget(self.detail_panel)
        self.details_splitter.setSizes([620, 380])
        self.graph_stack = QStackedWidget()
        self.graph_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.compact_tabs = QTabWidget()
        self.graph_stack.addWidget(self.trace_canvas)
        self.graph_stack.addWidget(self.compact_tabs)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.graph_stack)
        self.splitter.addWidget(self.details_splitter)
        self.splitter.setSizes([550, 450])
        layout.addWidget(self.splitter, 1)

        self.table.currentCellChanged.connect(self._row_changed)
        self.dataset_picker.currentIndexChanged.connect(self._dataset_chosen)
        self.point_number.valueChanged.connect(self._point_chosen)
        self.set_language()
        self.refresh(None, [], 0, "linear")

    def set_language(self):
        self.heading.setText(self._tr("Dataset"))
        self.heading.setBuddy(self.dataset_picker)
        self.dataset_picker.setAccessibleName(self._tr("Dataset"))
        self.point_label.setText(self._tr("Point"))
        self.point_number.setAccessibleName(self._tr("Point"))
        self.detail_title.setText(self._tr("Source detail"))
        self.source_detail.setAccessibleName(self._tr("Source detail"))
        self.trace_canvas.setAccessibleName(self._tr("Calibrated trace"))
        self.trace_canvas.setToolTip(self._tr("Click a trace point to locate it in the table and source image."))
        self.table.setHorizontalHeaderLabels([self._tr("Dataset"), "X", "Y"])
        if self._compact:
            self.compact_tabs.setTabText(0, self._tr("Calibrated trace"))
            self.compact_tabs.setTabText(1, self._tr("Source detail"))
        self._describe_selection()

    def refresh_palette(self):
        palette = self.palette()
        # The trace is populated just before its tab becomes enabled. Capturing
        # the disabled color group here would leave the data faint after entry.
        base = palette.color(QPalette.Active, QPalette.Base).name()
        text = palette.color(QPalette.Active, QPalette.Text).name()
        self.figure.set_facecolor(base)
        self.axes.set_facecolor(base)
        self.axes.tick_params(colors=text, labelsize=self._plot_font_size())
        for spine in self.axes.spines.values():
            spine.set_color(text)
        for axis in (self.axes.xaxis, self.axes.yaxis):
            axis.label.set_color(text)
            axis.label.set_size(self._plot_font_size())
            axis.get_offset_text().set_color(text)
        if self._line is not None:
            self._line.set_color(palette.color(QPalette.Active, QPalette.Highlight).name())
            self._marker.set_markerfacecolor(base)
            self._marker.set_markeredgecolor(text)
        self.trace_canvas.draw_idle()
        self.source_detail.update()

    def _plot_font_size(self):
        font = self.font()
        return font.pointSizeF() if font.pointSizeF() > 0 else max(8, font.pixelSize() * 72 / self.logicalDpiY())

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "source_detail") and event.type() in (QEvent.PaletteChange, QEvent.FontChange):
            self.refresh_palette()
        if hasattr(self, "graph_stack") and event.type() == QEvent.FontChange:
            self._arrange()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()

    def _arrange(self):
        compact = self.width() < max(420, self.fontMetrics().horizontalAdvance("0") * 43)
        if compact == self._compact:
            return
        self._compact = compact
        self.controls.removeWidget(self.dataset_picker)
        self.detail_title.setVisible(not compact)
        if compact:
            self.controls.addWidget(self.dataset_picker, 1, 0, 1, 3)
            self.compact_tabs.addTab(self.trace_canvas, self._tr("Calibrated trace"))
            self.compact_tabs.addTab(self.detail_panel, self._tr("Source detail"))
            self.graph_stack.setCurrentWidget(self.compact_tabs)
        else:
            self.controls.addWidget(self.dataset_picker, 0, 1)
            self.compact_tabs.removeTab(1)
            self.compact_tabs.removeTab(0)
            self.graph_stack.addWidget(self.trace_canvas)
            self.details_splitter.addWidget(self.detail_panel)
            self.detail_panel.show()
            self.graph_stack.setCurrentWidget(self.trace_canvas)
            self.details_splitter.setSizes([620, 380])

    @staticmethod
    def _pixel(dataset, index):
        pixels = dataset.get("pts_px", [])
        if index >= len(pixels):
            return None
        return tuple(pixels[index])

    def refresh(self, image, datasets, active_index, scale_type):
        """Update the table without losing a surviving point's identity after a refit or deletion."""
        same_image = image is self.source_detail.source
        previous_point = self.selection[1] if self.selection is not None else 0
        previous_dataset = self._selected_dataset
        previous_pixel = self._selected_pixel
        self._datasets = datasets
        self._scale = scale_type
        self._drawn_dataset = None
        self.source_detail.set_source(image)
        populated = [i for i, ds in enumerate(datasets) if len(ds.get("pts_real", []))]
        active = active_index if active_index in populated else next(iter(populated), None)
        if same_image and active_index == self._external_active:
            active = next((i for i in populated if datasets[i] is previous_dataset), active)
        self._external_active = active_index

        self._offsets = []
        size = sum(len(ds.get("pts_real", [])) for ds in datasets)
        with QSignalBlocker(self.dataset_picker), QSignalBlocker(self.table):
            self.dataset_picker.clear()
            self.table.setRowCount(size)
            row = 0
            for ds_index, dataset in enumerate(datasets):
                self._offsets.append(row)
                label = dataset.get("label", "")
                if ds_index in populated:
                    self.dataset_picker.addItem(label, ds_index)
                for point_index, (x, y) in enumerate(dataset.get("pts_real", [])):
                    name = QTableWidgetItem(label)
                    name.setData(Qt.UserRole, (ds_index, point_index))
                    self.table.setItem(row, 0, name)
                    for column, value in ((1, x), (2, y)):
                        number = QTableWidgetItem(f"{value:.6g}")
                        number.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        self.table.setItem(row, column, number)
                    row += 1
            self.table.setCurrentCell(-1, -1)
        self.dataset_picker.setEnabled(bool(populated))
        self.point_number.setEnabled(bool(populated))
        if active is None:
            self.selection = self._selected_dataset = self._selected_pixel = None
            self.source_detail.set_point(None)
            with QSignalBlocker(self.point_number):
                self.point_number.setRange(0, 0)
            self._plot_dataset(None)
            self._describe_selection()
            return

        current = datasets[active]
        point = 0
        if same_image and current is previous_dataset:
            point = min(previous_point, len(current["pts_real"]) - 1)
            if previous_pixel is not None and self._pixel(current, point) != previous_pixel:
                point = next((i for i, p in enumerate(current.get("pts_px", []))
                              if tuple(p) == previous_pixel and i < len(current["pts_real"])), point)
        with QSignalBlocker(self):
            self.select_point(active, point)

    def selected_point(self):
        return self.selection

    def select_point(self, dataset_index, point_index):
        if 0 <= dataset_index < len(self._datasets) and 0 <= point_index < len(self._datasets[dataset_index].get("pts_real", [])):
            self.table.setCurrentCell(self._offsets[dataset_index] + point_index, 1)

    def _dataset_chosen(self, combo_index):
        dataset_index = self.dataset_picker.itemData(combo_index)
        if dataset_index is not None:
            self.select_point(dataset_index, 0)

    def _point_chosen(self, point):
        if self.selection is not None:
            self.select_point(self.selection[0], point - 1)

    def _row_changed(self, row, column, previous_row, previous_column):
        if row < 0 or self.table.item(row, 0) is None:
            return
        dataset_index, point_index = self.table.item(row, 0).data(Qt.UserRole)
        dataset = self._datasets[dataset_index]
        changed = self._selected_dataset is not dataset or self.selection != (dataset_index, point_index)
        self.selection = (dataset_index, point_index)
        self._selected_dataset = dataset
        self._selected_pixel = self._pixel(dataset, point_index)
        if self._drawn_dataset != dataset_index:
            self._plot_dataset(dataset_index)
        if self._plottable[point_index]:
            x, y = self._trace[point_index]
            self._marker.set_data([x], [y])
        else:
            self._marker.set_data([], [])
        self.trace_canvas.draw_idle()
        self.source_detail.set_point(self._selected_pixel)
        with QSignalBlocker(self.point_number), QSignalBlocker(self.dataset_picker):
            self.point_number.setRange(1, len(self._trace))
            self.point_number.setValue(point_index + 1)
            self.dataset_picker.setCurrentIndex(self.dataset_picker.findData(dataset_index))
        self._describe_selection()
        if changed:
            self.pointSelected.emit(dataset_index, point_index)

    def _describe_selection(self):
        if self.selection is None:
            self.point_number.setSuffix("")
            self.source_position.setText(self._tr("Extract a curve to review its points."))
        else:
            dataset_index, _ = self.selection
            self.point_number.setSuffix(self._tr(" of {total}").format(total=len(self._datasets[dataset_index]["pts_real"])))
            pixel = self.source_detail.point
            self.source_position.setText(
                self._tr("Source pixel ({x}, {y})").format(x=f"{pixel[0]:.6g}", y=f"{pixel[1]:.6g}")
                if pixel is not None else self._tr("Source pixel unavailable")
            )
        self.source_detail.setAccessibleDescription(self.source_position.text())

    def _plot_dataset(self, index):
        self.axes.clear()
        self._drawn_dataset = index
        self._marker = self._line = None
        self._trace = np.empty((0, 2))
        self._plottable = np.zeros(0, dtype=bool)
        self.axes.set_axis_on()
        if index is None:
            self.axes.set_axis_off()
        else:
            self._trace = np.asarray(self._datasets[index]["pts_real"], dtype=float)
            self._plottable = np.isfinite(self._trace).all(axis=1)
            if self._scale in ("semilogx", "loglog"):
                self.axes.set_xscale("log")
                self._plottable &= self._trace[:, 0] > 0
            if self._scale in ("semilogy", "loglog"):
                self.axes.set_yscale("log")
                self._plottable &= self._trace[:, 1] > 0
            x, y = np.where(self._plottable[:, None], self._trace, np.nan).T
            if not self._plottable.any():
                # Keep all original table rows and their pixels when a chosen
                # log axis cannot represent any of the recorded values.
                if self.axes.get_xscale() == "log":
                    self.axes.set_xlim(1, 10)
                if self.axes.get_yscale() == "log":
                    self.axes.set_ylim(1, 10)
            dataset = self._datasets[index]
            if dataset.get("kind") == "curve" and len(dataset["pts_px"]) == len(x):
                breaks = curve_breaks(dataset["pts_px"])
                x, y = np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)
            self._line, = self.axes.plot(x, y, linewidth=1.5, marker="." if len(x) == 1 else None)
            self._marker, = self.axes.plot([], [], linestyle="none", marker="o", markersize=8, markeredgewidth=2)
            self.axes.set_xlabel("X")
            self.axes.set_ylabel("Y")
        self.refresh_palette()

    def _pick_trace(self, event):
        if event.inaxes is not self.axes or event.button != 1 or self._drawn_dataset is None:
            return
        # Display distances respect log scales and the plot's unequal X/Y units.
        indices = np.flatnonzero(self._plottable)
        if indices.size:
            screen = self.axes.transData.transform(self._trace[indices])
            distances = np.square(screen - (event.x, event.y)).sum(axis=1)
            self.select_point(self._drawn_dataset, int(indices[np.argmin(distances)]))
