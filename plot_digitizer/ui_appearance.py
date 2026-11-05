"""Readable native Qt controls with font-relative spacing and semantic colors."""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QTableView, QTableWidget


ITEM_TONE_ROLE = int(Qt.UserRole) + 538


def contrast_ratio(first: QColor, second: QColor) -> float:
    def luminance(color):
        channels = [color.redF(), color.greenF(), color.blueF()]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                  for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def readable_foreground(background: QColor, preferred: QColor) -> QColor:
    if contrast_ratio(background, preferred) >= 4.5:
        return QColor(preferred)
    return max((QColor("#000000"), QColor("#ffffff")),
               key=lambda color: contrast_ratio(background, color))


def _blend_color(first: QColor, second: QColor, amount: float) -> QColor:
    return QColor.fromRgbF(
        first.redF() * (1 - amount) + second.redF() * amount,
        first.greenF() * (1 - amount) + second.greenF() * amount,
        first.blueF() * (1 - amount) + second.blueF() * amount,
        first.alphaF() * (1 - amount) + second.alphaF() * amount,
    )


def soften_default_palette(palette: QPalette) -> QPalette:
    """Gently separate native surfaces without changing explicit user palettes."""
    softened = QPalette(palette)
    # Qt's native application palette has no explicitly resolved roles. A
    # custom palette, or a palette already adjusted here, must remain intact.
    if palette.resolveMask() != 0:
        return softened
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        base = palette.color(group, QPalette.Base)
        window = palette.color(group, QPalette.Window)
        dark = base.lightnessF() < 0.5
        surface = _blend_color(window, base, 0.10 if dark else 0.15)
        if not dark:
            surface.setRed(min(255, surface.red() + 1))
            surface.setBlue(max(0, surface.blue() - 1))
        softened.setColor(group, QPalette.Window, surface)
        softened.setColor(group, QPalette.AlternateBase, _blend_color(
            palette.color(group, QPalette.AlternateBase), surface, 0.20,
        ))
    for group in (QPalette.Active, QPalette.Inactive):
        softened.setColor(group, QPalette.HighlightedText, readable_foreground(
            palette.color(group, QPalette.Highlight), palette.color(group, QPalette.HighlightedText),
        ))
    return softened


def semantic_colors(palette: QPalette, tone: str) -> tuple[QColor, QColor]:
    """Return an explicit background/text pair for light or dark UI surfaces."""
    dark = palette.color(QPalette.Base).lightnessF() < 0.5
    colors = {
        "success": ("#e7f3ec", "#175535", "#183d2b", "#b8f0ce"),
        "warning": ("#fff2cc", "#714b00", "#493916", "#ffe4a3"),
        "error": ("#fdebed", "#9e1b32", "#49252a", "#ffc1c5"),
        "info": ("#e5f1fa", "#174c70", "#183c55", "#c6e9ff"),
    }
    pair = colors[tone][2:] if dark else colors[tone][:2]
    return QColor(pair[0]), QColor(pair[1])


def set_item_tone(item, tone: str, palette: QPalette) -> None:
    background, foreground = semantic_colors(palette, tone)
    item.setBackground(background)
    item.setForeground(foreground)
    item.setData(ITEM_TONE_ROLE, tone)


def set_status_tone(widget, tone: str | None = None) -> None:
    widget.setProperty("statusTone", tone or "")
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.updateGeometry()
    widget.update()


def window_stylesheet(palette: QPalette) -> str:
    window = palette.color(QPalette.Window)
    base = palette.color(QPalette.Base)
    text = palette.color(QPalette.WindowText)
    dark = base.lightnessF() < 0.5
    outline = _blend_color(window, text, 0.24 if dark else 0.18)
    section_outline = _blend_color(window, text, 0.16 if dark else 0.11)
    section = _blend_color(window, base, 0.16 if dark else 0.28)
    button = _blend_color(palette.color(QPalette.Button), base, 0.12 if dark else 0.55)
    accent = palette.color(QPalette.Highlight)
    accent_text = readable_foreground(accent, palette.color(QPalette.HighlightedText))
    button_hover = _blend_color(button, accent, 0.14 if dark else 0.07)
    button_pressed = _blend_color(button, accent, 0.24 if dark else 0.14)
    hover = accent.darker(112) if accent_text.lightnessF() > 0.5 else accent.lighter(112)
    pressed = accent.darker(125) if accent_text.lightnessF() > 0.5 else accent.lighter(125)
    disabled_text = palette.color(QPalette.Disabled, QPalette.ButtonText).name()
    rules = [f"""
        QPushButton {{ padding: 0.3em 0.7em; min-height: 1.4em;
                      border: 1px solid {outline.name()}; border-radius: 4px;
                      background-color: {button.name()}; color: palette(button-text); }}
        QPushButton:hover:enabled {{ border-color: palette(highlight);
                                    background-color: {button_hover.name()}; }}
        QPushButton:pressed:enabled {{ background-color: {button_pressed.name()}; }}
        QPushButton:focus {{ border-color: palette(highlight); border-style: dashed; }}
        QPushButton:disabled {{ background-color: palette(button);
                               color: {disabled_text}; border-color: {section_outline.name()}; }}
        QLineEdit, QComboBox, QAbstractSpinBox {{ min-height: 1.5em; }}
        QGroupBox {{ margin-top: 0.85em; padding: 0.6em 0.5em 0.45em;
                     border: 1px solid {section_outline.name()}; border-radius: 8px;
                     background-color: {section.name()}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 0.6em; padding: 0 0.25em; }}
        QTabBar::tab {{ padding: 0.35em 0.8em; }}
        QHeaderView::section {{ padding: 0.35em 0.5em; background-color: {section.name()};
                               color: palette(button-text); border: none;
                               border-bottom: 1px solid {section_outline.name()}; }}
        QTableView {{ border: 1px solid {section_outline.name()};
                      gridline-color: {section_outline.name()}; selection-color: {accent_text.name()}; }}
        QTableView::item {{ padding: 0.15em 0.35em; }}
        QScrollArea {{ border: none; }}
    """]
    primary = "QPushButton#primary, QPushButton#extract"
    rules.append(f"""
        {primary} {{ background-color: {accent.name()}; color: {accent_text.name()};
                     border-color: {accent.name()}; font-weight: 600; }}
        QPushButton#primary:hover:enabled, QPushButton#extract:hover:enabled {{
            background-color: {hover.name()}; }}
        QPushButton#primary:pressed:enabled, QPushButton#extract:pressed:enabled {{
            background-color: {pressed.name()}; }}
        QPushButton#primary:focus, QPushButton#extract:focus {{
            border-color: {accent_text.name()}; border-style: dashed; }}
    """)
    for tone in ("success", "warning", "error", "info"):
        background, foreground = semantic_colors(palette, tone)
        rules.append(f"""
            QLabel[statusTone="{tone}"] {{ background-color: {background.name()};
                color: {foreground.name()}; padding: 0.3em 0.45em; border-radius: 4px; }}
        """)
    for name, tone in (("danger", "error"), ("xbtn", "warning"), ("ybtn", "info"), ("roi", "success")):
        background, foreground = semantic_colors(palette, tone)
        border = _blend_color(background, foreground, 0.30)
        special_hover = _blend_color(background, foreground, 0.06)
        rules.append(f"""
            QPushButton#{name} {{ background-color: {background.name()};
                color: {foreground.name()}; border-color: {border.name()}; }}
            QPushButton#{name}:hover:enabled {{ background-color: {special_hover.name()}; }}
            QPushButton#{name}:focus {{ border-color: {foreground.name()}; border-style: dashed; }}
        """)
    disabled_selectors = ", ".join(f"QPushButton#{name}:disabled"
                                   for name in ("primary", "extract", "danger", "xbtn", "ybtn", "roi"))
    rules.append(f"""
        {disabled_selectors} {{ background-color: palette(button); color: {disabled_text};
                               border-color: palette(midlight); }}
    """)
    return "\n".join(rules)


class _AppearanceUpdater(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Show, QEvent.FontChange, QEvent.ApplicationPaletteChange, QEvent.PaletteChange):
            self.timer.start(0)
        return False

    def refresh(self):
        stylesheet = window_stylesheet(self.window.palette())
        # Qt stylesheets resolve control fonts against the application font,
        # rather than inheriting a window's enlarged accessibility font.
        # Keep the window itself outside this rule so setFont() remains usable.
        font = self.window.font()
        size = f"{font.pointSizeF():g}pt" if font.pointSizeF() > 0 else f"{font.pixelSize()}px"
        stylesheet += f"""
            QAbstractButton, QLabel, QLineEdit, QComboBox, QAbstractSpinBox,
            QGroupBox, QAbstractItemView, QPlainTextEdit, QTextEdit, QMenuBar,
            QMenu, QTabBar, QStatusBar, PatternPlot, PlotCanvas {{ font-size: {size}; }}
        """
        if self.window.styleSheet() != stylesheet:
            self.window.setStyleSheet(stylesheet)
        for table in self.window.findChildren(QTableView):
            height = max(28, table.fontMetrics().lineSpacing() + 12)
            table.verticalHeader().setMinimumSectionSize(height)
            table.verticalHeader().setDefaultSectionSize(height)
            if isinstance(table, QTableWidget):
                for row in range(table.rowCount()):
                    for column in range(table.columnCount()):
                        item = table.item(row, column)
                        tone = item.data(ITEM_TONE_ROLE) if item else None
                        if tone:
                            set_item_tone(item, tone, self.window.palette())


def apply_window_appearance(window) -> None:
    QApplication.setAttribute(Qt.AA_UseStyleSheetPropagationInWidgetStyles, True)
    if not hasattr(window, "_appearance_updater"):
        window._appearance_updater = _AppearanceUpdater(window)
    window._appearance_updater.refresh()


def configure_application_appearance(app: QApplication) -> None:
    """Keep native fonts and palettes, with a readable initial body-text size."""
    QApplication.setAttribute(Qt.AA_UseStyleSheetPropagationInWidgetStyles, True)
    font = QFont(app.font())
    # Windows' offscreen plugin resolves generic Sans Serif to Agency FB.
    # Use the native UI face for representative screenshots and layout checks.
    if sys.platform == "win32" and app.platformName() in {"offscreen", "minimal"}:
        if "Segoe UI" in QFontDatabase.families():
            font.setFamily("Segoe UI")
    if font.pointSizeF() > 0 and font.pointSizeF() < 10.5:
        font.setPointSizeF(10.5)
    app.setFont(font)
    palette = soften_default_palette(app.palette())
    if palette != app.palette():
        app.setPalette(palette)
