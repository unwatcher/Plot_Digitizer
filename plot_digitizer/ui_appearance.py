"""Readable native Qt controls with font-relative spacing and semantic colors."""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QMenu, QTableView, QTableWidget


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


def digitizer_palette(scheme: str, base_palette: QPalette) -> QPalette:
    """Build a window-local palette. System follows the caller's native colors."""
    palette = QPalette(base_palette)
    if scheme not in ("paper", "graphite"):
        return palette

    # Set both light and dark roles explicitly, including the shades used by
    # Fusion's native spin buttons, combo boxes, and disabled selections.
    graphite = scheme == "graphite"
    for roles, paper, dark in (
        ((QPalette.Window, QPalette.Base), "#ffffff", "#18242d"),
        ((QPalette.AlternateBase, QPalette.Button, QPalette.ToolTipBase), "#f3f6f6", "#20303a"),
        ((QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.ToolTipText), "#172b34", "#f2f6f8"),
        ((QPalette.Highlight, QPalette.Accent, QPalette.Link), "#005e62", "#86e1d4"),
        ((QPalette.HighlightedText,), "#ffffff", "#102a30"),
        ((QPalette.PlaceholderText,), "#485e67", "#bfced4"),
        ((QPalette.LinkVisited,), "#54417f", "#d8b9f3"),
        ((QPalette.Mid,), "#73848a", "#82979e"),
        ((QPalette.Light,), "#ffffff", "#50646d"),
        ((QPalette.Midlight,), "#cbd5d6", "#43565f"),
        ((QPalette.Dark,), "#66787f", "#101a21"),
        ((QPalette.Shadow,), "#485e67", "#0c151b"),
        ((QPalette.BrightText,), "#ffffff", "#ffffff"),
    ):
        color = QColor(dark if graphite else paper)
        for role in roles:
            palette.setColor(role, color)

    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText, QPalette.PlaceholderText):
        palette.setColor(QPalette.Disabled, role, QColor("#a6b9c1" if graphite else "#536870"))
    for role in (QPalette.Highlight, QPalette.Accent):
        palette.setColor(QPalette.Disabled, role, QColor("#43565f" if graphite else "#cbd5d6"))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor("#e1e9ed" if graphite else "#41565e"))
    return palette


def highscore_palette(base_palette: QPalette | None = None) -> QPalette:
    """Build the consistent HighScore Precision cool-slate light palette (no Windows dark-mode adaptation)."""
    palette = QPalette(base_palette) if base_palette is not None else QPalette()
    for roles, hex_color in (
        ((QPalette.Window,), "#dce4ee"),
        ((QPalette.Base,), "#ffffff"),
        ((QPalette.AlternateBase,), "#f2f6fb"),
        ((QPalette.Button,), "#ffffff"),
        ((QPalette.ToolTipBase,), "#0b5cad"),
        ((QPalette.ToolTipText,), "#ffffff"),
        ((QPalette.WindowText, QPalette.Text, QPalette.ButtonText), "#0f172a"),
        ((QPalette.Highlight, QPalette.Accent, QPalette.Link), "#0b5cad"),
        ((QPalette.HighlightedText,), "#ffffff"),
        ((QPalette.PlaceholderText,), "#526b84"),
        ((QPalette.Mid,), "#7d93aa"),
        ((QPalette.Midlight,), "#c5d2df"),
        ((QPalette.Dark,), "#556b82"),
        ((QPalette.Light,), "#ffffff"),
        ((QPalette.Shadow,), "#3f5368"),
        ((QPalette.BrightText,), "#ffffff"),
    ):
        color = QColor(hex_color)
        for role in roles:
            palette.setColor(role, color)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText, QPalette.PlaceholderText):
        palette.setColor(QPalette.Disabled, role, QColor("#64748b"))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor("#edf2f7"))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#cbd5e1"))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor("#475569"))
    return palette


def configure_workstation_theme(app: QApplication) -> None:
    """Force the consistent HighScore Precision light color scheme and crisp un-squished font."""
    QApplication.setAttribute(Qt.AA_UseStyleSheetPropagationInWidgetStyles, True)
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    font = QFont(app.font())
    if sys.platform == "win32" and "Segoe UI" in QFontDatabase.families():
        font.setFamily("Segoe UI")
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    app.setFont(font)
    app.setPalette(highscore_palette(app.palette()))


def semantic_colors(palette: QPalette, tone: str) -> tuple[QColor, QColor]:
    """Return an explicit background/text pair for light or dark UI surfaces."""
    dark = palette.color(QPalette.Base).lightnessF() < 0.5
    colors = {
        "success": ("#dcfce7", "#14532d", "#183d2b", "#b8f0ce"),
        "warning": ("#fef3c7", "#78350f", "#493916", "#ffe4a3"),
        "error": ("#fee2e2", "#881337", "#49252a", "#ffc1c5"),
        "info": ("#e0f2fe", "#0c4a6e", "#183c55", "#c6e9ff"),
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


def window_stylesheet(palette: QPalette, *, digitizer_colors: bool = False, highscore_theme: bool = False) -> str:
    window = palette.color(QPalette.Window)
    base = palette.color(QPalette.Base)
    text = palette.color(QPalette.WindowText)
    dark = base.lightnessF() < 0.5
    explicit = digitizer_colors or highscore_theme
    outline = QColor("#9fb0c2") if (highscore_theme and not dark) else (
        palette.color(QPalette.Mid) if digitizer_colors else _blend_color(window, text, 0.24 if dark else 0.18)
    )
    section_outline = QColor("#b8c6d4") if (highscore_theme and not dark) else _blend_color(window, text, 0.16 if dark else 0.11)
    section = QColor("#f5f8fb") if (highscore_theme and not dark) else _blend_color(window, base, 0.16 if dark else 0.28)
    chrome = QColor("#eaf0f6") if (highscore_theme and not dark) else _blend_color(window, base, 0.08 if dark else 0.12)
    rail_bg = QColor("#dfe6ee") if (highscore_theme and not dark) else _blend_color(window, text, 0.06)
    button = QColor("#ffffff") if (highscore_theme and not dark) else _blend_color(palette.color(QPalette.Button), base, 0.12 if dark else 0.55)
    accent = palette.color(QPalette.Highlight)
    accent_text = readable_foreground(accent, palette.color(QPalette.HighlightedText))
    accent_soft = QColor("#d9e7f7") if (highscore_theme and not dark) else _blend_color(base, accent, 0.18)
    button_hover = _blend_color(button, accent, 0.14 if dark else 0.10)
    button_pressed = _blend_color(button, accent, 0.24 if dark else 0.18)
    hover = accent.darker(115) if accent_text.lightnessF() > 0.5 else accent.lighter(115)
    pressed = accent.darker(128) if accent_text.lightnessF() > 0.5 else accent.lighter(128)
    disabled_text = palette.color(QPalette.Disabled, QPalette.ButtonText).name()
    button_text = palette.color(QPalette.ButtonText).name() if explicit else "palette(button-text)"
    focus = accent.name() if explicit else "palette(highlight)"
    disabled_button = palette.color(QPalette.Disabled, QPalette.Button).name() if explicit else "palette(button)"
    disabled_border = palette.color(QPalette.Disabled, QPalette.Midlight).name() if explicit else "palette(midlight)"
    rules = [f"""
        QPushButton {{ padding: 0.28em 0.68em; min-height: 1.4em;
                      border: 1px solid {outline.name()}; border-radius: 4px;
                      background-color: {button.name()}; color: {button_text}; }}
        QPushButton:hover:enabled {{ border-color: {focus};
                                    background-color: {button_hover.name()}; }}
        QPushButton:pressed:enabled {{ background-color: {button_pressed.name()}; }}
        QPushButton:focus {{ border-color: {focus}; border-style: dashed; }}
        QPushButton:disabled {{ background-color: {disabled_button};
                               color: {disabled_text}; border-color: {section_outline.name()}; }}
        QLineEdit, QComboBox, QAbstractSpinBox {{ min-height: 1.5em;
            border: 1px solid {outline.name()}; border-radius: 4px;
            background-color: {base.name()}; color: {text.name()}; padding: 0.1em 0.35em; }}
        QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {focus}; }}
        QGroupBox {{ margin-top: 0.85em; padding: 0.55em 0.55em 0.45em;
                     border: 1px solid {section_outline.name()}; border-radius: 6px;
                     background-color: {section.name()}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 0.6em; padding: 0 0.3em;
                            font-weight: 600;
                            color: {accent.name() if (highscore_theme and not dark) else text.name()}; }}
        QTabBar::tab {{ padding: 0.35em 0.8em; }}
        QHeaderView::section {{ padding: 0.35em 0.5em; background-color: {chrome.name()};
                               color: {button_text}; border: none; font-weight: 600;
                               border-bottom: 1px solid {section_outline.name()}; }}
        QTableView, QTreeView {{ border: 1px solid {section_outline.name()};
                      background-color: {base.name()};
                      gridline-color: {section_outline.name()}; selection-color: {accent_text.name()}; }}
        QTableView::item, QTreeView::item {{ padding: 0.18em 0.35em; }}
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
        border = outline if digitizer_colors else _blend_color(background, foreground, 0.30)
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
        {disabled_selectors} {{ background-color: {disabled_button}; color: {disabled_text};
                               border-color: {disabled_border}; }}
    """)
    if highscore_theme:
        soft_sel = "#d6e7fa"
        soft_border = "#6ea3dc"
        crisp_border = "#889eb5"
        crisp_bottom = "#6f87a0"
        panel_bg = "#eef3f8"
        divider = "#9eb1c5"
        rules.append(f"""
            QMainWindow, QDialog, QDockWidget {{ background-color: {window.name()}; color: {text.name()}; }}
            QToolBar#unifiedTopToolBar {{ background-color: #e9eff6; border: none;
                                          border-bottom: 1px solid {divider};
                                          padding: 0px; margin: 0px; }}
            QFrame#unifiedHeader {{ background-color: #e9eff6; border: none; }}
            QLabel#logoBadge {{ background-color: {accent.name()}; color: {accent_text.name()};
                                border-radius: 7px; padding: 2px 7px; font-weight: 800; }}
            QLabel#brandTitle {{ color: {text.name()}; font-weight: 700; padding-right: 6px; }}
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget, QStackedWidget {{
                background-color: {window.name()}; color: {text.name()}; }}
            QDockWidget {{ border-left: 1px solid {divider}; }}
            QFrame#catalogPaneFrame {{ background-color: {panel_bg}; border: 1px solid {divider};
                                       border-radius: 8px; }}
            QLabel {{ color: {text.name()}; background: transparent; }}
            QCheckBox, QRadioButton {{ color: {text.name()}; spacing: 6px; }}
            QPushButton {{ padding: 0.24em 0.72em; min-height: 1.35em;
                           border: 1px solid {crisp_border}; border-bottom-color: {crisp_bottom};
                           border-radius: 8px;
                           background-color: #ffffff; color: {text.name()}; }}
            QPushButton:hover:enabled {{ border-color: {accent.name()}; background-color: #ebf3fc; }}
            QPushButton:pressed:enabled {{ background-color: {soft_sel}; border-color: {accent.name()}; }}
            QPushButton#primary {{ background-color: {accent.name()}; color: {accent_text.name()};
                                   border: 1px solid #08488a; border-radius: 8px; font-weight: 600; }}
            QPushButton#primary:hover:enabled {{ background-color: {hover.name()}; border-color: {hover.name()}; }}
            QPushButton#primary:pressed:enabled {{ background-color: {pressed.name()}; }}
            QPushButton#ghostButton {{ background-color: transparent; border: 1px solid transparent;
                                       border-radius: 7px; padding: 0.22em 0.6em; color: {text.name()}; }}
            QPushButton#ghostButton:hover:enabled {{ background-color: #dce7f4; border-color: {crisp_border}; }}
            QPushButton#ghostButton::menu-indicator {{ width: 0px; }}
            QLineEdit, QComboBox, QAbstractSpinBox {{
                min-height: 1.55em; padding: 0.14em 0.5em;
                background-color: #ffffff; color: {text.name()};
                border: 1px solid {crisp_border}; border-radius: 7px; }}
            QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{
                border: 1.5px solid {accent.name()}; background-color: #ffffff; }}
            QComboBox QAbstractItemView {{
                background-color: #ffffff; color: {text.name()};
                selection-background-color: {soft_sel}; selection-color: {text.name()};
                border: 1px solid {crisp_border}; border-radius: 6px; }}
            QPlainTextEdit, QTextEdit, QTextBrowser, QListWidget {{
                background-color: #ffffff; color: {text.name()};
                border: 1px solid {divider}; border-radius: 7px; }}
            QListWidget::item {{ padding: 6px 9px; border-radius: 6px; }}
            QListWidget::item:selected {{ background-color: {soft_sel}; color: {accent.name()}; font-weight: 600; }}
            QMenuBar {{ background-color: #e9eff6; color: {text.name()};
                        border-bottom: 1px solid {divider}; }}
            QMenu {{ background-color: #ffffff; color: {text.name()};
                     border: 1px solid {crisp_border}; border-radius: 7px; padding: 4px; }}
            QMenu::item {{ padding: 5px 16px; border-radius: 5px; }}
            QMenu::item:selected {{ background-color: {soft_sel}; color: {text.name()}; }}
            QStatusBar {{ background-color: #e9eff6; color: #334e68;
                          border-top: 1px solid {divider}; }}
            QFrame#activityRail {{ background-color: #d6e0ec; border-right: 1px solid {divider}; }}
            QPushButton#railButton {{ background-color: transparent; border: 1px solid transparent;
                                      border-radius: 8px; padding: 0.35em 0.3em; font-weight: 600;
                                      color: #2c435a; }}
            QPushButton#railButton:hover:enabled {{ background-color: #e4edf7; border-color: {crisp_border}; color: {text.name()}; }}
            QPushButton#railButton:checked {{ background-color: #ffffff; color: {accent.name()};
                                              border: 1px solid {soft_border}; }}
            QFrame#modeSwitchFrame {{ background-color: #d4dfec; border: 1px solid {crisp_border};
                                      border-radius: 10px; }}
            QPushButton#modeButton {{ padding: 0.18em 0.62em; min-height: 1.3em; font-weight: 600;
                                      border: 1px solid transparent; background-color: transparent; color: #2c435a;
                                      border-radius: 8px; }}
            QPushButton#modeButton:hover:enabled {{ color: {text.name()}; background-color: #e8f0fa; }}
            QPushButton#modeButton:checked {{ background-color: {accent.name()}; color: {accent_text.name()};
                                              border-color: #08488a; }}
            QFrame#spectrumHeader {{ background-color: #e9eff6; border: 1px solid {divider};
                                     border-bottom: none; border-top-left-radius: 7px; border-top-right-radius: 7px; }}
            QFrame#plotToolbar {{ background-color: {panel_bg}; border-left: 1px solid {divider};
                                  border-right: 1px solid {divider};
                                  border-bottom: 1px solid {divider}; }}
            QPushButton#plotToggle {{ padding: 0.14em 0.52em; min-height: 1.15em; font-weight: 600;
                                      border-radius: 6px; border: 1px solid {crisp_border};
                                      background-color: #ffffff; color: #2c435a; }}
            QPushButton#plotToggle:hover:enabled {{ border-color: {accent.name()}; background-color: #ebf3fc; color: {text.name()}; }}
            QPushButton#plotToggle:checked {{ background-color: {accent.name()}; color: {accent_text.name()};
                                              border-color: #08488a; }}
            QPushButton#promptChip {{ padding: 0.18em 0.7em; min-height: 1.2em; border-radius: 12px;
                                      background-color: #ffffff; color: #2c435a;
                                      border: 1px solid {crisp_border}; }}
            QPushButton#promptChip:hover:enabled {{ background-color: {soft_sel}; color: {accent.name()}; border-color: {accent.name()}; }}
            QLabel#stepPill {{ padding: 0.18em 0.45em; border-radius: 6px; font-weight: 600;
                               background-color: #ffffff; color: #47617a;
                               border: 1px solid {divider}; }}
            QLabel#stepPill[stepState="done"] {{ background-color: #dcfce7; color: #14532d; border-color: #6ee7b7; }}
            QLabel#stepPill[stepState="current"] {{ background-color: {accent.name()}; color: {accent_text.name()}; border-color: #08488a; }}
            QLabel#kpiChip {{ background-color: #ffffff; color: {text.name()};
                              border: 1px solid {crisp_border};
                              border-radius: 6px; padding: 0.14em 0.52em; font-weight: 600; }}
            QLabel#sectionEyebrow {{ color: #1e3a5f; font-weight: 700; letter-spacing: 0.04em; }}
            QTabWidget::pane {{ border: 1px solid {divider}; border-radius: 7px; background-color: {panel_bg}; }}
            QTabBar::tab {{ background-color: #dfe7f1; color: #334e68;
                            padding: 0.34em 0.85em;
                            border: 1px solid {divider}; border-bottom: none;
                            border-top-left-radius: 7px; border-top-right-radius: 7px; margin-right: 3px; }}
            QTabBar::tab:hover:!selected {{ background-color: #e9f1fa; color: {text.name()}; }}
            QTabBar::tab:selected {{ background-color: #ffffff; color: {accent.name()};
                                     font-weight: 600; border-top: 2px solid {accent.name()}; }}
            QSplitter::handle {{ background-color: {divider}; }}
            QTreeWidget#catalogTree {{ background-color: {panel_bg}; border: none; outline: 0; }}
            QTreeWidget#catalogTree::item {{ border: none; padding: 0px; }}
            QTreeWidget#catalogTree::item:selected {{ background: transparent; }}
            QTableWidget {{ background-color: #ffffff; alternate-background-color: #f4f8fc;
                            color: {text.name()}; border: 1px solid {divider}; border-radius: 6px; }}
            QTableWidget::item:selected {{ background-color: {soft_sel}; color: {text.name()}; }}
            QGroupBox {{ border: 1px solid {divider}; border-radius: 8px; background-color: #ffffff;
                         margin-top: 1.65em; padding: 0.9em 0.7em 0.6em; font-weight: 700; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                                left: 10px; top: 0.15em; padding: 1px 6px;
                                font-weight: 700; color: {accent.name()}; background-color: transparent; }}
            QGroupBox QLabel, QGroupBox QLineEdit, QGroupBox QComboBox, QGroupBox QAbstractSpinBox,
            QGroupBox QPlainTextEdit, QGroupBox QCheckBox, QGroupBox QRadioButton,
            QGroupBox QAbstractItemView, QGroupBox QPushButton {{ font-weight: 400; }}
            QGroupBox QPushButton#primary, QGroupBox QPushButton#plotToggle {{ font-weight: 600; }}
            QScrollBar:vertical, QScrollBar:horizontal {{ background: #e4ebf3; border: none; }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background: #9db0c4; border-radius: 5px; min-height: 24px; min-width: 24px; }}
            QScrollBar::handle:hover {{ background: {accent.name()}; }}
        """)
    if digitizer_colors:
        rules.append(f"""
            QTableView {{ selection-background-color: {accent.name()}; selection-color: {accent_text.name()}; }}
            QTableView:disabled {{ selection-background-color: {palette.color(QPalette.Disabled, QPalette.Highlight).name()};
                                   selection-color: {palette.color(QPalette.Disabled, QPalette.HighlightedText).name()}; }}
            QTabBar::tab:selected {{ color: {text.name()}; background: {base.name()};
                                     border-bottom: 2px solid {accent.name()}; }}
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
        owner = self.window
        while owner is not None and owner.property("digitizerScheme") is None:
            owner = owner.parentWidget()
        digitizer_colors = owner is not None and owner.property("digitizerScheme") in ("paper", "graphite")
        highscore_theme = self.window.property("workstationTheme") == "highscore"
        if digitizer_colors or highscore_theme:
            for menu in self.window.findChildren(QMenu):
                menu.setAttribute(Qt.WA_WindowPropagation)
        stylesheet = window_stylesheet(
            self.window.palette(),
            digitizer_colors=digitizer_colors,
            highscore_theme=highscore_theme,
        )
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
