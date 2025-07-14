from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from vidio.read import OpenCVReader

from video_embedding.utils import EmbeddingStore
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers


class VideoPlayer(QWidget):

    def __init__(self, width=160):
        super().__init__()

        self.video_label = QLabel()
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)

        self.video_info = None  # [path, start, end]
        self.video_array = None
        self.current_frame = None
        self.fps = None
        self.width = width
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video_label)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_frame(self):
        if self.video_array is None:
            return
        frame = self.video_array[self.current_frame]
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.current_frame = (self.current_frame + 1) % len(self.video_array)

    def load_video(self, video_info):
        self.clear_video()
        path, start, end = video_info
        reader = OpenCVReader(path)
        self.video_array = [reader[i] for i in range(start, end)]
        self.fps = reader.fps
        self.current_frame = 0
        self._resize_label()
        self.frame_timer.start(1000 // self.fps)

    def clear_video(self):
        self.frame_timer.stop()
        self.video_label.clear()
        self.video_array = self.current_frame = None

    def set_background_color(self, color):
        palette = self.palette()
        palette.setColor(QPalette.Window, color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def set_width(self, w):
        self.width = w
        self._resize_label()

    def _resize_label(self):
        if self.video_array is None:
            ar = 9 / 16  # provisional aspect ratio
        else:
            h_px, w_px, _ = self.video_array[0].shape
            ar = h_px / w_px
        h = int(round(self.width * ar))
        self.setFixedSize(self.width, h)
        self.video_label.setFixedSize(self.width, h)


class VideoPopup(QWidget):
    def __init__(self, video_info, label, main_window):
        super().__init__(None, Qt.ToolTip)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.main = main_window

        # inner player
        self.player = VideoPlayer(width=main_window.clip_width)
        self.player.load_video(video_info)

        # title
        title = QLabel(label)
        title.setStyleSheet("color:white; font-weight:600;")

        # header layout
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(title)
        header.addStretch()

        # widget layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.player)

        main_window.clip_width_changed.connect(self.set_width)
        self.set_width(main_window.clip_width)
        self.adjustSize()

    def mousePressEvent(self, ev):
        self.close()

    def closeEvent(self, ev):
        self.player.clear_video()
        self.main._popup_closed(self)
        super().closeEvent(ev)

    @Slot(int)
    def set_width(self, w):
        self.player.set_width(w)
        self.adjustSize()
        self.setFixedWidth(w)


class VariableSelectDialog(QDialog):
    DEFAULT_X = "UMAP x"
    DEFAULT_Y = "UMAP y"

    def __init__(self, numeric_columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Variables")
        layout = QVBoxLayout(self)

        self.x_combo = QComboBox(self)
        self.x_combo.addItems(numeric_columns)
        layout.addWidget(QLabel("X-axis:"))
        layout.addWidget(self.x_combo)

        self.y_combo = QComboBox(self)
        self.y_combo.addItems(numeric_columns)
        layout.addWidget(QLabel("Y-axis:"))
        layout.addWidget(self.y_combo)

        if self.DEFAULT_X in numeric_columns:
            self.x_combo.setCurrentText(self.DEFAULT_X)
        if self.DEFAULT_Y in numeric_columns:
            self.y_combo.setCurrentText(self.DEFAULT_Y)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self) -> tuple[str, str]:
        return (self.x_combo.currentText(), self.y_combo.currentText())


class ScatterPlot(QWidget):
    point_pressed = Signal(int)
    point_released = Signal(int, Qt.KeyboardModifiers)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._points = np.zeros((0, 2))
        self._point_size = 12
        self._face_colors = None  # pre-computed RGBA (N,4)
        self._values = None  # original column – kept only for re-use
        self._discrete = False

        layout = QVBoxLayout(self)
        self.canvas = SceneCanvas(keys="interactive", show=False)
        self.canvas.native.setMouseTracking(True)
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.scatter = Markers(parent=self.view.scene)
        self.view.add(self.scatter)

        self.view.events.mouse_press.connect(self._on_mouse_press)
        self.view.events.mouse_release.connect(self._on_mouse_release)

    def set_data(self, points: np.ndarray):
        """Replace the XY coordinates of every point."""
        self._points = np.asarray(points, float)
        # force colours to have same length as points
        self._face_colors = None
        self._update_plot(auto_range=True)

    def set_point_size(self, size: int):
        self._point_size = size
        self._update_plot(auto_range=False)

    def set_point_colors(self, values: pd.Series | None = None):
        """
        Pre-compute a colour for every point and cache it.
        • None         → uniform grey
        • categorical  → tab20
        • numeric      → viridis
        """
        if values is None:
            self._values = None
            self._face_colors = None  # will fall back to grey in _update_plot
            self._update_plot(auto_range=False)
            return

        self._values = values
        self._discrete = values.dtype.kind in ("O", "U", "b", "u")

        if self._discrete:  # categorical → tab20
            cats, inv = np.unique(values, return_inverse=True)
            cmap = plt.get_cmap("tab20")
            self._face_colors = cmap(np.linspace(0, 1, len(cats)))[inv]
        else:  # numeric → viridis
            v = values.astype(float).to_numpy()
            t = (v - v.min()) / (v.max() - v.min() + 1e-12)
            self._face_colors = plt.get_cmap("viridis")(t)

        self._update_plot(auto_range=False)

    def _update_plot(self, *, auto_range: bool):
        """Push the cached geometry/appearance to VisPy and optionally
        auto-range the camera."""
        if self._points.size == 0:
            return

        # fall back to uniform grey if no colours cached
        if self._face_colors is None or len(self._face_colors) != len(self._points):
            self._face_colors = np.tile([0.7, 0.7, 0.7, 1.0], (len(self._points), 1))

        self.scatter.set_data(
            pos=self._points,
            size=self._point_size,
            face_color=self._face_colors,
        )
        if auto_range:
            xmin, ymin = self._points.min(axis=0)
            xmax, ymax = self._points.max(axis=0)
            self.view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), margin=0.1)
        self.canvas.update()

    def _nearest_idx_in_radius(self, xy_px: np.ndarray):
        """Return index of the nearest point if the mouse location lies within the point’s visual
        radius; otherwise return None."""
        if not len(self._points):
            return None
        screen_pts = self.view.scene.transform.map(self._points)[:, :2]
        dists = np.linalg.norm(screen_pts - xy_px, axis=1)
        idx = int(dists.argmin())
        return idx if dists[idx] <= self._point_size * 0.5 else None

    def _on_mouse_press(self, ev):
        if ev.button != 1:
            return
        idx = self._nearest_idx_in_radius(np.array(ev.pos[:2]))
        if idx is not None:
            self.point_pressed.emit(idx)

    def _on_mouse_release(self, ev):
        if ev.button != 1:
            return
        idx = self._nearest_idx_in_radius(np.array(ev.pos[:2]))
        if idx is not None:
            mods = QApplication.keyboardModifiers()
            self.point_released.emit(idx, mods)

    def data_to_global(self, pt_xy: np.ndarray) -> QPoint:
        px, py = self.view.scene.transform.map(pt_xy)[:2]
        return self.canvas.native.mapToGlobal(QPoint(int(px), int(py)))


def set_style(app):
    # https://www.wenzhaodesign.com/devblog/python-pyside2-simple-dark-theme
    # button from here https://github.com/persepolisdm/persepolis/blob/master/persepolis/gui/palettes.py
    app.setStyle(QStyleFactory.create("Fusion"))
    darktheme = QPalette()
    darktheme.setColor(QPalette.Window, QColor(45, 45, 45))
    darktheme.setColor(QPalette.WindowText, QColor(222, 222, 222))
    darktheme.setColor(QPalette.Button, QColor(45, 45, 45))
    darktheme.setColor(QPalette.ButtonText, QColor(222, 222, 222))
    darktheme.setColor(QPalette.AlternateBase, QColor(222, 222, 222))
    darktheme.setColor(QPalette.ToolTipBase, QColor(222, 222, 222))
    darktheme.setColor(QPalette.Highlight, QColor(45, 45, 45))
    darktheme.setColor(QPalette.Disabled, QPalette.Light, QColor(60, 60, 60))
    darktheme.setColor(QPalette.Disabled, QPalette.Shadow, QColor(50, 50, 50))
    darktheme.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(111, 111, 111))
    darktheme.setColor(QPalette.Disabled, QPalette.Text, QColor(122, 118, 113))
    darktheme.setColor(QPalette.Disabled, QPalette.WindowText, QColor(122, 118, 113))
    darktheme.setColor(QPalette.Disabled, QPalette.Base, QColor(32, 32, 32))
    app.setPalette(darktheme)

    app.setStyleSheet(
        """
        /* Closed combo box text + background */
        QComboBox {
            color: white;
            background-color: #3E3E3E;
            border: 1px solid #555;
            padding: 2px 6px;
            border-radius: 3px;
        }
        /* Popup list */
        QComboBox QAbstractItemView {
            background-color: #2E2E2E;
            color: white;
            selection-background-color: #555;
            outline: none;
        }
        /* Drop‐down arrow (optional—uses system arrow if you omit this) */
        QComboBox::down-arrow {
            width: 10px; height: 10px;
        }
    """
    )
    return app


def make_labeled_widget(label_text, widget):
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    label = QLabel(label_text)
    label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    layout.addWidget(label)
    layout.addWidget(widget)

    container = QWidget()
    container.setLayout(layout)
    container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    return container


class MainWindow(QMainWindow):
    clip_width_changed = Signal(float)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.df = None
        self.x_var = None
        self.y_var = None
        self.clip_width = 160

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # top controls
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # open button
        open_button = QPushButton("Open")
        open_button.clicked.connect(self.on_open)
        button_layout.addWidget(open_button)

        # point size slider
        point_size_slider = QSlider(Qt.Horizontal)
        point_size_slider.setRange(1, 50)
        point_size_slider.setValue(12)
        point_size_slider.setMinimumWidth(50)
        point_size_slider.setMaximumWidth(100)
        point_size_slider.valueChanged.connect(self.on_point_size_changed)
        button_layout.addWidget(make_labeled_widget("Point size:", point_size_slider))

        # clip-window size
        clip_width_slider = QSlider(Qt.Horizontal)
        clip_width_slider.setRange(80, 300)
        clip_width_slider.setSingleStep(10)
        clip_width_slider.setValue(160)
        clip_width_slider.setMinimumWidth(50)
        clip_width_slider.setMaximumWidth(100)
        clip_width_slider.valueChanged.connect(self.on_clip_width_changed)
        button_layout.addWidget(make_labeled_widget("Clip size:", clip_width_slider))

        # color‐by dropdown
        self.color_combo = QComboBox()
        self.color_combo.addItem("(No color)")
        self.color_combo.currentTextChanged.connect(self.on_color_changed)
        button_layout.addWidget(make_labeled_widget("Color by:", self.color_combo))

        # hide all clips button
        hide_all_btn = QPushButton("Hide all clips")
        hide_all_btn.clicked.connect(self.hide_all_clips)
        button_layout.addWidget(hide_all_btn)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # scatter plot
        self.scatter_plot = ScatterPlot(self)
        self.scatter_plot.point_pressed.connect(self.on_point_pressed)
        self.scatter_plot.point_released.connect(self.on_point_released)
        self.scatter_plot.canvas.events.draw.connect(self.reposition_popups)
        main_layout.addWidget(self.scatter_plot)

        # video player popup(s)
        self.clip_scale = 1.0
        self.transient_popup = None
        self.pinned_popups = {}

        if args:
            self.load_embedding_store(args[0])

    def load_embedding_store(self, path: str):
        try:
            store = EmbeddingStore.load(path)
        except Exception as e:
            print("Failed to load:", e)
            return

        self.df = store.metadata
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric metadata.")
            return

        dialog = VariableSelectDialog(numeric_cols, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        self.x_var, self.y_var = dialog.get_selection()

        # repopulate color‐by combo
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        self.color_combo.addItem("(No color)")
        for c in self.df.columns:
            self.color_combo.addItem(c)
        self.color_combo.blockSignals(False)

        pts = self.df[[self.x_var, self.y_var]].to_numpy()
        # initial plot (no color)
        self.scatter_plot.set_data(pts)

    def on_point_size_changed(self, val: int):
        self.scatter_plot.set_point_size(val)

    def on_color_changed(self, col_name: str):
        if self.df is None:
            return
        if col_name == "(No color)":
            self.scatter_plot.set_point_colors(None)
        else:
            self.scatter_plot.set_point_colors(self.df[col_name])

    def on_clip_width_changed(self, val: int):
        self.clip_width = val
        self.clip_width_changed.emit(val)

    @Slot()
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EmbeddingStore File", filter="HDF5 Files (*.h5 *.hdf5)"
        )
        if path:
            self.load_embedding_store(path)

    def _popup_key(self, idx):
        x, y = self.df.loc[idx, [self.x_var, self.y_var]]
        return (x, y)

    def _popup_pos(self, idx):
        pt = self._popup_key(idx)
        gpt = self.scatter_plot.data_to_global(pt)
        return gpt + QPoint(8, 8)  # offset so marker not hidden

    @Slot(int, Qt.KeyboardModifiers)
    def on_point_released(self, idx: int, mods: Qt.KeyboardModifiers):
        if self.transient_popup is None:
            return
        if mods & Qt.ShiftModifier:  # keep it → pin
            self.transient_popup.setWindowOpacity(1.0)
            self.pinned_popups[idx] = self.transient_popup
        else:  # drop it
            self.transient_popup.close()
        self.transient_popup = None

    @Slot(int)
    def on_point_pressed(self, idx: int):
        if self.transient_popup and not self.transient_popup.isHidden():
            self.transient_popup.close()

        info = (
            self.df.video_path[idx],
            int(self.df.start_frame[idx]),
            int(self.df.end_frame[idx]),
        )
        self.transient_popup = VideoPopup(info, f"clip #{idx}", self)
        self.transient_popup.idx = idx
        self.transient_popup.setWindowOpacity(0.9)
        self.transient_popup.move(self._popup_pos(idx))
        self.transient_popup.show()

    def _spawn_popup(self, idx, transient):
        info = (
            self.df.video_path[idx],
            int(self.df.start_frame[idx]),
            int(self.df.end_frame[idx]),
        )
        pop = VideoPopup(info, f"clip #{idx}", self)
        pop.idx = idx
        pop.move(self._popup_pos(idx))
        pop.show()
        if transient:
            pop.setWindowOpacity(0.9)
        return pop

    def reposition_popups(self, *a):
        # move pinned first
        for idx, pop in self.pinned_popups.items():
            pop.move(self._popup_pos(idx))
        # move transient (if it still exists)
        if self.transient_popup:
            self.transient_popup.move(self._popup_pos(self.transient_popup.idx))

    def _popup_closed(self, pop: QWidget):
        """Called by VideoPopup.closeEvent → drop every reference we hold to *pop*."""
        if pop is self.transient_popup:
            self.transient_popup = None
        for idx, vp in list(self.pinned_popups.items()):
            if vp is pop:
                del self.pinned_popups[idx]
                break

    @Slot()
    def hide_all_clips(self):
        """Close every visible clip window – both transient and pinned."""
        if self.transient_popup:
            self.transient_popup.close()
            self.transient_popup = None

        for pop in list(self.pinned_popups.values()):
            pop.close()
        self.pinned_popups.clear()


def run():
    app = QApplication(sys.argv)
    app = set_style(app)
    window = MainWindow(sys.argv[1:])
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
