import sys
import cv2 as cv
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QButtonGroup, QSlider, QFormLayout, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from pathlib import Path
from datetime import datetime


def cvimg_to_qpix(img_bgr):
    if img_bgr is None:
        return QPixmap()
    if len(img_bgr.shape) == 2:
        h, w = img_bgr.shape
        qimg = QImage(img_bgr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


PRIMARY = "#1976D2"
HOVER   = "#1E88E5"
PRESSED = "#0D47A1"


class OpenCVQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV x PyQt (Material Style)")
        self.setMinimumSize(1200, 1400)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.src_img = None
        self.current = None
        self.current_filter = None

        self.ksize = 7
        self.thresh = 127
        self.canny_low = 80
        self.canny_high = 160
        self.morph_iter = 1

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background: white;")

        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(16)

        header = QWidget()
        header.setStyleSheet(f"background: {PRIMARY};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        title = QLabel("OpenCV")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: 700;")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title, alignment=Qt.AlignCenter)
        self.main_layout.addWidget(header)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setText("이미지를 열거나 카메라를 시작하세요")
        self.view.setStyleSheet("font-size: 14px; color: #666; background: white;")
        self.view.setMinimumSize(720, 420)
        self.view.setScaledContents(False)
        self.main_layout.addWidget(self.view, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.setContentsMargins(20, 0, 20, 0)

        def make_ctrl_btn(text):
            b = QPushButton(text)
            b.setMinimumHeight(38)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {PRIMARY};
                    color: white;
                    font-size: 13px;
                    font-weight: 600;
                    border-radius: 10px;
                    padding: 8px 18px;
                }}
                QPushButton:hover  {{ background-color: {HOVER}; }}
                QPushButton:pressed, QPushButton:checked {{ background-color: {PRESSED}; }}
            """)
            return b

        self.open_btn = make_ctrl_btn("이미지 열기")
        self.cam_btn  = make_ctrl_btn("카메라 시작")
        self.save_btn = make_ctrl_btn("프레임 저장")

        self.open_btn.clicked.connect(self.action_open_image)
        self.cam_btn.clicked.connect(self.toggle_camera)
        self.save_btn.clicked.connect(self.action_save_frame)

        controls.addStretch(1)
        controls.addWidget(self.open_btn)
        controls.addWidget(self.cam_btn)
        controls.addWidget(self.save_btn)
        controls.addStretch(1)

        self.main_layout.addLayout(controls)

        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)

        button_names = [
            ("GaussianBlur", "gauss"),
            ("Threshold", "thresh"),
            ("Canny", "canny"),
            ("GrayScale", "gray"),
            ("RGBExtraction", "rgb"),
            ("HSVExtraction", "hsv"),
            ("Erode", "erode"),
            ("Dilate", "dilate"),
        ]

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        def make_filter_btn(text):
            b = QPushButton(text)
            b.setMinimumHeight(44)
            b.setCheckable(True)
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {PRIMARY};
                    color: white;
                    font-size: 14px;
                    font-weight: 700;
                    border-radius: 12px;
                }}
                QPushButton:hover {{ background-color: {HOVER}; }}
                QPushButton:checked {{ background-color: {PRESSED}; }}
            """)
            return b

        for text, name in button_names:
            btn = make_filter_btn(text)
            btn.clicked.connect(lambda _, n=name: self.apply_filter(n))
            self.btn_group.addButton(btn)
            buttons_layout.addWidget(btn)

        self.main_layout.addLayout(buttons_layout)

        self.param_box = QGroupBox("파라미터")
        self.param_box.setStyleSheet("QGroupBox{font-weight:700; background:white;}")
        self.param_layout = QFormLayout()
        self.param_box.setLayout(self.param_layout)
        self.param_box.setVisible(False)
        self.main_layout.addWidget(self.param_box)

        app_dir = Path(__file__).resolve().parent
        candidates = [app_dir / "img" / "opencv_logo.png", app_dir / "img" / "opencv_log.png"]
        logo_path = next((p for p in candidates if p.exists()), None)
        if logo_path:
            logo = QPixmap(str(logo_path))
            self.view.setPixmap(logo.scaled(self.view.width(), self.view.height(),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def toggle_camera(self):
        if self.cap:
            self.action_stop_camera()
            self.cam_btn.setText("카메라 시작")
        else:
            self.action_start_camera()
            self.cam_btn.setText("카메라 중지")

    def action_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "이미지 열기", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.stop_camera()
        self.cam_btn.setText("카메라 시작")
        img = cv.imread(path)
        if img is None:
            QMessageBox.warning(self, "오류", "이미지를 읽을 수 없습니다.")
            return
        self.src_img = img
        self.current = img.copy()
        self.current_filter = None
        self._render()

    def action_save_frame(self):
        if self.current is None:
            QMessageBox.information(self, "알림", "저장할 프레임이 없습니다.")
            return

        out_dir = Path(__file__).resolve().parent / "captures"
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"

        ok = cv.imwrite(str(fname), self.current)
        if ok:
            QMessageBox.information(self, "저장됨", f"저장 위치:\n{fname}")
        else:
            QMessageBox.warning(self, "오류", "저장에 실패했습니다.")

    def action_start_camera(self):
        self.start_camera(0)
        self.cam_btn.setText("카메라 중지")

    def action_stop_camera(self):
        self.stop_camera()
        self.cam_btn.setText("카메라 시작")

    def start_camera(self, index=0):
        self.stop_camera()
        self.cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "오류", "카메라를 열 수 없습니다.")
            self.cap = None
            return
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.src_img = frame
        self._apply_and_render()

    def apply_filter(self, name: str):
        self.current_filter = name
        self._update_params_ui(name)
        self._apply_and_render()

    def _update_params_ui(self, name: str):
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if name == "gauss":
            sl = QSlider(Qt.Horizontal); sl.setRange(1, 21); sl.setValue(self.ksize)
            sl.valueChanged.connect(self._update_ksize)
            self.param_layout.addRow("커널(홀수):", sl); self.param_box.setVisible(True)

        elif name == "thresh":
            sl = QSlider(Qt.Horizontal); sl.setRange(0, 255); sl.setValue(self.thresh)
            sl.valueChanged.connect(self._update_thresh)
            self.param_layout.addRow("임계값:", sl); self.param_box.setVisible(True)

        elif name == "canny":
            sl_low = QSlider(Qt.Horizontal);  sl_low.setRange(0, 255);  sl_low.setValue(self.canny_low)
            sl_high = QSlider(Qt.Horizontal); sl_high.setRange(0, 255); sl_high.setValue(self.canny_high)
            sl_low.valueChanged.connect(self._update_canny)
            sl_high.valueChanged.connect(self._update_canny)
            self.param_layout.addRow("Canny Low:", sl_low)
            self.param_layout.addRow("Canny High:", sl_high)
            self.param_box.setVisible(True)

        elif name in ("erode", "dilate"):
            sl = QSlider(Qt.Horizontal); sl.setRange(1, 10); sl.setValue(self.morph_iter)
            sl.valueChanged.connect(self._update_iter)
            self.param_layout.addRow("반복 횟수:", sl); self.param_box.setVisible(True)

        else:
            self.param_box.setVisible(False)

    def _update_ksize(self, v):
        if v % 2 == 0:
            v += 1
        self.ksize = v
        self._apply_and_render()

    def _update_thresh(self, v):
        self.thresh = v
        self._apply_and_render()

    def _update_canny(self, _):
        self.canny_low = self.param_layout.itemAt(1).widget().value()
        self.canny_high = self.param_layout.itemAt(3).widget().value()
        if self.canny_high <= self.canny_low:
            self.canny_high = self.canny_low + 1
        self._apply_and_render()

    def _update_iter(self, v):
        self.morph_iter = v
        self._apply_and_render()

    def _apply_and_render(self):
        if self.src_img is None:
            return
        self.current = self._process(self.src_img, self.current_filter)
        self._render()

    def _process(self, img, name):
        if name is None:
            return img
        if name == "gauss":
            k = max(1, self.ksize | 1)
            return cv.GaussianBlur(img, (k, k), 0)
        if name == "thresh":
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, bw = cv.threshold(gray, self.thresh, 255, cv.THRESH_BINARY)
            return cv.cvtColor(bw, cv.COLOR_GRAY2BGR)
        if name == "canny":
            edges = cv.Canny(img, self.canny_low, self.canny_high)
            return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        if name == "gray":
            g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            return cv.cvtColor(g, cv.COLOR_GRAY2BGR)
        if name == "rgb":
            b, g, r = cv.split(img)
            r_img = cv.merge([np.zeros_like(b), np.zeros_like(g), r])
            g_img = cv.merge([np.zeros_like(b), g, np.zeros_like(r)])
            b_img = cv.merge([b, np.zeros_like(g), np.zeros_like(r)])
            top = np.hstack([r_img, g_img, b_img])
            return top
        if name == "hsv":
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            h_col = cv.applyColorMap(cv.convertScaleAbs(h, alpha=255/180.0), cv.COLORMAP_HSV)
            s_col = cv.cvtColor(s, cv.COLOR_GRAY2BGR)
            v_col = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
            top = np.hstack([h_col, s_col, v_col])
            return top
        if name == "erode":
            k = np.ones((self.ksize | 1, self.ksize | 1), np.uint8)
            return cv.erode(img, k, iterations=self.morph_iter)
        if name == "dilate":
            k = np.ones((self.ksize | 1, self.ksize | 1), np.uint8)
            return cv.dilate(img, k, iterations=self.morph_iter)
        return img

    def _render(self):
        if self.current is None:
            return
        pix = cvimg_to_qpix(self.current)
        if pix.isNull():
            return
        max_w = self.view.width()
        max_h = self.view.height()
        img_w = pix.width()
        img_h = pix.height()
        if img_w <= max_w and img_h <= max_h:
            self.view.setPixmap(pix)
        else:
            self.view.setPixmap(pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.stop_camera()
        self.cam_btn.setText("카메라 시작")
        return super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OpenCVQt()
    win.show()
    sys.exit(app.exec_())