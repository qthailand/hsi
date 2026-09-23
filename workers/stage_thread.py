# workers/stage_thread.py
from PyQt5.QtCore import QObject, QThread, pyqtSignal as Signal

from Stage8SMC1 import Stage8SMC1, Stage8SMC1Linear


class StageThread(QThread):
    stagebefore = Signal()
    stageafter = Signal()
    degree_per_second = 1
    degree = 10
    distance_mm = 10.0
    speed_um_s = 1000.0
    stage_type = 'rotation'

    def __init__(self, parent: QObject = None, stage_type: str = 'rotation'):
        super().__init__(parent)
        self.stage_type = str(stage_type).lower()
        self._stage_ref = None          # ★ เก็บ reference

    def request_stop(self):
        """สั่งหยุด hardware ทันที"""
        if self._stage_ref is not None and self._stage_ref._opened:
            try:
                self._stage_ref._axis.command_stop()
            except Exception:
                pass

    def run(self):
        StageClass = Stage8SMC1Linear if self.stage_type == 'linear' else Stage8SMC1
        with StageClass() as stage:
            self._stage_ref = stage      # ★ เก็บ reference

            if self.stage_type == 'linear':
                stage.set_speed(10)
            else:
                stage.set_speed_dps(10)
            stage.set_position(0, 0)
            self.stagebefore.emit()

            if self.stage_type == 'linear':
                stage.set_speed_mm_s(self.speed_um_s / 1000.0)
                stage.move_le_motor_mm(self.distance_mm)
            else:
                stage.set_speed_dps(self.degree_per_second)
                stage.set_degree(self.degree)

            self.stageafter.emit()
            if self.stage_type == 'linear':
                stage.set_speed_mm_s(self.speed_um_s / 1000.0)
            else:
                stage.set_speed_dps(self.degree_per_second)
            stage.set_position(0, 0)

            self._stage_ref = None       # ★ ล้าง reference
