from typing import List
from .backends import SmileBackend, SmileParams
from .backends.joe import JoeSmileBackend


_BACKEND_REGISTRY = {
    "joe": JoeSmileBackend,
}


class SmileAdaptor:
    def __init__(self, backend_name: str = "joe"):
        key = backend_name.lower()
        if key not in _BACKEND_REGISTRY:
            raise ValueError(f"Unsupported smile backend: {backend_name}")
        self.backend: SmileBackend = _BACKEND_REGISTRY[key]()

    def detect_peaks(self, raw_img, params: SmileParams):
        return self.backend.detect_peaks(raw_img, params)

    def build_correction(self, raw_img, detected_peaks, keystone_model=None):
        return self.backend.build_correction(raw_img, detected_peaks, keystone_model)

    def remap(self, raw_img, mx, my):
        return self.backend.remap(raw_img, mx, my)

    def diagnose(self, corrected_img, detected_peaks, params: SmileParams):
        return self.backend.diagnose(corrected_img, detected_peaks, params)

    def create_keystone_model(self, keystone_ref_img):
        return self.backend.create_keystone_model(keystone_ref_img)

    @staticmethod
    def supported_backends() -> List[str]:
        return list(_BACKEND_REGISTRY.keys())
