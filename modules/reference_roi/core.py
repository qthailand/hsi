"""Spatial ROI profiles and immutable, atomic active-profile selection (Qt-free).

Data conventions:
    * A profile is a half-open range of spatial rows ``[y_start, y_end)`` along the
      slit axis, in the post-orientation coordinate space of a spectral image
      ``(slit, wavelength)`` (``coordinate_space = 'post_orientation_spatial_rows'``).
      Rows are integer pixels at the camera's own hardware bin.
    * A *preview image* is ``uint8`` RGB of shape ``(slit, scan, 3)``: spatial row on
      axis 0, scan frame on axis 1.
    * ``compatibility`` is a dict with the keys in ``COMPATIBILITY_KEYS`` describing
      the camera setup the ROI was drawn on.
    * Files: ``<project>/calibration/reference_roi/<profile_id>.json`` (profile,
      ``format_version`` 1) and ``active.json`` (``{'format_version': 1,
      'active_profile': <filename or null>}``). Both are written atomically.

Threading:
    Pure functions and plain file I/O; no shared state. Store methods block on disk
    access, so call them from a worker thread if latency matters.

Example:
    >>> import tempfile
    >>> from modules.reference_roi import ReferenceROIProfile, ReferenceROIStore
    >>> ctx = dict(camera_backend='x', camera_id='0', camera_index=0, camera_bin=1,
    ...            raw_inverted=False, spatial_height=100)
    >>> store = ReferenceROIStore(tempfile.mkdtemp())
    >>> saved = store.save_active(ReferenceROIProfile('ref', 10, 20, ctx))
    >>> store.load_active().y_end
    20
    >>> store.clear_active(); store.load_active() is None
    True
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import uuid
import datetime
import numpy as np

COMPATIBILITY_KEYS = ('camera_backend', 'camera_id', 'camera_index', 'camera_bin', 'raw_inverted', 'spatial_height')

@dataclass(frozen=True)
class ReferenceROIProfile:
    """Immutable named spatial-row range plus the camera context it is valid for.

    Attributes:
        name: Non-empty display name.
        y_start: First spatial row (inclusive), ``int``.
        y_end: One past the last spatial row (exclusive), ``int``.
        compatibility: Dict containing every key of ``COMPATIBILITY_KEYS``;
            ``spatial_height`` must be a positive ``int``.
        profile_id: Unique hex id assigned by ``ReferenceROIStore.save_active``.
        created_at: UTC ISO-8601 timestamp assigned by ``save_active``.
    """
    name: str
    y_start: int
    y_end: int
    compatibility: dict
    profile_id: str = ''
    created_at: str = ''

    def validate(self):
        """Check the profile is well formed.

        Returns:
            ReferenceROIProfile: ``self``, so calls can be chained.

        Raises:
            ValueError: invalid spatial height, ``y_start``/``y_end`` not ints or not
                ``0 <= y_start < y_end <= spatial_height``, empty name, or a missing
                compatibility key.
        """
        height = self.compatibility.get('spatial_height')
        if type(height) is not int or height <= 0: raise ValueError('Invalid spatial height')
        if type(self.y_start) is not int or type(self.y_end) is not int or not 0 <= self.y_start < self.y_end <= height:
            raise ValueError('ROI must satisfy 0 <= y_start < y_end <= spatial height')
        if not self.name.strip(): raise ValueError('Profile name is required')
        if any(key not in self.compatibility for key in COMPATIBILITY_KEYS): raise ValueError('Incomplete camera compatibility')
        return self

    def matches(self, context):
        """Return True if ``context`` equals this profile's compatibility values.

        Args:
            context: Dict describing the current camera setup; only the keys in
                ``COMPATIBILITY_KEYS`` are compared.

        Returns:
            bool: True when every compared key is equal.
        """
        return all(self.compatibility.get(key) == context.get(key) for key in COMPATIBILITY_KEYS)

    def to_dict(self):
        """Return a JSON-serialisable dict including ``format_version`` (1) and ``coordinate_space``."""
        result = asdict(self); result['format_version'] = 1; result['coordinate_space'] = 'post_orientation_spatial_rows'
        return result

    @classmethod
    def from_dict(cls, data):
        """Build a validated profile from ``to_dict`` output.

        Args:
            data: Dict read from a profile JSON file.

        Raises:
            ValueError: unknown ``format_version``/``coordinate_space``, or the
                profile fails ``validate``.
            KeyError: a profile field is missing.
        """
        if data.get('format_version') != 1 or data.get('coordinate_space') != 'post_orientation_spatial_rows':
            raise ValueError('Unsupported reference ROI profile')
        return cls(**{key: data[key] for key in cls.__dataclass_fields__}).validate()

class ReferenceROIStore:
    """Persists profiles under ``<project_root>/calibration/reference_roi`` with one active profile.

    Profile files are never modified after being written; activating a profile only
    atomically rewrites ``active.json``.
    """

    def __init__(self, project_root):
        """Create the store.

        Args:
            project_root: Project directory; the folder is created on first save.
        """
        self.folder = Path(project_root) / 'calibration' / 'reference_roi'
        self.manifest = self.folder / 'active.json'

    @staticmethod
    def _atomic(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_active(self):
        """Load the active profile.

        Returns:
            ReferenceROIProfile | None: ``None`` if there is no manifest or the
            selection was cleared.

        Raises:
            ValueError: unsupported manifest/profile version, invalid profile, or an
                ``active_profile`` that is not a plain filename.
            OSError: the referenced profile file cannot be read.
        """
        if not self.manifest.exists(): return None
        manifest = json.loads(self.manifest.read_text(encoding='utf-8'))
        if manifest.get('format_version') != 1: raise ValueError('Unsupported reference ROI manifest')
        selected = manifest.get('active_profile')
        if selected is None: return None
        if Path(selected).name != selected: raise ValueError('Invalid active ROI profile filename')
        return ReferenceROIProfile.from_dict(json.loads((self.folder / selected).read_text(encoding='utf-8')))

    def save_active(self, profile):
        """Validate, store as a new immutable file and make it the active profile.

        Args:
            profile: Profile to save; its ``profile_id`` and ``created_at`` are ignored
                and replaced by fresh values.

        Returns:
            ReferenceROIProfile: the saved copy with ``profile_id`` and ``created_at`` set.

        Raises:
            ValueError: ``profile`` is invalid.
        """
        profile.validate()
        values = asdict(profile)
        values.update(profile_id=uuid.uuid4().hex, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        saved = ReferenceROIProfile(**values)
        filename = saved.profile_id + '.json'
        self._atomic(self.folder / filename, saved.to_dict())
        self._atomic(self.manifest, {'format_version': 1, 'active_profile': filename})
        return saved

    def clear_active(self):
        """Deselect the active profile; saved profile files are kept."""
        self._atomic(self.manifest, {'format_version': 1, 'active_profile': None})

def select_rgb(frame, wavelengths, targets=(645., 555., 465.)):
    """Extract three channels from a corrected/binned spectrogram, preserving y.

    Args:
        frame: 2-D spectral image ``(slit, wavelength)``.
        wavelengths: ``numpy.ndarray`` of nm values, one per column of ``frame``, or
            ``None``. If ``None`` or of the wrong length, three evenly spaced columns
            from last to first are used instead.
        targets: Wavelengths in nm for the three output channels (default R, G, B).

    Returns:
        numpy.ndarray: C-contiguous ``float32`` array ``(slit, 3)``; a copy.

    Raises:
        ValueError: ``frame`` is not 2-D or has no columns.
    """
    if frame.ndim != 2 or frame.shape[1] == 0: raise ValueError('Expected nonempty 2D spectrogram')
    if wavelengths is not None and len(wavelengths) == frame.shape[1]:
        indices = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
    else:
        indices = np.linspace(frame.shape[1] - 1, 0, 3).astype(int)
    return np.ascontiguousarray(frame[:, indices], dtype=np.float32)

def compose_preview(columns):
    """Create spatial-y × scan-x RGB; display scaling never changes row mapping.

    Args:
        columns: Non-empty sequence of ``(slit, 3)`` arrays (see ``select_rgb``), one
            per scan frame.

    Returns:
        numpy.ndarray: C-contiguous ``uint8`` image ``(slit, scan, 3)``, scaled per
        channel between the 1st and 99th percentiles.

    Raises:
        ValueError: ``columns`` is empty.
    """
    if not columns: raise ValueError('No preview frames captured')
    image = np.stack(columns, axis=1)
    low, high = np.percentile(image, (1, 99), axis=(0, 1))
    normalized = np.clip((image - low) / np.maximum(high - low, 1e-6), 0, 1)
    return np.ascontiguousarray(normalized * 255, dtype=np.uint8)
