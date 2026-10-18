"""Split configuration and atomically selected wavelength profiles.

app.json is published last;
profile writes use unique names so a failed save cannot replace the active one.
Paths are resolved against the project directory.

Layout (project root = parent of the settings directory)::

    config/app.json                      manifest: format_version, sections, wavelength_base, active_wavelength
    config/<section>.json                camera, scan, processing, dark, smile (keys in SECTIONS)
    calibration/wavelength/base.json     {"m", "c", "units": "nm"} (unbinned base dispersion)
    calibration/wavelength/profile_<uuid>.json   saved profile + "compatibility" block
    calibration/wavelength/<backend>/wavelength_bin<N>.json   per camera setting (see wavelength_folder)

Data conventions:
    ``m`` is nm per pixel (unbinned sensor), ``c`` is nm, exposure is ms.

Threading:
    Not thread-safe; use from one thread at a time and never in the frame loop.
"""
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import uuid

from modules.contracts import WavelengthCalibration

logger=logging.getLogger(__name__)
SECTIONS={
 'camera':('camera_backend','camera_id','camera_index','cti_path','default_gain','default_exposure_ms','default_bin','sensor_pixel_size','target_temperature_c'),
 'scan':('stage_type','rotation_degree','rotation_speed_dps','linear_distance_mm','linear_speed_um_s','scan_trim_extra_ms'),
 'processing':('resolution','resolution_source','fwhm_step_factor','raw_inverted','rgb_target_wavelengths','preprocessing_backend','profile_timing'),
 'dark':('dark_frame_folder','dark_frame_n_frames'),
 'smile':('smile_map_folder',),
}

class ConfigStore:
    """Reads and writes the split settings directory and the active wavelength profile.

    Args:
        directory: Settings directory (e.g. ``<project>/config``). Resolved to an absolute
            path; its parent is the project root used for all manifest-relative paths.

    Attributes:
        directory: Absolute settings directory.
        root: Project root (``directory.parent``).
        manifest: Path of ``app.json`` inside ``directory``.
    """
    def __init__(self, directory, sections=None):
        """Resolve ``directory`` and derive the project root and manifest path (no I/O).

        Args:
            directory: Settings directory.
            sections: ``{section name: tuple of setting names}``; defaults to the legacy
                :data:`SECTIONS`. Applications pass the sections their modules own
                (see ``config.config_sections``).
        """
        self.sections=dict(SECTIONS if sections is None else sections)
        self.directory=Path(directory).resolve()
        self.root=self.directory.parent
        self.manifest=self.directory/'app.json'

    @staticmethod
    def _read(path):
        """Parse a JSON file that must contain an object.

        Raises:
            ValueError: The top-level JSON value is not an object.
        """
        result=json.loads(Path(path).read_text(encoding='utf-8-sig'))   # tolerate a BOM (Notepad/PowerShell)
        if not isinstance(result,dict):raise ValueError(f'Configuration must be an object: {path}')
        return result

    @staticmethod
    def _atomic(path,data):
        """Write ``data`` as JSON to a unique temporary file and move it over ``path``.

        Parent directories are created. The temp file is removed on failure, so ``path`` is
        either the old or the new content, never partial. NaN/inf are rejected.
        """
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def initialize(self,data):
        """Explicitly create a new settings directory; never overwrite one.

        Writes one JSON file per section (only the keys present in ``data``), the base
        wavelength file and finally the ``app.json`` manifest.

        Args:
            data: Flat settings dict containing the keys listed in ``SECTIONS`` (those
                that are present are stored) and the base dispersion ``m`` (nm/px,
                unbinned) and ``c`` (nm).

        Raises:
            FileExistsError: ``directory`` already exists.
            KeyError: ``m`` or ``c`` is missing from ``data``.
        """
        if self.directory.exists():raise FileExistsError(self.directory)
        refs={}
        for section,keys in self.sections.items():
            path=self.directory/(section+'.json')
            self._atomic(path,{k:data[k] for k in keys if k in data})
            refs[section]=path.name
        base=self.root/'calibration'/'wavelength'/'base.json'
        self._atomic(base,{'m':data['m'],'c':data['c'],'units':'nm'})
        self._atomic(self.manifest,{'format_version':1,'sections':refs,'wavelength_base':str(base.relative_to(self.root)),'active_wavelength':None})

    def read(self):
        """Load all sections into one flat settings dict.

        The base ``m``/``c`` come from ``wavelength_base``. If the manifest names an
        ``active_wavelength`` profile, it is loaded, checked against the current camera
        settings, and overrides them: ``m`` becomes the profile's unbinned ``base_m``
        (nm/px), ``c`` its intercept (nm), and the profile object is added under
        ``_wavelength_profile``.

        Returns:
            dict: Merged section values plus ``m``, ``c`` (and ``_wavelength_profile``).

        An active profile whose ``compatibility`` block (camera backend/id/index,
        raw_inverted, bin) differs from the current settings is not applied (warning; base
        ``m``/``c`` are used), so changing the camera in ``camera.json`` always takes effect.

        Raises:
            ValueError: Unsupported ``format_version``, non-object JSON, base units not nm,
                or non-finite or non-positive base ``m``.
            KeyError: A required section, key or file reference is missing.
            OSError: A referenced file cannot be read.
        """
        manifest=self._read(self.manifest)
        if manifest.get('format_version')!=1:raise ValueError('Unsupported config format')
        data={}
        for section in self.sections:
            data.update(self._read(self._section_path(manifest,section)))
        base=self._read(self.root/manifest['wavelength_base'])
        if base.get('units')!='nm':raise ValueError('Wavelength base units must be nm')
        data['m']=float(base['m']);data['c']=float(base['c'])
        if not math.isfinite(data['m']) or data['m']<=0 or not math.isfinite(data['c']):
            raise ValueError('Invalid wavelength base coefficients')
        active=manifest.get('active_wavelength')
        if active:
            bundle=self._read(self.root/active)
            profile=WavelengthCalibration.load(self.root/active)
            expected=bundle['compatibility']
            current={'camera_backend':str(data['camera_backend']).lower(),'camera_id':int(data['camera_id']),'camera_index':int(data.get('camera_index',0)),'raw_inverted':bool(data['raw_inverted']),'camera_bin':int(data['default_bin'])}
            if current!=expected:
                # The camera/bin/orientation was changed (e.g. camera.json edited): never apply a fit
                # measured on another setting, but do not block startup either; use the base m/c.
                logger.warning('Active wavelength profile %s is for %s, current settings are %s; using base m/c',active,expected,current)
            else:
                data['m']=profile.base_m;data['c']=profile.c
                data['_wavelength_profile']=profile
        return data

    def wavelength_folder(self,camera_backend):
        """Folder of the per-setting wavelength profiles (``wavelength_bin<N>.json``) of one backend.

        These files (``modules.contracts.CalibrationStore``) take precedence over the ``m``/``c``
        returned by :meth:`read` for the camera bin they were measured at.
        """
        return self.root/'calibration'/'wavelength'/str(camera_backend).lower()

    def _section_path(self,manifest,section):
        """File holding ``section``: the one named in the manifest, unless ``<section>.json`` was edited later.

        Older versions saved the startup bin into a generated ``camera_<uuid>.json`` and pointed the
        manifest at it, so edits to ``camera.json`` were ignored. The user-edited file wins when newer.
        """
        named=self.directory/manifest['sections'][section]
        plain=self.directory/(section+'.json')
        if named!=plain and plain.exists() and (not named.exists() or plain.stat().st_mtime>named.stat().st_mtime):
            logger.warning('Using %s (edited after %s named in app.json)',plain.name,named.name)
            return plain
        return named

    def update_section(self,section,values):
        """Write ``values`` into ``<section>.json``, keeping its other keys.

        Used to remember settings the user changes (camera gain/exposure/bin/target temperature,
        stage scan values) for the next start. The section the manifest names is read; the
        result goes to the plain ``<section>.json`` (the file users edit), and the manifest is
        repointed at it if it named another file. Unchanged values are not rewritten.

        Args:
            section: A section name, e.g. ``'camera'`` or ``'scan'``.
            values: ``{setting name: JSON value}``.

        Returns:
            bool: True if the file was written.

        Raises:
            KeyError: ``section`` is not in the manifest.
            ValueError: a value is NaN/inf (rejected by the atomic writer).
        """
        manifest=self._read(self.manifest)
        source=self._section_path(manifest,section)
        current=self._read(source) if source.exists() else {}
        target=self.directory/(section+'.json')
        merged={**current,**values}
        if merged==current and source==target:
            return False
        self._atomic(target,merged)
        if manifest['sections'][section]!=target.name:
            manifest['sections'][section]=target.name
            self._atomic(self.manifest,manifest)
        return True

    def save_wavelength_default(self,profile,config):
        """Persist after successful worker application, preserving split settings.

        Writes a new ``calibration/wavelength/profile_<uuid>.json`` (profile fields plus a
        ``compatibility`` block), republishes ``app.json`` with ``active_wavelength`` pointing
        at the new profile (and its camera section at ``camera.json``), then writes the
        profile's bin as ``default_bin`` into ``camera.json``. Older files are not deleted.

        Args:
            profile: A ``WavelengthCalibration`` (validated here).
            config: Object with ``camera_backend`` (str), ``camera_id``, ``camera_index``
                and ``raw_inverted`` attributes describing the camera the profile was
                measured with.

        Returns:
            pathlib.Path: The profile file that was written.

        Raises:
            ValueError: The profile is invalid.
            KeyError: The manifest or camera section lacks required keys.
        """
        profile.validate()
        path=self.root/'calibration'/'wavelength'/('profile_'+uuid.uuid4().hex+'.json')
        data=asdict(profile);data.update(format_version=1,units='nm',stage='post_dark_hot_pixel_orientation')
        data['compatibility']={'camera_backend':config.camera_backend.lower(),'camera_id':config.camera_id,'camera_index':config.camera_index,'raw_inverted':config.raw_inverted,'camera_bin':profile.camera_bin}
        self._atomic(path,data)
        manifest=self._read(self.manifest)
        source=self._section_path(manifest,'camera')
        camera=self._read(source)
        camera_path=self.directory/'camera.json'
        manifest['sections']['camera']=camera_path.name
        manifest['active_wavelength']=str(path.relative_to(self.root)).replace('\\','/')
        self._atomic(self.manifest,manifest)
        # camera.json stays the one file users edit: the startup bin is written into it, not a copy.
        if camera['default_bin'] != profile.camera_bin or source!=camera_path:
            camera['default_bin']=profile.camera_bin
            self._atomic(camera_path,camera)
        return path
