"""Split configuration migration, persistence and safe startup selection."""
import json
from pathlib import Path
import pytest
from config import load_app_config
from modules.wavelength_calibration import fit_wavelength

def settings(tmp_path,**updates):
    from config import AppConfig
    from dataclasses import asdict
    from modules.configuration import ConfigStore
    data=asdict(AppConfig());data.update(default_bin=2,m=.29,c=400,dark_frame_folder='dark_frames');data.update(updates)
    directory=tmp_path/'config';ConfigStore(directory).initialize(data);return directory

def test_split_files_and_explicit_paths(tmp_path):
    p=settings(tmp_path);config=load_app_config(str(p))
    for name in ['app','camera','scan','processing','dark','smile']:assert (p/f'{name}.json').exists()
    assert config.dark_frame_folder==str(tmp_path/'dark_frames'/config.camera_backend)
    camera=p/'camera.json';data=json.loads(camera.read_text());data['default_gain']=77;camera.write_text(json.dumps(data))
    assert load_app_config(str(p)).default_gain==77

def test_default_reload_and_startup_bin(tmp_path):
    p=settings(tmp_path);config=load_app_config(str(p));profile=fit_wavelength([0,100],[410,530],camera_bin=4)
    saved=config._config_store.save_wavelength_default(profile,config)
    assert saved.exists()
    loaded=load_app_config(str(p))
    assert loaded.m==pytest.approx(.3) and loaded.c==pytest.approx(410)
    assert loaded.default_bin==4 and loaded._wavelength_profile==profile
    assert json.loads((p/'camera.json').read_text())['default_bin']==4        # written into camera.json itself
    assert json.loads(store_manifest(p))['sections']['camera']=='camera.json'
    # Later device/orientation changes never apply the incompatible fit, and still start.
    proc=tmp_path/'config/processing.json';data=json.loads(proc.read_text());data['raw_inverted']=False;proc.write_text(json.dumps(data))
    loaded=load_app_config(str(p))
    assert loaded._wavelength_profile is None and loaded.m==pytest.approx(.29) and loaded.c==pytest.approx(400)

def store_manifest(p):
    return (p/'app.json').read_text()

def test_changing_camera_in_camera_json_takes_effect_after_save_default(tmp_path):
    p=settings(tmp_path,camera_backend='genicam');config=load_app_config(str(p))
    config._config_store.save_wavelength_default(fit_wavelength([0,100],[410,530],camera_bin=4),config)
    camera=p/'camera.json';data=json.loads(camera.read_text());data['camera_backend']='ximea';camera.write_text(json.dumps(data))
    loaded=load_app_config(str(p))
    assert loaded.camera_backend=='ximea' and loaded._wavelength_profile is None

def test_legacy_generated_camera_file_loses_to_a_later_edit_of_camera_json(tmp_path):
    import os
    p=settings(tmp_path,camera_backend='genicam')
    generated=p/'camera_0123abcd.json';data=json.loads((p/'camera.json').read_text());generated.write_text(json.dumps(data))
    manifest=json.loads(store_manifest(p));manifest['sections']['camera']=generated.name;(p/'app.json').write_text(json.dumps(manifest))
    os.utime(p/'camera.json',(1,1))
    assert load_app_config(str(p)).camera_backend=='genicam'                    # generated file is newer: still used
    data['camera_backend']='ximea';(p/'camera.json').write_text(json.dumps(data))
    os.utime(generated,(1,1))
    assert load_app_config(str(p)).camera_backend=='ximea'                      # the user's edit wins

def test_failed_profile_write_keeps_active_pointer(tmp_path,monkeypatch):
    p=settings(tmp_path);config=load_app_config(str(p));store=config._config_store
    profile=fit_wavelength([0,100],[410,470],2)
    store.save_wavelength_default(profile,config);before=store.manifest.read_bytes()
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr(store,'_atomic',fail)
    with pytest.raises(OSError):store.save_wavelength_default(profile,config)
    assert store.manifest.read_bytes()==before
    assert load_app_config(str(p)).c==pytest.approx(410)

def test_missing_active_profile_falls_back_to_base(tmp_path):
    p=settings(tmp_path);config=load_app_config(str(p));store=config._config_store
    manifest=json.loads(store.manifest.read_text());manifest['active_wavelength']='missing.json';store.manifest.write_text(json.dumps(manifest))
    loaded=load_app_config(str(p))
    assert loaded._wavelength_profile is None and loaded.m==pytest.approx(.29) and loaded.c==pytest.approx(400)
    assert any('missing.json' in r for r in loaded._config_repairs)
    assert json.loads(store.manifest.read_text())['active_wavelength'] is None
    assert load_app_config(str(p))._config_repairs==[]                     # repaired once, then quiet


def test_intact_config_reports_no_repairs(tmp_path):
    assert load_app_config(str(settings(tmp_path)))._config_repairs==[]


def test_missing_config_directory_is_recreated_from_defaults(tmp_path):
    from config import AppConfig
    p=tmp_path/'config'
    config=load_app_config(str(p))
    for name in ['app','camera','scan','processing','dark','smile']:assert (p/f'{name}.json').exists()
    base=json.loads((tmp_path/'calibration/wavelength/base.json').read_text())
    assert base=={'m':AppConfig.m,'c':AppConfig.c,'units':'nm'}
    assert config.camera_backend==AppConfig.camera_backend and config.m==pytest.approx(AppConfig.m)
    assert json.loads((p/'dark.json').read_text())['dark_frame_folder']=='dark_frames'    # relative, as shipped
    assert config.dark_frame_folder==str(tmp_path/'dark_frames'/config.camera_backend)
    assert len(config._config_repairs)==7                                     # manifest, 5 sections, base


@pytest.mark.parametrize('section',['camera','scan','processing','dark','smile'])
def test_deleted_section_file_is_recreated_and_others_kept(tmp_path,section):
    p=settings(tmp_path,default_gain=77,rotation_degree=123.0)
    (p/f'{section}.json').unlink()
    config=load_app_config(str(p))
    assert (p/f'{section}.json').exists()
    assert [r for r in config._config_repairs if f'{section}.json' in r]
    assert len(config._config_repairs)==1
    if section!='camera':assert config.default_gain==77               # untouched files keep their values
    if section!='scan':assert config.rotation_degree==123.0


def test_deleted_manifest_is_recreated(tmp_path):
    p=settings(tmp_path,default_gain=77);(p/'app.json').unlink()
    config=load_app_config(str(p))
    assert config.default_gain==77 and config.m==pytest.approx(.29)          # sections and base kept
    assert len(config._config_repairs)==1 and 'app.json' in config._config_repairs[0]


def test_deleted_wavelength_base_is_recreated(tmp_path):
    from config import AppConfig
    p=settings(tmp_path);base=tmp_path/'calibration/wavelength/base.json';base.unlink()
    config=load_app_config(str(p))
    assert base.exists() and config.m==pytest.approx(AppConfig.m) and config.c==pytest.approx(AppConfig.c)
    assert len(config._config_repairs)==1 and 'base.json' in config._config_repairs[0]


def test_corrupt_files_are_never_overwritten(tmp_path):
    p=settings(tmp_path);camera=p/'camera.json';camera.write_text('{ broken')
    with pytest.raises(ValueError):load_app_config(str(p))
    assert camera.read_text()=='{ broken'
    base=tmp_path/'calibration/wavelength/base.json';camera.unlink();base.write_text('[1, 2]')
    with pytest.raises(ValueError):load_app_config(str(p))
    assert base.read_text()=='[1, 2]'

def test_failed_manifest_commit_preserves_camera_and_profile(tmp_path,monkeypatch):
    directory=settings(tmp_path);config=load_app_config(str(directory));store=config._config_store
    old=fit_wavelength([0,100],[400,458],2);store.save_wavelength_default(old,config)
    original=store._atomic
    def fail_manifest(path,data):
        if Path(path)==store.manifest:raise OSError('manifest write failed')
        original(path,data)
    monkeypatch.setattr(store,'_atomic',fail_manifest)
    with pytest.raises(OSError):store.save_wavelength_default(fit_wavelength([0,100],[410,530],4),config)
    loaded=load_app_config(str(directory));assert loaded.default_bin==2
    assert loaded._wavelength_profile==old

def test_runtime_has_no_old_config_dependency():
    root=Path(__file__).resolve().parents[1]
    for file in ['config.py','console.py','windows/main_window.py','modules/configuration/store.py']:
        assert 'backend.json' not in (root/file).read_text(encoding='utf-8')


def test_sections_are_derived_from_module_owned_dataclasses():
    """Each module owns its settings; AppConfig only composes them and the store follows."""
    from dataclasses import fields
    from config import AppConfig, config_sections, SECTION_CLASSES
    from modules.configuration.store import SECTIONS
    assert config_sections() == {k: v for k, v in SECTIONS.items() if k in config_sections()} or \
        {k: set(v) for k, v in config_sections().items()} == {k: set(v) for k, v in SECTIONS.items()}
    names = {f.name for f in fields(AppConfig)}
    for cls in SECTION_CLASSES.values():
        assert {f.name for f in fields(cls)} <= names               # nothing a module declares is lost
    assert AppConfig().camera_backend == 'poa' and AppConfig().resolution == 2

def test_profile_timing_from_processing_json_defaults_on(tmp_path):
    p=settings(tmp_path);proc=p/'processing.json';data=json.loads(proc.read_text())
    assert data['profile_timing'] is True and load_app_config(str(p)).profile_timing is True
    data['profile_timing']=False;proc.write_text(json.dumps(data))
    assert load_app_config(str(p)).profile_timing is False
    del data['profile_timing'];proc.write_text(json.dumps(data))      # older processing.json
    assert load_app_config(str(p)).profile_timing is True
