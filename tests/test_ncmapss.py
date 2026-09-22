
from rul.data.ncmapss import load_ncmapss


def test_load_ncmapss_shapes_and_columns(synthetic_ncmapss_path):
    f = load_ncmapss(synthetic_ncmapss_path)

    assert f.dev.columns["W"] == ["alt", "Mach", "TRA", "T2"]
    assert f.dev.X_s.shape[1] == 3
    assert f.dev.W.shape[0] == f.dev.Y.shape[0]

    assert set(f.dev.unit_ids.tolist()) == {1.0, 2.0, 3.0}
    assert set(f.test.unit_ids.tolist()) == {11.0}


def test_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_ncmapss(tmp_path / "does_not_exist.h5")
