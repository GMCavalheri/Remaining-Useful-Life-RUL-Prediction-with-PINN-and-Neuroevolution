import textwrap

from rul.config import load_config


def test_load_simple_config(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(textwrap.dedent("""
        model:
          hidden_size: 64
        lr: 0.001
    """))

    config = load_config(path)
    assert config["model"]["hidden_size"] == 64
    assert config["lr"] == 0.001


def test_load_config_with_defaults_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(textwrap.dedent("""
        model:
          hidden_size: 64
          activation: relu
        lr: 0.001
    """))

    child = tmp_path / "child.yaml"
    child.write_text(textwrap.dedent("""
        defaults: base.yaml
        model:
          hidden_size: 128
    """))

    config = load_config(child)
    # overridden value
    assert config["model"]["hidden_size"] == 128
    # inherited value, untouched
    assert config["model"]["activation"] == "relu"
    assert config["lr"] == 0.001
