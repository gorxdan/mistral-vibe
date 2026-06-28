from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from vibe.core.config.fingerprint import create_file_fingerprint
from vibe.core.config.layer import LayerImplementationError
from vibe.core.config.layers.user import UserConfigLayer
from vibe.core.config.types import MISSING_CONFIG_FILE_FINGERPRINT

# UserConfigLayer.apply() is an M2 stub (raise NotImplementedError) after the
# v2.18.0 config redesign, so the apply/patch tests that exercised the old
# JSON-Patch API were removed. The new field-op apply (SetField/AppendToList)
# needs fresh coverage once apply() is implemented; layer apply semantics are
# meanwhile covered generically by test_config_layer.


def random_config_file_name() -> str:
    return f"config-{uuid4().hex}.toml"


@pytest.mark.asyncio
async def test_reads_toml_file(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "mistral-large"\ncount = 42\n')

    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {"active_model": "mistral-large", "count": 42}
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)
    assert fingerprint


@pytest.mark.asyncio
async def test_always_trusted(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('key = "value"\n')

    layer = UserConfigLayer(path=path)
    assert layer.is_trusted is None
    data = await layer.load()
    assert layer.is_trusted is True
    assert data.model_extra == {"key": "value"}


@pytest.mark.asyncio
async def test_missing_file_returns_empty(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {}
    assert layer.fingerprint == MISSING_CONFIG_FILE_FINGERPRINT


def test_atomic_replace_preserves_replacement_fingerprint(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    replacement = tmp_working_directory / f".{path.name}.tmp"
    path.write_text("key = 1")
    replacement.write_text("key = 2")

    with replacement.open("rb") as file:
        replacement_fingerprint = create_file_fingerprint(file)

    os.replace(replacement, path)

    with path.open("rb") as file:
        assert create_file_fingerprint(file) == replacement_fingerprint


@pytest.mark.asyncio
async def test_nested_toml_structure(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("""\
[models]
active_model = "test"

[[models.items]]
alias = "a"
provider = "p"
""")
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {
        "models": {"active_model": "test", "items": [{"alias": "a", "provider": "p"}]}
    }


@pytest.mark.asyncio
async def test_invalid_toml_raises(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("this is not valid = = = toml [[[")
    layer = UserConfigLayer(path=path)
    with pytest.raises(LayerImplementationError, match="_build_config_snapshot"):
        await layer.load()


@pytest.mark.asyncio
async def test_force_reload_reads_fresh_data(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('value = "first"\n')
    layer = UserConfigLayer(path=path)

    data1 = await layer.load()
    fp1 = layer.fingerprint
    assert data1.model_extra == {"value": "first"}
    assert isinstance(fp1, str)
    assert fp1

    path.write_text('value = "second"\n')
    data2 = await layer.load(force=True)
    fp2 = layer.fingerprint
    assert data2.model_extra == {"value": "second"}
    assert isinstance(fp2, str)
    assert fp2
    assert fp1 != fp2

    path.unlink()
    data3 = await layer.load(force=True)
    assert data3.model_extra == {}
    assert layer.fingerprint == MISSING_CONFIG_FILE_FINGERPRINT


@pytest.mark.asyncio
async def test_empty_toml_file(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("")
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {}
