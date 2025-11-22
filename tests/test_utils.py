from pathlib import Path

import pytest

from src.utils import getData


def test_get_data_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "sample.csv"
    bad_file.write_text("header1,header2\nvalue1,value2", encoding="utf-8")

    with pytest.raises(ValueError):
        getData(str(bad_file))
