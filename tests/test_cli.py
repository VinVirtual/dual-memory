from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_memory.cli import main


def test_generate_and_bench(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "d.json"
    assert main(["generate", "--out", str(data), "--personas", "2", "--sessions", "4", "--seed", "1"]) == 0
    assert "2 personas" in capsys.readouterr().out
    out = tmp_path / "res"
    code = main(
        [
            "bench",
            "--data",
            str(data),
            "--out",
            str(out),
            "--budget",
            "128",
            "--budgets",
            "128,256",
            "--methods",
            "no-memory,dual-memory",
            "--no-chart",
        ]
    )
    assert code == 0
    res = json.loads((out / "results.json").read_text())
    assert [r["method"] for r in res["results"]] == ["no-memory", "dual-memory"]
    assert (out / "budget-256" / "results.json").exists()
    assert len(json.loads((out / "sweep.json").read_text())) == 2


def test_chart_command(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    data = tmp_path / "d.json"
    main(["generate", "--out", str(data), "--personas", "1", "--sessions", "3"])
    out = tmp_path / "res"
    main(
        [
            "bench",
            "--data",
            str(data),
            "--out",
            str(out),
            "--methods",
            "dual-memory,rag-budget-filled",
            "--no-chart",
        ]
    )
    assert main(["chart", str(out)]) == 0
    assert (out / "chart.png").exists()


def test_bad_input_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "--out", str(tmp_path / "x.json"), "--personas", "0"]) == 2
    assert main(["bench", "--data", str(tmp_path / "missing.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_method_exits_2(tmp_path: Path) -> None:
    data = tmp_path / "d.json"
    main(["generate", "--out", str(data), "--personas", "1", "--sessions", "3"])
    assert (
        main(["bench", "--data", str(data), "--out", str(tmp_path / "r"), "--methods", "nope", "--no-chart"])
        == 2
    )


def test_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == 0
    assert "FTS5: ok" in capsys.readouterr().out


def test_compare_results_script(tmp_path: Path) -> None:
    import subprocess
    import sys

    ref = {"results": [{"method": "dual-memory", "accuracy": 0.9, "by_type": {"current": 1.0}}]}
    (tmp_path / "a.json").write_text(json.dumps(ref))
    (tmp_path / "b.json").write_text(json.dumps(ref))
    script = Path(__file__).parent.parent / "scripts" / "compare_results.py"
    ok = subprocess.run([sys.executable, str(script), str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert ok.returncode == 0
    ref["results"][0]["accuracy"] = 0.8
    (tmp_path / "b.json").write_text(json.dumps(ref))
    bad = subprocess.run([sys.executable, str(script), str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert bad.returncode == 1
