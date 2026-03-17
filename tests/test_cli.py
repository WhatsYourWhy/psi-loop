from pathlib import Path

from psi_loop import cli


def test_cli_lists_bundled_tasks(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["psi-loop", "--list-tasks"])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "retry_backoff" in output


def test_cli_runs_with_default_bundled_fixture(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["psi-loop", "--task", "retry_backoff"])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Psi0 selection" in output
    assert "Baseline selection" in output


def test_cli_accepts_explicit_fixture(monkeypatch, capsys):
    fixture = Path(__file__).parent / "fixtures" / "sample_tasks.json"
    monkeypatch.setattr(
        "sys.argv",
        ["psi-loop", "--fixture", str(fixture), "--task", "retry_backoff"],
    )

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "novel_backoff_jitter" in output
