from click.testing import CliRunner

from epitopecraft.main import cli


def test_cli_help_does_not_import_heavy_modeling_backends():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "pipeline" in result.output
    assert "refold" in result.output
    assert "inspect" in result.output
    assert "redesign" in result.output
    assert "standard-design" in result.output
