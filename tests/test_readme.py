from pathlib import Path


def test_readme_embeds_the_current_results_table():
    """`make reproduce` regenerates results.md; the README must carry that exact table."""
    root = Path(__file__).resolve().parents[1]
    table = (root / "results.md").read_text().strip()
    assert table and table in (root / "README.md").read_text()
