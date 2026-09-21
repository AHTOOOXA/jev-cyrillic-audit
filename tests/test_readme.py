from pathlib import Path


def test_readme_embeds_the_current_results_tables():
    """`make reproduce` regenerates results.md and results2.md; the README must carry those exact tables."""
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    for name in ("results.md", "results2.md"):
        table = (root / name).read_text().strip()
        assert table and table in readme, name
