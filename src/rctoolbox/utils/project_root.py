from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def find_project_root(
    start: Path | str | None = None,
    markers: tuple[str, ...] = ("pyproject.toml", ".git"),
) -> Path:
    """Locate the thermal-rc-toolbox repository root.

    Walks up from ``start`` (if given), otherwise from this module's own
    file, then from ``Path.cwd()``, and returns the first ancestor directory
    that contains any of ``markers``.

    Anchoring on ``__file__`` first makes the result independent of where
    Jupyter/Python was launched from: for an editable install the source
    tree is inside the repo, so the walk lands on the toolbox root even
    when the current working directory sits in a different project.
    ``Path.cwd()`` is only used as a fallback (e.g. when the package was
    installed from a wheel).

    Raises
    ------
    RuntimeError
        If no ancestor of any candidate contains any of the markers.
    """
    if start is not None:
        candidates = [Path(start).resolve()]
    else:
        candidates = [_THIS_FILE, Path.cwd().resolve()]

    for p in candidates:
        for ancestor in (p, *p.parents):
            if any((ancestor / m).exists() for m in markers):
                return ancestor

    raise RuntimeError(
        f"Project root not found: none of {markers} present in any parent "
        f"of {[str(c) for c in candidates]}."
    )
