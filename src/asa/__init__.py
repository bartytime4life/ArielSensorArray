__all__ = ["__version__"]
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("arielsensorarray")
except Exception:
    __version__ = "0.1.0"
