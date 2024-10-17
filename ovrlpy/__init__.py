from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ovrlpy")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version
