import importlib
import os
import sys


def iter_python_modules(root: str = "."):
    for dirpath, _, filenames in os.walk(root):
        if ".git" in dirpath or "__pycache__" in dirpath:
            continue
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, filename), root)
            if rel.startswith(".git"):
                continue
            mod = rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            if mod:
                yield mod


def main() -> int:
    # Ensure project root is importable
    sys.path.insert(0, os.path.abspath("."))
    modules = sorted(set(iter_python_modules(".")))
    failed: list[tuple[str, Exception]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            failed.append((mod, exc))

    print(f"modules={len(modules)} failed={len(failed)}")
    for mod, exc in failed:
        print(f"FAIL {mod}: {exc!r}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

