"""Light theme by default: new localStorage key (old visits saved "dark" on every load), fallback "light".

    python3 theme_light.py <file.html> ...
Idempotent.
"""
import re
import sys
from pathlib import Path

for name in sys.argv[1:]:
    p = Path(name)
    s = p.read_text(encoding="utf-8")
    if "cascadeThemeV2" in s:
        print(name, "already light")
        continue
    t = s.replace('localStorage.cascadeTheme || "dark"', 'localStorage.cascadeThemeV2 || "light"')
    t = t.replace("localStorage.cascadeTheme = ", "localStorage.cascadeThemeV2 = ")
    t = t.replace('setTheme(document.documentElement.dataset.theme || "dark")',
                  'setTheme(document.documentElement.dataset.theme || "light")')
    left = re.findall(r"localStorage\.cascadeTheme\b(?!V2)", t)
    assert not left, f"{name}: {len(left)} cascadeTheme uses left"
    assert t != s, f"{name}: nothing changed"
    p.write_text(t, encoding="utf-8")
    print(name, "light by default")
