"""Launcher for the standalone ForgeCal Post EXE (PyInstaller).

Double-clicking the frozen EXE starts a LOCAL Streamlit server and opens the
post-processing app in the browser — no Python install needed. Everything runs
on the user's own machine (no server load) and is bound to localhost only.
"""
import os
import socket
import sys

os.environ.setdefault("MPLBACKEND", "Agg")   # no GUI backend (figures -> images)


def _pick_port(preferred=8765):
    """Find a free localhost port so the EXE never collides with whatever else
    a user happens to run (Grafana on 3000, other dev servers, etc.). Tries a
    friendly fixed port first, then a small range, then any free port."""
    for p in [preferred, *range(8766, 8800)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))                  # OS picks any free port
    p = s.getsockname()[1]
    s.close()
    return p


def _app_path():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "post_app.py")


def main():
    from streamlit import config
    from streamlit.web import bootstrap
    # populate the config registry first (otherwise _set_option asserts that
    # _config_options is empty)
    config.get_config_options()
    # PyInstaller puts streamlit OUTSIDE site-packages, which streamlit
    # mis-reads as 'developmentMode' (then it serves the frontend from a
    # non-existent Node dev server on :3000 and ignores server.port). Force
    # production mode so it serves the bundled frontend on our chosen port.
    config._set_option("global.developmentMode", False, "frozen app")
    config._set_option("server.address", "localhost", "frozen app")  # local only
    config._set_option("server.port", _pick_port(), "frozen app")    # free port
    config._set_option("browser.gatherUsageStats", False, "frozen app")
    # ship: open the browser automatically (to the chosen port). Tests set
    # FORGECAL_HEADLESS=1 to skip the auto-open.
    headless = os.environ.get("FORGECAL_HEADLESS", "0") == "1"
    config._set_option("server.headless", headless, "frozen app")
    bootstrap.run(_app_path(), False, [], {})


if __name__ == "__main__":
    main()
