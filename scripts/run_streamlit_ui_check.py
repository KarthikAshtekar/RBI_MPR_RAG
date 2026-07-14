from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/streamlit_ui_polish"
LOG_DIR = OUT / "run_logs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_free_port(preferred: int = 8501) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_http(url: str, timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except Exception:
            time.sleep(1)
    return False


def start_streamlit(port: int, stdout_path: Path, stderr_path: Path) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def scroll_to_text(page: Any, text: str) -> None:
    page.evaluate(
        """
        (needle) => {
            const elements = Array.from(document.querySelectorAll('h1,h2,h3,p,div,span'));
            const el = elements.find(node => (node.innerText || '').includes(needle));
            if (el) {
                el.scrollIntoView({behavior: 'instant', block: 'start'});
                window.scrollBy(0, -80);
            }
        }
        """,
        text,
    )
    # Streamlit commonly scrolls inside an app container rather than the window.
    # Wheel events are more reliable across current Streamlit DOM variants.
    wheel_map = {
        "Try a saved demo question": 550,
        "Citations": 650,
        "Method comparison": 1450,
        "Limitations": 2100,
    }
    if text in wheel_map:
        page.mouse.wheel(0, wheel_map[text])
    page.wait_for_timeout(400)


def capture_screenshots(url: str, iteration_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at_utc": now_iso(),
        "url": url,
        "screenshots": [],
        "browser_status": "not_started",
        "visible_error_terms": [],
    }
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        manifest["browser_status"] = "playwright_import_failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        return manifest

    iteration_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            manifest["browser_status"] = "browser_launch_failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            return manifest
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.goto(url, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(1000)
            text = page.locator("body").inner_text(timeout=5000)
            for term in ["Traceback", "ModuleNotFoundError", "Exception", "KeyError"]:
                if term in text:
                    manifest["visible_error_terms"].append(term)
            captures = [
                ("home_desktop.png", 1440, 900, None),
                ("query_demo_desktop.png", 1440, 900, "Try a saved demo question"),
                ("answer_citations_desktop.png", 1440, 900, "Citations"),
                ("comparison_desktop.png", 1440, 900, "Method comparison"),
                ("limitations_desktop.png", 1440, 900, "Limitations"),
                ("home_1366x768.png", 1366, 768, None),
                ("home_1280x720.png", 1280, 720, None),
                ("home_mobile.png", 390, 844, None),
            ]
            for name, width, height, anchor in captures:
                page.set_viewport_size({"width": width, "height": height})
                if anchor:
                    scroll_to_text(page, anchor)
                else:
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(300)
                path = iteration_dir / name
                page.screenshot(path=str(path), full_page=False)
                manifest["screenshots"].append(
                    {"name": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "width": width, "height": height, "anchor": anchor}
                )
            full = iteration_dir / "full_page_desktop.png"
            page.set_viewport_size({"width": 1440, "height": 900})
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(full), full_page=True)
            manifest["screenshots"].append(
                {"name": "full_page_desktop.png", "path": str(full.relative_to(ROOT)).replace("\\", "/"), "width": 1440, "height": 900, "full_page": True}
            )
            manifest["browser_status"] = "screenshots_captured"
        finally:
            browser.close()
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    port = find_free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    stdout_path = LOG_DIR / f"iteration_{args.iteration}_streamlit_stdout.log"
    stderr_path = LOG_DIR / f"iteration_{args.iteration}_streamlit_stderr.log"
    process = start_streamlit(port, stdout_path, stderr_path)
    ready = wait_for_http(url)
    manifest = {
        "created_at_utc": now_iso(),
        "iteration": args.iteration,
        "requested_port": args.port,
        "port": port,
        "url": url,
        "streamlit_ready": ready,
        "stdout_log": str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
        "stderr_log": str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
        "process_returncode_before_stop": process.poll(),
    }
    screenshot_dir = OUT / "screenshots" / f"iteration_{args.iteration}"
    try:
        if ready:
            screenshot_manifest = capture_screenshots(url, screenshot_dir)
        else:
            screenshot_manifest = {"browser_status": "not_run_streamlit_not_ready", "screenshots": []}
        manifest["screenshot_manifest"] = screenshot_manifest
        write_json(screenshot_dir / "screenshot_manifest.json", screenshot_manifest)
    finally:
        stop_process(process)
        manifest["process_returncode_after_stop"] = process.poll()
    write_json(OUT / f"iteration_{args.iteration}_run_manifest.json", manifest)
    lines = [
        f"# Streamlit UI Check Iteration {args.iteration}",
        "",
        f"Streamlit ready: `{ready}`",
        f"URL: `{url}`",
        f"Screenshot status: `{manifest['screenshot_manifest'].get('browser_status')}`",
        f"Screenshot count: {len(manifest['screenshot_manifest'].get('screenshots', []))}",
        f"Stdout log: `{manifest['stdout_log']}`",
        f"Stderr log: `{manifest['stderr_log']}`",
    ]
    if manifest["screenshot_manifest"].get("error"):
        lines.append(f"Error: `{manifest['screenshot_manifest']['error']}`")
    write_md(OUT / f"iteration_{args.iteration}_run_manifest.md", lines)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
