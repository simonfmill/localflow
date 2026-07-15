"""Entrypoint: `python -m localflow`."""


LOG_FILE = "~/Library/Logs/LocalFlow.log"


def main():
    import logging
    from pathlib import Path

    import localflow
    from localflow.app import build_default
    from localflow.config import load_config

    log_path = Path(LOG_FILE).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    cfg = load_config()
    logging.getLogger("localflow").info(
        "LocalFlow %s starting — hotkey %s, correction %s, log file %s",
        localflow.__version__, cfg["hotkey"]["combo"],
        cfg["hotkey"].get("correction_combo"), log_path)
    app = build_default(cfg)
    print(f"LocalFlow running — hold {cfg['hotkey']['combo']} to dictate. Menubar icon: 🎙")
    app.run()


if __name__ == "__main__":
    main()
