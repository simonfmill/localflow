"""Entrypoint: `python -m localflow`."""


def main():
    import logging

    from localflow.app import build_default
    from localflow.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    app = build_default(cfg)
    print(f"LocalFlow running — hold {cfg['hotkey']['combo']} to dictate. Menubar icon: 🎙")
    app.run()


if __name__ == "__main__":
    main()
