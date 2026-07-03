"""One-shot hotkey diagnostic: logs 30 s of key press/release events.

Run from Terminal:  .venv/bin/python keytest.py
Writes /tmp/localflow-keytest.log (and echoes to the screen).
"""

import time

from pynput import keyboard

LOG = "/tmp/localflow-keytest.log"
start = time.time()
out = open(LOG, "w")


def emit(line):
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


def down(k):
    emit(f"{time.time() - start:6.2f}s DOWN {k} vk={getattr(k, 'vk', None)}")


def up(k):
    emit(f"{time.time() - start:6.2f}s UP   {k} vk={getattr(k, 'vk', None)}")


print("For the next 30 seconds:")
print("  1) type: abc")
print("  2) hold Fn ALONE for 3 seconds, release")
print("  3) hold Fn, then ALSO hold Cmd for 2 seconds, release both")
listener = keyboard.Listener(on_press=down, on_release=up)
listener.start()
time.sleep(30)
listener.stop()
emit("-- capture finished --")
out.close()
print(f"log written to {LOG}")
