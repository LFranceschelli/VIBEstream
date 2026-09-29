"""VIBEstream - dummy entry point (placeholder, to be replaced).

Single script to run. Select what to do with the flags below.
"""

# ---- Execution flags -------------------------------------------------------
DO_STREAM = False    # live stream from the event camera
DO_RECORD = False    # record events to disk
DO_PROCESS = True    # offline processing of a recorded file
# ---------------------------------------------------------------------------


class Camera:
    """Placeholder camera class (no hardware access yet)."""

    def stream(self):
        print("[dummy] streaming...")

    def record(self, path="events.raw"):
        print(f"[dummy] recording to {path}")


def process(path="events.raw"):
    print(f"[dummy] processing {path}")


if __name__ == "__main__":
    cam = Camera()
    if DO_STREAM:
        cam.stream()
    if DO_RECORD:
        cam.record()
    if DO_PROCESS:
        process()
