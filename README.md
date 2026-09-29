# VIBEstream

Event-based imaging velocimetry (EBIV) in Python: an importable library plus a GUI to stream, record and process event-camera data.

> **Status:** early placeholder. The code in this repository is a dummy skeleton; no functionality is implemented yet.

## Planned structure

- `Camera` class for live streaming and recording (e.g. `cam.record()`)
- Offline processing of recorded event files
- GUI front-end for the above

## Usage (dummy)

Set the execution flags at the top of `main.py` (`DO_STREAM`, `DO_RECORD`, `DO_PROCESS`) and run:

```bash
python main.py
```

## Package (dummy)

```python
import vibestream
print(vibestream.hello())
```

## Author

Luca Franceschelli, Universidad Carlos III de Madrid (UC3M)
