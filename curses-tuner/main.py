import math
import logging
import curses
import time
import statistics

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import aubio
import pyaudio
import wave


from time import sleep

logging.basicConfig(
    filename="debug.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger("curses")

p_detect = aubio.pitch(method="default", buf_size=1024, hop_size=1024, samplerate=44100)


def make_spec(sig, fs):
    f, t, Sxx = signal.spectrogram(sig, fs)

    # Plot the spectrogram
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


def scale_to_interval(value, min_value, max_value, m, n):
    # Scale the value to the interval [m, n]
    scaled_value = m + (value - min_value) * (n - m) / (max_value - min_value)

    return scaled_value


def scale(values):
    # Compute the mean and standard deviation of the values
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    # Scale the values
    scaled_values = [(value - mean) / stdev for value in values]

    return scaled_values


def beats_to_bpm(beats):
    if len(beats) > 1:
        bpms = 60.0 / np.diff(beats)
        return np.median(bpms)
    else:
        return 0


def generate_notes_dict():
    # List of musical notes
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # Dictionary of musical notes and their corresponding frequencies
    notes_dict = {}
    # Iterate over octaves
    for octave in range(10):
        # Iterate over notes
        for note in notes:
            # Calculate the frequency of the note
            frequency = 440.0 * (2.0 ** ((octave - 4) + (notes.index(note) - 9) / 12.0))
            # Add the note to the dictionary
            notes_dict[f"{note}{octave}"] = frequency
    return notes_dict


NOTES = generate_notes_dict()


def get_note(frequency, notes, threshold=50.0):
    # Convert the threshold from cents to Hz
    threshold_hz = 2 ** (threshold / 1200.0)
    # Find the closest note to the given frequency
    closest_note = min(notes, key=lambda x: abs(notes[x] - frequency))
    # Check if the difference between the frequency and the closest note is within the threshold
    if abs(notes[closest_note] - frequency) <= threshold_hz:
        # Return the closest note
        return closest_note
    else:
        # Return None if the difference is greater than the threshold
        return None


def get_closest_note(frequency, notes):
    if frequency <= 0:
        return None

    # Find the closest note to the given frequency
    closest_note = min(notes, key=lambda x: abs(notes[x] - frequency))
    # Calculate the number of semitones between the frequency and the closest note
    semitones = 12 * math.log2(frequency / notes[closest_note])
    # Check the sign of semitones
    if semitones > 0:
        # Return the next higher note
        notes_list = list(notes.keys())
        return notes_list[notes_list.index(closest_note) + 1]
    elif semitones < 0:
        # Return the next lower note
        notes_list = list(notes.keys())
        return notes_list[notes_list.index(closest_note) - 1]
    else:
        # Return the closest note
        return closest_note


def handle_events(stdscr, app):
    ch = stdscr.getch()

    # Check the character and do something based on it
    if ch == curses.KEY_UP:
        app["th"] += 0.02
    elif ch == curses.KEY_DOWN:
        app["th"] -= 0.02
    elif ch == curses.KEY_ENTER:
        pass

    return app


def setup():
    app = {"th": 0.5}
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024,
    )
    return app, pa, stream


def main(stdscr):
    app, pa, stream = setup()

    # Set the terminal window to non-blocking mode
    stdscr.nodelay(1)
    # hide cursor
    curses.curs_set(0)
    y_limit, x_limit = stdscr.getmaxyx()

    samples = np.empty(0)
    beats = []

    # Clear the screen
    stdscr.clear()

    # Set the cursor position
    stdscr.move(0, 0)
    p = aubio.pitch("default", 2048, 2048 // 2, 44100)
    t = aubio.tempo("specdiff", 2048, 2048 // 2, 44100)

    while True:
        sample = np.fromstring(stream.read(1024), dtype=aubio.float_type)

        is_beat = t(sample.astype(np.float32))
        if is_beat:
            beats.append(t.get_last_s())

        pitch = p(sample)[0]
        volume = np.sum(sample**2) / len(sample)
        s_volume = round(math.log(volume), 1)
        s_pitch = round(pitch, 1)
        key = get_closest_note(pitch, NOTES)
        is_tuned = get_note(pitch, NOTES, 50) == key

        # Clear the screen
        stdscr.clear()
        y_limit, x_limit = stdscr.getmaxyx()

        samples = np.concatenate((samples, sample), axis=0)
        samples = samples[-500 * 1024 :]
        # make_spec(samples, 44100)

        pos = 1  # int(y_limit // 1.1)
        stdscr.addstr(pos, 0, f"  Vol: {s_volume}db")
        stdscr.addstr(pos + 1, 0, f"    F: {s_pitch}Hz")
        stdscr.addstr(pos + 2, 0, f"  Key: {key}{'✓' if is_tuned else ''}")
        stdscr.addstr(
            pos + 3,
            0,
            f"    t: {round(beats_to_bpm(beats), 1)}bpm",
        )
        stdscr.addstr(pos + 4, 0, f"  Len: {len(samples)/44100}s")

        stdscr.refresh()
        # time.sleep(0.1)


# Initialize curses
cursed = curses.initscr()

# Run the main function
try:
    curses.wrapper(main)
except Exception as e:
    curses.endwin()
    print("crashed with", e)
    raise e
    exit(1)

# End curses
