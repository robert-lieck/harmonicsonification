#  Copyright (c) 2025 Robert Lieck
import random
import IPython.display
import numpy as np
from scipy.io.wavfile import write as write_wav
import matplotlib.pyplot as plt


sampling_rate = 44000


def seed_everything(seed):
    """Set random seed for reproducibility across multiple libraries."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module


def normalise_wave(wave, max_amp=0.95):
    wave /= abs(wave).max()
    wave *= max_amp


def fade_wave(wave, time=0.01, start=True, end=True):
    # ramp of correct length
    fade_vals = np.linspace(0, 1, int(np.ceil(time * sampling_rate)))
    if start:
        wave[:len(fade_vals)] *= fade_vals
    if end:
        wave[-len(fade_vals):] *= np.flip(fade_vals)


def render(wave, normalise=True, fade=True):
    wave = wave.copy()
    if normalise is True:
        normalise = dict()
    if normalise is not False:
        normalise_wave(wave, **normalise)
    if fade is True:
        fade = dict()
    if fade is not False:
        fade_wave(wave, **fade)
    return wave


def save(wave, file_name, normalise=True, fade=True):
    wave = render(wave=wave, normalise=normalise, fade=fade)
    # convert to 16bit integer
    wave = np.int16(wave * (np.iinfo(np.int16).max - 1))
    write_wav(file_name, sampling_rate, wave)


def audio(wave, fade=True):
    wave = render(wave=wave, normalise=False, fade=fade)
    IPython.display.display(IPython.display.Audio(data=wave, rate=sampling_rate))


def sound(func, phases=0., duration=1.):
    # time vector
    time = np.arange(0, duration, 1 / sampling_rate)
    # array of unit-angle steps (corresponding to frequency of 1Hz)
    angle_steps = np.full_like(time, duration * 2 * np.pi / len(time))
    # get frequencies and amplitudes over time
    if callable(func):
        freq_amps = func(time)
    else:
        freq_amps = func
    if not isinstance(freq_amps, tuple):
        freqs = freq_amps
        amps = 1.
    else:
        freqs, amps = freq_amps
    freqs = np.atleast_2d(freqs)
    amps = np.atleast_2d(amps)
    # effective change of angle for different frequencies (time-dependent)
    angle_steps = angle_steps[:, None] * freqs
    # actual angle at given time corresponds to the accumulated angle steps
    angles = np.cumsum(angle_steps, axis=0) + phases
    # generate oscillations, multiply by amplitudes and sum up
    return (amps * np.sin(angles)).sum(axis=1)


def harmonic_tone(f0, decay=1, n=20, amps=None, **kwargs):
    if amps is not None:
        n = len(amps)
    else:
        amps = [np.exp(-i * decay) for i in range(0, n)]
    freqs = [f0 * i for i in range(1, n + 1)]
    return sound(lambda time: (freqs, amps), **kwargs)
