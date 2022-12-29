import multiprocessing as mp
import numpy as np
from scipy import signal
import pyaudio
import pygame
import io

import matplotlib.pyplot as plot


import wave
import aubio


def setup():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=512,
    )
    return pa, stream


def hud(**kw):
    text = kw["font"].render(kw["text"], True, white, black)
    return text


def main():
    pygame.init()

    font = pygame.font.Font("freesansbold.ttf", 12)
    # Set up the display
    screen = pygame.display.set_mode((1920, 1080))

    pa, stream = setup()

    samples = np.empty(0)
    i = 0

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("quit!!! now")
                    running = False

        # Update the game state
        sample = np.fromstring(stream.read(512), dtype=aubio.float_type)

        samples = np.concatenate((samples, sample), axis=0)[-200 * 1024 :]

        screen.fill((0, 0, 0))

        if i % 100 == 0:
            plot.specgram(samples, Fs=44100, NFFT=1024)
            ax = plot.gca()
            ax.set_ylim([0, 10000])
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            buf = io.BytesIO()
            plot.savefig(buf, format="png", dpi=199, bbox_inches="tight")
            buf.seek(0)

            im = pygame.image.load(
                buf,
            )
            im_rect = im.get_rect()
            im_rect.center = (1920 // 2, 1080 // 2)

            # im.show()
            buf.close()

        # Draw the screen
        screen.blit(im, im_rect)

        text = hud(text=f"i={i}", font=font)
        textRect = text.get_rect()
        # set the center of the rectangular object.
        textRect.center = (1920 // 2, 10)
        screen.blit(text, textRect)

        pygame.display.flip()
        pygame.time.delay(1000 // (44100 // 8))
        i += 1


white = (255, 255, 255)
black = (0, 0, 0)
main()
