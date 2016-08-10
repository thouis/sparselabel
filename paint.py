import sys
import glob
import os

import numpy as np

import imread
import pygame
from pygame.constants import MOUSEBUTTONDOWN, MOUSEMOTION, BLEND_PREMULTIPLIED, SRCALPHA

def get_image_size(imfile):
    return imread.imread(imfile).shape


def get_labels(directory, index):
    label_filename = os.path.join(directory, 'labels_{:06d}.png'.format(index + 1))
    if os.path.exists(label_filename):
        return imread.imread(label_filename)
    return None


def put_labels(directory, index, label_data):
    label_filename = os.path.join(directory, 'labels_{:06d}.png'.format(index + 1))
    imread.imsave(label_filename, label_data)

def get_image(image_filename):
    return imread.imread(image_filename)


def make_background(image, previous_labels, next_labels, overlay_opacity=0.5):
    if previous_labels is None:
        previous_labels = 0 * image
    if next_labels is None:
        next_labels = 0 * image

    # previous layer = red overlay
    prev_mask = (previous_labels > 0)
    prev_alpha = overlay_opacity * prev_mask
    out_red = (prev_mask * prev_alpha * 255) + ((1.0 - prev_alpha) * image)

    # next layer = green overlay
    next_mask = (next_labels > 0)
    next_alpha = overlay_opacity * next_mask
    out_green = (next_mask * next_alpha * 255) + ((1.0 - next_alpha) * image)

    # stack to create image
    return pygame.surfarray.make_surface(np.dstack((out_red, out_green, image)))

# global
radius = 10

def annotate(bg_image, labels, screen):
    global radius
    erase_radius = 10

    label_surface = pygame.Surface(screen.get_size(), SRCALPHA)
    if labels is not None:
        pygame.surfarray.pixels_alpha(label_surface)[...] = labels * 0.5
        pygame.surfarray.pixels_blue(label_surface)[...] = labels * 0.5

    doquit = False
    doexit = False
    offset = 0
    while not doquit:
        screen.blit(bg_image, (0, 0))
        screen.blit(label_surface, (0, 0), special_flags=BLEND_PREMULTIPLIED)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                doquit = True
                doexit = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    offset = -1
                    doquit = True
                    break
                elif event.key == pygame.K_RIGHT:
                    offset = 1
                    doquit = True
                    break
                elif event.key == pygame.K_MINUS:
                    radius = max(1, radius - 1)
                elif event.key == pygame.K_PLUS:
                    radius += 1
            elif event.type == MOUSEBUTTONDOWN:
                x, y = lastpos = event.pos
                print (event.button)
                if event.button == 3:
                    pygame.draw.circle(label_surface, (0, 0, 0, 0), (x, y), erase_radius)
                elif event.button == 2:
                    pygame.draw.circle(label_surface, (0, 0, 128, 128), (x, y), radius)
            elif event.type == MOUSEMOTION:
                if event.buttons[2]:
                    x, y = event.pos
                    pygame.draw.line(label_surface, (0, 0, 0, 0), lastpos, (x, y), erase_radius * 2 + 1)
                    pygame.draw.circle(label_surface, (0, 0, 0, 0), (x, y), erase_radius)
                elif event.buttons[1]:
                    x, y = event.pos
                    pygame.draw.line(label_surface, (0, 0, 128, 128), lastpos, (x, y), radius * 2 + 1)
                    pygame.draw.circle(label_surface, (0, 0, 128, 128), (x, y), radius)
                lastpos = event.pos

    out_labels = (pygame.surfarray.pixels_blue(label_surface) > 0).astype(np.uint8) * 255
    return out_labels, offset, doexit

if __name__ == '__main__':
    pygame.init()

    image_directory = sys.argv[1]
    label_directory = sys.argv[2]

    image_files = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    current_index = 0

    screen = pygame.display.set_mode(get_image_size(image_files[0]))

    while True:
        previous_labels = get_labels(label_directory, current_index - 1)
        current_labels = get_labels(label_directory, current_index)
        next_labels = get_labels(label_directory, current_index + 1)

        current_image = get_image(image_files[current_index])

        background = make_background(current_image, previous_labels, next_labels)
        current_labels, offset, quit = annotate(background, current_labels, screen)
        put_labels(label_directory, current_index, current_labels)

        if quit:
            pygame.quit()
            sys.exit(0)

        current_index += offset
        if current_index < 0:
            current_index = 0
        elif current_index >= len(image_files):
            current_index = len(image_files) - 1
