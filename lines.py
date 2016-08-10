import sys
import glob
import os
import json

import imread
import pygame
from pygame.constants import MOUSEBUTTONDOWN, MOUSEMOTION

def get_image_size(imfile):
    return imread.imread(imfile).shape


def get_image(image_filename):
    return imread.imread(image_filename)

def get_labels(directory, index):
    label_filename = os.path.join(directory, 'labels_{:06d}.json'.format(index + 1))
    try:
        with open(label_filename, "r") as f:
            return json.load(f)
    except IOError:
        return []


def put_labels(directory, index, label_data):
    label_filename = os.path.join(directory, 'labels_{:06d}.json'.format(index + 1))
    json.dump(label_data, open(label_filename, "w"), indent=2)


def annotate(image, labels, screen):
    doquit = False
    doexit = False
    offset = 0
    active_label = []

    imsurface = pygame.surfarray.make_surface(image[..., None].repeat(3, -1))

    while not doquit:
        screen.blit(imsurface, (0, 0))
        for l in labels:
            if len(l) > 2:
                pygame.draw.lines(screen, (255, 0, 0), False, l, 2)
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
                    active_label = []
                    if len(labels) > 0:
                        labels.pop()
                    break
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 3:
                    # erase
                    pass
                elif event.button == 1:
                    active_label = [event.pos]
                    labels.append(active_label)
            elif event.type == MOUSEMOTION:
                if event.buttons[2]:
                    x, y = event.pos
                elif event.buttons[0]:
                    active_label.append(event.pos)

    return labels, offset, doexit

if __name__ == '__main__':
    pygame.init()

    image_directory = sys.argv[1]
    label_directory = sys.argv[2]

    image_files = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    current_index = 0

    screen = pygame.display.set_mode(get_image_size(image_files[0]))

    while True:
        current_image = get_image(image_files[current_index])
        current_labels = get_labels(label_directory, current_index)

        current_labels, offset, quit = annotate(current_image, current_labels, screen)
        put_labels(label_directory, current_index, current_labels)

        if quit:
            pygame.quit()
            sys.exit(0)

        current_index += offset
        if current_index < 0:
            current_index = 0
        elif current_index >= len(image_files):
            current_index = len(image_files) - 1
