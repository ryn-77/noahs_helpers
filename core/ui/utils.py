import pygame
from typing import Literal


def write_at(
    screen: pygame.Surface,
    font: pygame.font.Font,
    line: str,
    coord: tuple[int, int],
    align: Literal["left", "center", "right"] = "center",
    color=(0, 0, 0),
):
    text = font.render(line, True, color)

    # get rectangle to center the text
    match align:
        case x if x == "left":
            rect = text.get_rect()
            rect.midleft = coord
        case x if x == "center":
            rect = text.get_rect(center=coord)
        case x if x == "right":
            rect = text.get_rect()
            rect.midright = coord
        case _:
            raise Exception(f"invalid value for `align`: {align}")

    screen.blit(text, rect)


def render_img(
    screen: pygame.Surface, coord: tuple[int, int], img_path: str, scale: int = 1
):
    img_orig = pygame.image.load(img_path).convert_alpha()
    img = pygame.transform.scale(img_orig, (scale, scale))
    rect = img.get_rect(center=coord)
    screen.blit(img, rect)
