"""Theme utilities for Skyline UI components."""

from pathlib import Path

SKYLINE_HOME = Path("~").expanduser() / ".dipy-skyline"


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex colors to rgba

    Parameters
    ----------
    hex_color : str
        Hexcode for the color.
    alpha : float, optional
        Transparency of the color.

    Returns
    -------
    tuple
        RGBA color tuple.
    """

    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


FONT = SKYLINE_HOME / "fonts" / "Inter_18pt-Regular.ttf"
FONT_AWESOME = SKYLINE_HOME / "fonts" / "fa-solid-900.otf"

LICENSE_DIRS = ("ofl", "apache", "ufl")
THEME = {
    "background": hex_to_rgba("#191919"),
    "primary": hex_to_rgba("#EE942E"),
    "secondary": hex_to_rgba("#FFFFFF"),
    "text": hex_to_rgba("#838383"),
    "text_highlight": hex_to_rgba("#EE942E"),
    "shadow": hex_to_rgba("#000000", alpha=0.12),
}

SLIDER_THEME = {
    "track_color": THEME["secondary"],
    "track_bg": THEME["background"],
    "thumb_color": THEME["primary"],
    "track_covered_color": THEME["primary"],
    "label_color": THEME["text"],
    "value_color": THEME["text_highlight"],
    "shadow_color": THEME["shadow"],
}

WINDOW_THEME = {
    "title_color": THEME["text"],
    "title_active_color": THEME["text_highlight"],
    "background_color": THEME["background"],
}

SWITCH_THEME = {
    "background_color": THEME["background"],
    "active_color": THEME["text"],
    "inactive_text_color": THEME["text"],
    "active_text_color": THEME["secondary"],
    "border_color": THEME["text"],
}
