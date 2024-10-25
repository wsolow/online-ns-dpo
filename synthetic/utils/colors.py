
COLORS = [
    "darkblue",
    "orangered",
    "darkviolet",
    "darkorange",
    "limegreen",
    "saddlebrown",
    "gold",
    "slategrey"
]

COLORS = [
    "darkblue",
    "orangered",
    "darkviolet",
    "darkorange",
    "limegreen",
    "saddlebrown",
    "gold",
    "slategrey"
]

COLORS_B = [
    "darkblue",
    "dodgerblue",
    "blue",
    "darkcyan",
    "cyan",
    "lime",
    "mediumseagreen",
    "darkgreen"
]

COLORS_R = [
    "tomato",
    "darkred",
    "red",
    "darkorange",
    "saddlebrown",
    "chocolate",
    "gold",
    "magenta"
]

COLORS_G = [
    "darkgreen",
    "lime",
    "teal",
    "blueviolet",
    "violet",
    "fuchsia",
    "grey",
]

class ColorRevolver:
    def __init__(self, reverse=False, colorset=""):
        if colorset == "B":
            self.colors = COLORS_B
        elif colorset == "R":
            self.colors = COLORS_R
        elif colorset == "G":
            self.colors = COLORS_G
        else:
            self.colors = COLORS
        self.reverse = reverse
        self.idx = -1

    def get_color(self):
        self.idx = (self.idx + 1) % len(self.colors)
        if self.reverse:
            res = self.colors[-self.idx-1]
        else:
            res = self.colors[self.idx]
        return res
