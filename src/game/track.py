"""Race track of the game: the drawing, the road mask and the zones (finish, checkpoints, STOP, traffic light).

The track is drawn only once, when the game starts. Every frame then copies that image
and draws on top only what changes (the traffic light, the car and the texts).

How it follows the course labs:
- The road is a chain of very thick cv2.line segments along a centre line. OpenCV draws
  thick lines with round ends, so the joints close by themselves and fillPoly is not needed.
- The grass is painted by filling the image with NumPy. The walls, the lines, the trees and
  the traffic light are drawn with line, circle and rectangle (Labs 4 and 5).
- One-channel masks are drawn apart with the same lines, like the masks of Lab 1:
    road_mask: 255 on the road. Around the road there are walls.
    drivable:  255 where the centre of a car can be: the road minus CAR_MARGIN on each
               side, so the body of the car never goes into a wall.
    zones:     one number per zone (finish line, checkpoints, STOP area and line, traffic
               light line), so the game knows where a car is by reading a single pixel.
- Texts go through put_text of src/camera.py.

The centre line comes from a few hand-placed points whose corners are cut (Chaikin's
algorithm, a few lines of NumPy) to get round curves. This smoothing is not in the labs
(it is geometry, not computer vision): it is cited in the report.

Positions along the track are fractions of the lap, from 0 to 1, measured from the finish
line in the driving direction (anticlockwise on screen).

The track leaves free the two corners where main.py draws the FPS (top left) and the small
camera view (top right).

Demo, from the repository root:
    python -m src.game.demo
"""
import cv2
import numpy as np

from src.camera import put_text

SIZE = (1280, 720)                   # (width, height) of the image that Game.draw() returns
FPS_CORNER = (0, 0, 250, 60)         # (x0, y0, x1, y1) kept free: the FPS are written here
CAMERA_CORNER = (960, 0, 1280, 180)  # kept free: the small camera view is pasted here

ROAD_WIDTH = 110       # px: width of the thick lines that make the road
WALL_WIDTH = 12        # px: red and white walls just outside both edges of the road
CAR_MARGIN = 12        # px: the centre of the car stays this far from the walls (the car is 22 px wide)
LINE_ZONE_WIDTH = 24   # px: zone lines are wide, so a fast car cannot jump over one in a single frame

# Values of the zones mask (0 = the rest of the track)
ZONE_NONE = 0
ZONE_FINISH = 1
ZONE_CHECKPOINTS = (2, 3, 4, 5)   # C1..C4, in driving order
ZONE_STOP_AREA = 6                # the car has to stop completely here...
ZONE_STOP_LINE = 7                # ...before crossing this line
ZONE_LIGHT_LINE = 8               # crossing this line with the light red is a penalty

LIGHT_COLORS = ("green", "yellow", "red")

# Colours (BGR)
GRASS = ((58, 138, 58), (66, 153, 66))   # two greens, in horizontal stripes
GRASS_STRIPE = 40                        # px
ASPHALT = (85, 85, 85)
WALL_COLORS = ((0, 0, 210), (235, 235, 235))
WALL_BLOCK = 24                          # px of wall of each colour
LANE_COLOR = (220, 220, 220)
LANE_DASH = (14, 30)                     # (dash length, distance between dashes) of the centre line, px
WHITE = (255, 255, 255)
TREE_RADIUS = 24
# Traffic light lamps: (on, off, glow, shine)
LAMPS = {
    "red": ((0, 0, 255), (0, 0, 90), (80, 80, 255), (200, 200, 255)),
    "yellow": ((0, 200, 255), (0, 80, 100), (80, 220, 255), (200, 250, 255)),
    "green": ((0, 220, 0), (0, 80, 0), (80, 255, 80), (200, 255, 200)),
}

START_GAP = 30       # px: the car starts this far behind the finish line

# The single-player track
LAYOUT = {
    # Hand-placed points of the centre line (they are smoothed afterwards)
    "points": [(300, 620), (900, 620), (1130, 575), (1170, 400), (1110, 290), (930, 280), (790, 360),
               (640, 240), (520, 150), (330, 150), (170, 280), (150, 470), (200, 590)],
    "finish_point": (745, 620),            # the lap starts at the point of the centre line closest to this
    "stop_area": (0.164, 0.199),           # STOP area, on the way up on the right; the STOP line is at its end
    "light": 0.557,                        # traffic light, on the straight at the top
    "checkpoints": (0.275, 0.415, 0.615, 0.784),
    # Trees that would touch the road, the walls, a free corner or the border are left out
    "trees": [(560, 430), (640, 470), (470, 455), (330, 430), (315, 330), (400, 510), (900, 420), (980, 470),
              (60, 650), (1240, 660), (1240, 280), (40, 120), (880, 85), (780, 130), (1000, 380), (1060, 200)],
}


def chaikin(points, iterations=4):
    """Smooths a closed polygon by cutting its corners (Chaikin's algorithm).

    Every side is replaced by two points at 1/4 and 3/4 of it; repeating this a few
    times turns the corners into round curves.

    Args:
        points (list): (x, y) points of a closed polygon.
        iterations (int): how many times the corners are cut.

    Returns:
        np.array: N x 2 array with the smoothed closed curve.
    """
    p = np.array(points, float)
    for _ in range(iterations):
        nxt = np.roll(p, -1, axis=0)
        smoothed = np.zeros((2 * len(p), 2))
        smoothed[0::2] = 0.75 * p + 0.25 * nxt
        smoothed[1::2] = 0.25 * p + 0.75 * nxt
        p = smoothed
    return p


def _px(point):
    """(x, y) as a tuple of ints, as OpenCV's drawing functions want."""
    return int(round(point[0])), int(round(point[1]))


def _normal(u):
    """Unit vector perpendicular to the unit vector u."""
    return np.array([-u[1], u[0]])


class Track:
    """The race track: background image, road mask, zones mask and the geometry of the centre line."""

    def __init__(self, layout=LAYOUT, size=SIZE):
        """
        Args:
            layout (dict): points of the centre line and positions of everything along the lap.
            size (tuple): (width, height) of the image.
        """
        self.layout = layout
        self.size = size

        # Centre line, starting at the finish line
        centre = chaikin(layout["points"])
        first = int(np.argmin(np.linalg.norm(centre - np.array(layout["finish_point"], float), axis=1)))
        self.centre = np.roll(centre, -first, axis=0)
        self.seg_vec = np.roll(self.centre, -1, axis=0) - self.centre   # side i goes from point i to point i + 1
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)
        self.cum = np.hstack(([0.0], np.cumsum(self.seg_len)))     # distance from the finish to each point
        self.length = float(self.cum[-1])

        self._place_traffic_light()
        self.road_mask = self._draw_road_mask()
        self.drivable = self._draw_road_mask(margin=CAR_MARGIN)
        self.zones = self._draw_zones()
        self.background = self._draw_background()

    # ---- Geometry of the centre line ----

    def point_at(self, s):
        """Point of the centre line at fraction s of the lap, and the unit vector of the driving direction there."""
        d = (s % 1.0) * self.length
        i = min(int(np.searchsorted(self.cum, d, side="right")) - 1, len(self.centre) - 1)
        u = self.seg_vec[i] / self.seg_len[i]
        return self.centre[i] + u * (d - self.cum[i]), u

    def nearest(self, x, y):
        """Closest point of the centre line to (x, y). Used for the walls and to know how far along the lap a car is.

        Returns:
            tuple: (point, tangent, distance, s): the closest point, the driving direction there,
            how far (x, y) is from the centre line (the edge of the road is at ROAD_WIDTH / 2)
            and the fraction of the lap (0 at the finish line).
        """
        p = np.array([x, y], float)
        t = np.clip(((p - self.centre) * self.seg_vec).sum(axis=1) / self.seg_len ** 2, 0.0, 1.0)
        closest = self.centre + self.seg_vec * t[:, None]
        dist = np.linalg.norm(closest - p, axis=1)
        i = int(np.argmin(dist))
        s = (self.cum[i] + t[i] * self.seg_len[i]) / self.length
        return closest[i], self.seg_vec[i] / self.seg_len[i], float(dist[i]), float(s)

    def on_road(self, x, y):
        """True if (x, y) is road, read from the road mask (like the masks of Lab 1). Outside the image is False."""
        xi, yi = _px((x, y))
        h, w = self.road_mask.shape
        return 0 <= xi < w and 0 <= yi < h and self.road_mask[yi, xi] > 0

    def can_drive(self, x, y):
        """True if the centre of a car can be at (x, y) without its body going into a wall (drivable mask)."""
        xi, yi = _px((x, y))
        h, w = self.drivable.shape
        return 0 <= xi < w and 0 <= yi < h and self.drivable[yi, xi] > 0

    def zone_at(self, x, y):
        """Zone at (x, y): one of the ZONE_* values, read from the zones mask."""
        xi, yi = _px((x, y))
        h, w = self.zones.shape
        if not (0 <= xi < w and 0 <= yi < h):
            return ZONE_NONE
        return int(self.zones[yi, xi])

    def start_pose(self, side=0.0):
        """Where a car starts: just behind the finish line, facing the driving direction.

        Args:
            side (float): px to the right of the middle of the road, as seen by the driver
                (negative: to the left). With two players the cars start side by side.
        Returns:
            tuple: (x, y, angle), with the angle in degrees, anticlockwise from the x axis as
            seen on screen (the convention of cv2.getRotationMatrix2D).
        """
        p, u = self.point_at(-START_GAP / self.length)
        p = p + _normal(u) * side   # the normal points to the right of the driving direction on screen
        angle = float(np.arctan2(-u[1], u[0]) * 180 / np.pi)
        return float(p[0]), float(p[1]), angle

    # ---- Drawing ----

    def draw(self, light_color="green"):
        """Image of the track for this frame: a copy of the background with the traffic light on top.

        Args:
            light_color (str): "green", "yellow" or "red".

        Returns:
            np.array: BGR image of SIZE.
        """
        image = self.background.copy()
        self.draw_traffic_light(image, light_color)
        return image

    def draw_traffic_light(self, image, color):
        """Draws the parts of the traffic light that change: the lit lamp and the colour of its line."""
        cv2.line(image, self.light_line[0], self.light_line[1], LAMPS[color][0], 12)
        for k, name in enumerate(("red", "yellow", "green")):
            q = _px(self.light_box + (-40 + 40 * k, 0))
            on, off, glow, shine = LAMPS[name]
            if name == color:
                cv2.circle(image, q, 21, glow, -1)
                cv2.circle(image, q, 15, on, -1)
                cv2.circle(image, q, 6, shine, -1)
            else:
                cv2.circle(image, q, 15, off, -1)   # an unlit lamp keeps a dark version of its colour

    def _chain(self, image, points, color, thickness, closed=True):
        """Joins the points with thick cv2.line segments (the way the road is drawn)."""
        n = len(points) if closed else len(points) - 1
        for i in range(n):
            cv2.line(image, _px(points[i]), _px(points[(i + 1) % len(points)]), color, thickness)

    def _across(self, s, extra=0.0):
        """Ends of a line across the road at fraction s of the lap (extra px beyond each edge)."""
        p, u = self.point_at(s)
        half = ROAD_WIDTH / 2 + extra
        return _px(p + _normal(u) * half), _px(p - _normal(u) * half)

    def _place_traffic_light(self):
        """Where the traffic light goes: its line across the road and the light itself, inside the loop."""
        p, u = self.point_at(self.layout["light"])
        n = _normal(u)
        side = n if np.dot(n, self.centre.mean(axis=0) - p) > 0 else -n   # towards the inside of the loop
        self.light_line = self._across(self.layout["light"])
        self.light_post = (_px(p + side * (ROAD_WIDTH / 2 + WALL_WIDTH)), _px(p + side * (ROAD_WIDTH / 2 + 50)))
        self.light_box = p + side * (ROAD_WIDTH / 2 + 70)

    def _draw_road_mask(self, margin=0):
        """The road drawn with the same thick lines on a one-channel image (margin px narrower on each side)."""
        width, height = self.size
        mask = np.zeros((height, width), np.uint8)
        self._chain(mask, self.centre, 255, ROAD_WIDTH - 2 * margin)
        return mask

    def _draw_zones(self):
        width, height = self.size
        zones = np.zeros((height, width), np.uint8)
        # STOP area: filled with short lines across the road, 3 px apart and 10 px thick, so there
        # are no gaps even on the outer side of a curve. Thick lines along the road would not work
        # here, because their round ends would also paint some road after the STOP line.
        a, b = self.layout["stop_area"]
        step = 3.0 / self.length
        for s in np.arange(a, b, step):
            p1, p2 = self._across(s, extra=3)   # a bit past the edges: the road mask cuts it afterwards
            cv2.line(zones, p1, p2, ZONE_STOP_AREA, 10)
        # Lines across the road, drawn after the area so the STOP line keeps its own value
        lines = [(0.0, ZONE_FINISH), (b, ZONE_STOP_LINE), (self.layout["light"], ZONE_LIGHT_LINE)]
        lines += list(zip(self.layout["checkpoints"], ZONE_CHECKPOINTS))
        for s, zone in lines:
            p1, p2 = self._across(s, extra=3)
            cv2.line(zones, p1, p2, zone, LINE_ZONE_WIDTH)
        # Zones only exist on the road: keep the pixels inside the road mask, as in Lab 1
        return cv2.bitwise_and(zones, zones, mask=self.road_mask)

    def _keep_tree(self, x, y):
        """A tree is drawn only if it does not touch the road, the walls, a free corner or the border."""
        r = TREE_RADIUS + 6   # with its shadow
        width, height = self.size
        if x - r < 0 or y - r < 0 or x + r >= width or y + r >= height:
            return False
        for (x0, y0, x1, y1) in (FPS_CORNER, CAMERA_CORNER):
            if x + r > x0 and x - r < x1 and y + r > y0 and y - r < y1:
                return False
        return self.nearest(x, y)[2] > ROAD_WIDTH / 2 + WALL_WIDTH + r + 4

    def _draw_background(self):
        width, height = self.size
        image = np.zeros((height, width, 3), np.uint8)

        # Grass in stripes
        for k, y in enumerate(range(0, height, GRASS_STRIPE)):
            image[y:y + GRASS_STRIPE] = GRASS[k % 2]

        # Trees: shadow, crown and a lighter spot
        for (x, y) in self.layout["trees"]:
            if self._keep_tree(x, y):
                cv2.circle(image, (x + 5, y + 6), TREE_RADIUS, (35, 80, 35), -1)
                cv2.circle(image, (x, y), TREE_RADIUS, (30, 110, 30), -1)
                cv2.circle(image, (x - 7, y - 7), 10, (60, 150, 60), -1)

        # Road
        self._chain(image, self.centre, ASPHALT, ROAD_WIDTH)

        # Walls: red and white blocks just outside both edges
        step = 6.0
        offset = ROAD_WIDTH / 2 + WALL_WIDTH / 2
        prev = None
        for k in range(int(self.length // step) + 1):
            p, u = self.point_at(k * step / self.length)
            ends = (p + _normal(u) * offset, p - _normal(u) * offset)
            if prev is not None:
                color = WALL_COLORS[int(k * step // WALL_BLOCK) % 2]
                cv2.line(image, _px(prev[0]), _px(ends[0]), color, WALL_WIDTH)
                cv2.line(image, _px(prev[1]), _px(ends[1]), color, WALL_WIDTH)
            prev = ends

        # Dashed centre line
        dash, period = LANE_DASH
        for k in range(int(self.length // period)):
            p1, _ = self.point_at(k * period / self.length)
            p2, _ = self.point_at((k * period + dash) / self.length)
            cv2.line(image, _px(p1), _px(p2), LANE_COLOR, 3)

        # Finish line: two rows of black and white squares across the road
        p, u = self.point_at(0.0)
        n = _normal(u)
        square = ROAD_WIDTH / 10
        for k in range(-5, 5):
            for j in range(2):
                corner = p + n * (k * square) + u * ((j - 1) * square)
                color = WHITE if (k + j) % 2 == 0 else (0, 0, 0)
                cv2.rectangle(image, _px(corner), _px(corner + n * square + u * square), color, -1)

        # STOP: thin mark where the area starts, thick white line at its end and the word STOP
        a, b = self.layout["stop_area"]
        p1, p2 = self._across(a)
        cv2.line(image, p1, p2, (190, 190, 190), 2)
        p1, p2 = self._across(b)
        cv2.line(image, p1, p2, WHITE, 8)
        middle, _ = self.point_at((a + b) / 2)
        put_text(image, "STOP", _px(middle + (-40, 10)), WHITE, 0.8)

        # Traffic light, the parts that never change: white edges of its line, post, yellow plate and housing
        cv2.line(image, self.light_line[0], self.light_line[1], WHITE, 20)
        cv2.line(image, self.light_post[0], self.light_post[1], (70, 70, 70), 6)
        cv2.rectangle(image, _px(self.light_box - (72, 26)), _px(self.light_box + (72, 26)), (0, 210, 255), -1)
        cv2.rectangle(image, _px(self.light_box - (62, 19)), _px(self.light_box + (62, 19)), (25, 25, 25), -1)
        return image
