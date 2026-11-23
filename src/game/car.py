"""A car of the game: how it moves and how it is drawn.

How it moves (simple physics with the real dt between frames, so the speed does not depend on the FPS):
- throttle > 0 accelerates up to MAX_SPEED, throttle = 0 keeps the speed and throttle < 0 brakes
  down to 0. There is no reverse gear (the wheel has no way to ask for it): braking stops the car
  and it stays stopped.
- steering turns the car, also when it is stopped, so it can never get stuck against a wall.
- If visible is False (the tracker has lost the wheel), the car brakes by itself.
- Walls: the centre of the car can only be where the drivable mask of the track is 255 (the road
  minus a margin, so the body never goes into a wall). A move that would leave it slides along
  the wall instead (only the part of the move along the road is kept). When the car hits a wall
  it keeps only the part of its speed that goes along the wall (a grazing touch costs little, a
  head-on crash stops it), and while it keeps pushing against the wall it cannot go faster than
  WALL_SPEED. The car does not steer by itself: without turning away from the wall it ends up
  stopped against it (in a curve, for example), and turning frees it.
- The move of each frame is done in small steps (at most STEP_PX), so even at low FPS a car can
  never jump over a wall or over a zone line, and the game gets every position it went through.

How it is drawn, as in Lab 1: the car is drawn once facing right with rectangle, rotated with
getRotationMatrix2D + warpAffine and pasted on the track with a mask, using bitwise_and and
bitwise_or. Its plate is written above it with put_text, in the colour of the car, by a method
of its own (draw_plate), so the game can write the plates after drawing all the cars.
"""
import cv2
import numpy as np

from src.camera import put_text

MAX_SPEED = 260.0      # px/s
ACCELERATION = 180.0   # px/s^2 with full throttle
BRAKING = 340.0        # px/s^2 with full brake: from full speed to 0 in about 0.8 s
TURN_RATE = 130.0      # degrees/s with full steering, also when stopped
WALL_SPEED = 100.0     # px/s: maximum speed while the car is pushing against a wall
WALL_DRAG = 500.0      # px/s^2 at which a car pushing against a wall slows down to WALL_SPEED
STOPPED_SPEED = 5.0    # px/s: below this the car counts as stopped (for the STOP)
STEP_PX = 4.0          # px: the move of a frame is done in steps of at most this
MAX_DT = 0.1           # s: a longer frame (for example, while the window is dragged) counts as this

CAR_COLOR = (0, 0, 220)   # BGR: red
SPRITE = 60   # px: side of the square image where the car is drawn, big enough to rotate it inside


def make_sprite(color):
    """Draws the car once, facing right (the front is on the right), and its mask.

    Args:
        color (tuple): BGR colour of the body.

    Returns:
        tuple: (sprite, mask): SPRITE x SPRITE BGR image of the car and one-channel mask (255 = car).
    """
    sprite = np.zeros((SPRITE, SPRITE, 3), np.uint8)
    mask = np.zeros((SPRITE, SPRITE), np.uint8)
    wheels = [(13, 15, 21, 19), (39, 15, 47, 19), (13, 41, 21, 45), (39, 41, 47, 45)]
    for (x0, y0, x1, y1) in wheels:
        cv2.rectangle(sprite, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    cv2.rectangle(sprite, (10, 19), (50, 41), color, -1)          # body: 40 x 22 px
    cv2.rectangle(mask, (10, 19), (50, 41), 255, -1)
    cv2.rectangle(sprite, (34, 22), (42, 38), (60, 40, 30), -1)   # windscreen, at the front
    return sprite, mask


class Car:
    """The car: its position, direction and speed, the walls and its drawing."""

    def __init__(self, track, plate, color=CAR_COLOR, side=0.0):
        """
        Args:
            track (Track): the track it drives on.
            plate (str): license plate, written above the car.
            color (tuple): BGR colour of the body.
            side (float): where it starts across the road, in px to the right of the middle (see Track.start_pose).
        """
        self.track = track
        self.plate = plate
        self.color = color
        self.side = side
        self.sprite, self.sprite_mask = make_sprite(color)
        self.reset()

    def reset(self):
        """Puts the car back on its place on the start, stopped."""
        self.x, self.y, self.angle = self.track.start_pose(self.side)
        self.speed = 0.0
        self.touching_wall = False

    @property
    def stopped(self):
        """True if the car is (almost) not moving."""
        return self.speed < STOPPED_SPEED

    def heading(self):
        """Unit vector of the direction the car is facing (the y axis of the image grows downwards)."""
        a = np.deg2rad(self.angle)
        return np.array([np.cos(a), -np.sin(a)])

    def update(self, controls, dt):
        """Moves the car one frame.

        Args:
            controls (Controls): steering, throttle and visible for this frame (any value from -1 to +1).
            dt (float): real time since the previous frame, in seconds.

        Returns:
            list: (x, y) positions the car went through in this frame; the last one is where it is now.
        """
        dt = min(max(dt, 0.0), MAX_DT)
        steering = float(np.clip(controls.steering, -1.0, 1.0))
        throttle = float(np.clip(controls.throttle, -1.0, 1.0))
        if not controls.visible:   # the wheel is lost: brake with the wheels straight
            steering, throttle = 0.0, -1.0

        # Steering to the right (+) turns clockwise on screen, so the angle goes down
        self.angle = (self.angle - steering * TURN_RATE * dt) % 360.0

        if throttle > 0:
            self.speed = min(MAX_SPEED, self.speed + throttle * ACCELERATION * dt)
        else:
            self.speed = max(0.0, self.speed + throttle * BRAKING * dt)   # never below 0: no reverse gear

        # Move in small steps; a step that would go into a wall slides along it instead
        distance = self.speed * dt
        steps = max(1, int(np.ceil(distance / STEP_PX)))
        step = self.heading() * (distance / steps)
        was_touching = self.touching_wall
        self.touching_wall = False
        along_road = None
        start = (self.x, self.y)
        path = []
        for _ in range(steps):
            x, y = self.x + step[0], self.y + step[1]
            if not self.track.can_drive(x, y):
                self.touching_wall = True
                closest, along_road, _, _ = self.track.nearest(self.x, self.y)
                slide = along_road * np.dot(step, along_road)   # keep only the part of the step along the road
                inwards = closest - (self.x, self.y)             # towards the middle of the road
                inwards = inwards / max(float(np.linalg.norm(inwards)), 1e-6)
                # On a curve, going straight along the wall would leave the road: push 1-2 px inwards
                for push in (0.0, 1.0, 2.0):
                    x, y = self.x + slide[0] + inwards[0] * push, self.y + slide[1] + inwards[1] * push
                    if self.track.can_drive(x, y):
                        break
                else:
                    x, y = self.x, self.y   # blocked: it stays, and it can still turn away from the wall
            self.x, self.y = x, y
            path.append((x, y))
        if self.speed > 0 and (self.x, self.y) == start:
            self.speed = 0.0   # completely blocked by a wall this frame: it really is stopped

        if self.touching_wall:
            if not was_touching:
                # The crash: only the speed along the wall is kept (cosine of the angle with the wall)
                self.speed *= abs(float(np.dot(self.heading(), along_road)))
            if self.speed > WALL_SPEED:   # pushing against the wall: slow down to WALL_SPEED
                self.speed = max(WALL_SPEED, self.speed - WALL_DRAG * dt)
        return path

    def draw(self, image):
        """Draws the car on the image (rotated as in Lab 1, pasted with a mask). Its plate is written by draw_plate."""
        M = cv2.getRotationMatrix2D((SPRITE / 2, SPRITE / 2), self.angle, 1.0)
        # INTER_NEAREST keeps the mask with only 0 and 255, so the edges of the car stay clean
        sprite = cv2.warpAffine(self.sprite, M, (SPRITE, SPRITE), flags=cv2.INTER_NEAREST)
        mask = cv2.warpAffine(self.sprite_mask, M, (SPRITE, SPRITE), flags=cv2.INTER_NEAREST)

        # Part of the image under the sprite, cut at the borders of the image
        h, w = image.shape[:2]
        x0, y0 = int(round(self.x)) - SPRITE // 2, int(round(self.y)) - SPRITE // 2
        ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + SPRITE, w), min(y0 + SPRITE, h)
        if ix0 >= ix1 or iy0 >= iy1:
            return
        sprite = sprite[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        mask = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        roi = image[iy0:iy1, ix0:ix1]
        back = cv2.bitwise_and(roi, roi, mask=255 - mask)   # the track where the car is not
        front = cv2.bitwise_and(sprite, sprite, mask=mask)  # the car
        roi[:] = cv2.bitwise_or(back, front)

    def draw_plate(self, image):
        """Writes the plate above the car, in the colour of the car (the black outline of put_text keeps it readable)."""
        put_text(image, self.plate, (int(round(self.x)) - 6 * len(self.plate), int(round(self.y)) - 32),
                 self.color, 0.5)
