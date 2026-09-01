from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# GLOBALS
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "cokeVideo.mp4"
RESULTS_DIR = BASE_DIR / "output"
FRAMES_DIR = RESULTS_DIR / "frames"

MODEL_NAME = "yolov8n.pt"
TARGET_CLASS = "bottle"
TARGET_ID = 39
SCORE_THRESHOLD = 0.50


MAX_CORNERS = 150
QUALITY_LEVEL = 0.20
MIN_DISTANCE = 8
BOX_SHRINK = 0.90

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
FB_THRESHOLD = 2.0
MAX_ERROR = 30.0

MIN_TRACKS = 12
MIN_RATIO = 0.35
MAX_JUMP = 90.0
COOLDOWN = 5

PROC_WIDTH = 1280
SAVE_FRAMES = [0, 300, 775, 810, 900, 1200, 1500, 1814]

TRACKING = "TRACKING"
REDETECTING = "RE-DETECTING"
LOST = "SEARCH/LOST"

STATE_COLOURS = {TRACKING: (0, 220, 0), REDETECTING: (0, 200, 255), LOST: (0, 0, 255)}



# week 3 lab
def inside_frame(points, frame_shape):
    height, width = frame_shape[:2]
    x, y = points[:, 0], points[:, 1]
    return np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)


# week 3 lab
def make_roi_mask(shape, roi):
    x, y, w, h = map(int, roi)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask


# week 3 lab
def detect(gray, roi, max_corners=100, quality=0.1, min_distance=5):
    return cv.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality,
                                  minDistance=min_distance, mask=make_roi_mask(gray.shape, roi))


# week 3 lab
def motionLabel(dx, threshold=2):
    if dx > threshold:
        return "MOVING RIGHT"
    elif dx < -threshold:
        return "MOVING LEFT"
    else:
        return "STATIONARY"


# week 5 lab
def boxCentre(box):
    x1, y1, x2, y2 = box

    return (x1 + x2) / 2, (y1 + y2) / 2


# week 5 lab
def bestDetection(detections, target_class):
    matches = [detection for detection in detections if detection[1] == target_class]

    if not matches:
        return None

    return max(matches, key=lambda detection: detection[2])


# week 5 lab
def drawBox(frame, box, label, colour=(0, 255, 0), label_below=False):
    x1, y1, x2, y2 = map(int, box)
    height = frame.shape[0]

    cv.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

    text_y = min(y2 + 32, height - 10) if label_below else max(y1 - 10, 25)

    cv.putText(frame, label, (x1, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    return frame


# week 3 lab
def drawKeypoints(frame, points):
    result = frame.copy()

    if points is not None:
        for u, v in points.reshape(-1, 2):
            cv.circle(result, (round(u), round(v)), 4, (0, 0, 255), -1)

    return result


# week 5 lab, show flag added to not block figure
def showFrames(frames, titles, save_as=None, columns=3, show=True):
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)

    for axis, frame, title in zip(axes.flat, frames, titles):
        axis.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        axis.set_title(title, fontsize=10)
        axis.axis('off')

    for axis in axes.flat[len(frames):]:
        axis.axis('off')

    plt.tight_layout()

    if save_as is not None:
        RESULTS_DIR.mkdir(exist_ok=True)
        plt.savefig(RESULTS_DIR / save_as, dpi=120)
        print(f"Saved {RESULTS_DIR / save_as}")

    if show:
        plt.show()
    else:
        plt.close(figure)


# part 1 functions -- OPEN AND INSPECT VIDEO

def openVideo(video_path):
    cap = cv.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot read video: {video_path}")

    return cap


def videoInfo(cap):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30

    return width, height, frames, fps


# part 2 functions -- DETECTION

def loadDetector():
    return YOLO(MODEL_NAME)


# discard all detections except the bottle
def detectTarget(model, frame):
    prediction = model.predict(frame, classes=[TARGET_ID], conf=SCORE_THRESHOLD, verbose=False)[0]

    # from gpu -> cpu, then np list
    detections = [
        (box.tolist(), TARGET_CLASS, float(score))
        for box, score in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy())
    ]

    best = bestDetection(detections, TARGET_CLASS)

    if best is None:
        return None, 0.0

    box, _, score = best
    x1, y1, x2, y2 = map(int, box)

    return (x1, y1, x2, y2), score


# shrinks box bc of the carpets interference
def boxToRoi(box, shrink=BOX_SHRINK):
    x1, y1, x2, y2 = box
    cx, cy = boxCentre(box)
    w, h = (x2 - x1) * shrink, (y2 - y1) * shrink

    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


# calls detect from week 3 with the shrunken box
def featuresInBox(gray, box):
    points = detect(gray, boxToRoi(box), MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE)

    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)

    return points


# part 3 functions -- TRACKING

def trackPoints(old_gray, new_gray, p0, frame_shape):
    if len(p0) == 0:
        return None, None, (0.0, 0.0)

    p1, status, error = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **LK_PARAMS)

    if p1 is None:
        return None, None, (0.0, 0.0)

    # backwards track to check that the points are still valid post occlusion
    p0_back, status_back, _ = cv.calcOpticalFlowPyrLK(new_gray, old_gray, p1, None, **LK_PARAMS)
    fb_error = np.linalg.norm(p0.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)

    # same validity test as labs with the backward check added
    valid = ((status.ravel() == 1) & (status_back.ravel() == 1)
             & (error.ravel() < MAX_ERROR) & (fb_error < FB_THRESHOLD)
             & inside_frame(p1.reshape(-1, 2), frame_shape))

    good_new = p1[valid].reshape(-1, 2)
    good_old = p0[valid].reshape(-1, 2)

    if len(good_new) == 0:
        return None, None, (0.0, 0.0)

    # median so that carpet points dont drag 
    dx, dy = np.median(good_new - good_old, axis=0)

    return good_new, good_old, (float(dx), float(dy))


# distance of points from median centre
def pointSpread(points):
    centre = np.median(points, axis=0)
    return float(np.median(np.linalg.norm(points - centre, axis=1)))


# for the distance, takes ratio of relative spread to determine distance from cam, tolerance was essential picked at random
def scaleLabel(scale, tolerance=0.15):
    if scale > 1 + tolerance:
        return "CLOSER"
    elif scale < 1 - tolerance:
        return "FURTHER"
    else:
        return "SAME SIZE"


# draw tracks, similar to what we did in the labs
def drawTracks(frame, trails, good_new, good_old):
    for new, old in zip(good_new, good_old):
        a, b = np.round(new).astype(int)
        c, d = np.round(old).astype(int)

        cv.line(trails, (c, d), (a, b), (0, 180, 0), 2)

    frame = cv.add(frame, trails)

    return drawKeypoints(frame, good_new)


# same
def drawCentre(frame, centre):
    cx, cy = int(centre[0]), int(centre[1])
    cv.drawMarker(frame, (cx, cy), (0, 0, 255), cv.MARKER_CROSS, 24, 3)

    return frame


# part 4 functions -- VALIDATION

# number of valid tracks, ratio of valid tracks to start count, and jump in median position are all used to determine if the tracking is reliable
def isReliable(n_tracks, start_count, displacement):
    if n_tracks < MIN_TRACKS:
        return False, f"only {n_tracks} valid tracks"

    ratio = n_tracks / start_count if start_count else 0.0

    if ratio < MIN_RATIO:
        return False, f"inlier ratio {ratio:.2f}"

    jump = max(abs(displacement[0]), abs(displacement[1]))

    if jump > MAX_JUMP:
        return False, f"median jump {jump:.0f}px"

    return True, "ok"


# part 5 functions -- ROBOT

# essentially a lab function, however with the safety check to stop movement with no target tracked
def robotAction(state, centre, width):

    # safety check
    if state != TRACKING or centre is None:
        return "SEARCH"

    cx = centre[0]

    if cx < width / 3:
        return "TURN LEFT"
    elif cx > 2 * width / 3:
        return "TURN RIGHT"
    else:
        return "MOVE FORWARD"


# obvious
def drawThirds(frame):
    height, width = frame.shape[:2]

    for third in (width // 3, 2 * width // 3):
        cv.line(frame, (third, 0), (third, height), (90, 90, 90), 1)

    return frame


# overlay similar to old lab work with extensions specific to this project
def drawOverlay(frame, state, reason, frame_idx, total, n_tracks, centre, displacement, scale, action, attempts, successes):
    height, width = frame.shape[:2]
    colour = STATE_COLOURS[state]
    centre_text = f"({centre[0]:.0f}, {centre[1]:.0f})" if centre is not None else "n/a"
    scale_text = f"x{scale:.2f} {scaleLabel(scale)}" if scale else "n/a"

    lines = [
        f"frame {frame_idx}/{total}   tracks = {n_tracks}   centre = {centre_text}",
        f"median dx, dy = ({displacement[0]:+.1f}, {displacement[1]:+.1f}) px   {motionLabel(displacement[0])}",
        f"scale vs initial = {scale_text}   re-detections = {successes}/{attempts} successful",
    ]

    cv.rectangle(frame, (0, 0), (width, 108), (0, 0, 0), -1)
    cv.putText(frame, state, (12, 34), cv.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    for i, line in enumerate(lines):
        cv.putText(frame, line, (12, 60 + i * 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    if state != TRACKING and reason not in ("ok", "startup"):
        cv.putText(frame, f"reason: {reason}", (width - 370, 34),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    cv.rectangle(frame, (0, height - 52), (330, height), (0, 0, 0), -1)
    cv.putText(frame, action, (14, height - 14), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0) if action != "SEARCH" else (0, 0, 255), 3)

    return frame


# part 6 functions -- EVALUATION

# graphs valid track over time [AI ASSISTED FUNCTION]
def plotValidTracks(track_counts, states, save_as="valid_tracks_plot.png"):
    frames = list(range(len(track_counts)))

    figure, axis = plt.subplots(figsize=(11, 4.2))

    axis.plot(frames, track_counts, linewidth=1.1, color="tab:blue", label="valid tracked points")
    axis.axhline(MIN_TRACKS, linestyle="--", linewidth=1, color="tab:red",
                 label=f"reliability threshold ({MIN_TRACKS})")

    # shade every frame the system was not tracking
    for i, state in enumerate(states):
        if state != TRACKING:
            axis.axvspan(i - 0.5, i + 0.5, color="#ffb3b3", alpha=0.35, linewidth=0)

    axis.set_xlabel("Frame number")
    axis.set_ylabel("Valid tracked feature points")
    axis.set_title("Valid tracked feature points per frame (shaded = re-detecting or lost)")
    axis.set_xlim(0, len(frames) - 1)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / save_as, dpi=150)
    plt.close(figure)

    print(f"Saved {RESULTS_DIR / save_as}")


# manual csv save
def saveMetrics(rows, save_as="metrics.csv"):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / save_as

    with open(path, "w") as file:
        file.write("frame,valid_tracks,centre_x,centre_y,median_dx,median_dy,scale,state,action,score\n")

        for row in rows:
            file.write(",".join(str(value) for value in row) + "\n")

    print(f"Saved {path}")


# obvious [AI ASSISTED FUNCTION]
def printSummary(states, actions, track_counts, attempts, successes):
    total = len(states)

    print("\n")
    print("RUN SUMMARY")
    print("\n")
    print(f"frames processed        : {total}")
    print(f"re-detection attempts   : {attempts}")
    print(f"successful recoveries   : {successes}")

    if attempts:
        print(f"recovery rate           : {100 * successes / attempts:.1f}%")

    print("\nframes in each state:")

    for state in (TRACKING, REDETECTING, LOST):
        count = states.count(state)
        print(f"  {state:15s} {count:5d}  ({100 * count / total:5.1f}%)")

    print("\nframes for each action:")

    for action in ("MOVE FORWARD", "TURN LEFT", "TURN RIGHT", "SEARCH"):
        count = actions.count(action)
        print(f"  {action:15s} {count:5d}  ({100 * count / total:5.1f}%)")

    tracked = [n for n, state in zip(track_counts, states) if state == TRACKING]

    if tracked:
        print(f"\nvalid tracks while tracking: min = {min(tracked)}, "
              f"max = {max(tracked)}, mean = {sum(tracked) / len(tracked):.1f}")

    print("\n")
