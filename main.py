# my main file for this project, runs the pipeline and saves the results

# libs
import sys
import cv2 as cv
import numpy as np


# functions from utils
from utils import (
    BASE_DIR, VIDEO_PATH, RESULTS_DIR, FRAMES_DIR,
    TARGET_CLASS, SCORE_THRESHOLD, MIN_TRACKS, COOLDOWN, PROC_WIDTH, SAVE_FRAMES,
    TRACKING, REDETECTING, LOST,
    openVideo, videoInfo, loadDetector, detectTarget, featuresInBox, drawBox,
    drawKeypoints, trackPoints, drawTracks, drawCentre, pointSpread, isReliable, robotAction,
    drawThirds, drawOverlay, showFrames, plotValidTracks, saveMetrics, printSummary,
)


# GLOBALS
OUTPUT_VIDEO = RESULTS_DIR / "annotated_output.mp4"
SHOW_WINDOW = "--no-display" not in sys.argv

# condition frames [AI ASSISTED]
CONDITION_FRAMES = {
    0: "normal, detected at the start, centre third",
    300: "normal tracking, right third",
    1200: "normal tracking, left third",
    775: "challenge 1, hand starting to occlude",
    810: "challenge 1, fully occluded, tracking lost",
    900: "recovered by automatic re-detection",
    1500: "challenge 2, scale down, furthest away",
    1814: "challenge 2, scale up, closest to camera",
}


def main():
    # setup
    RESULTS_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)

    cap = openVideo(VIDEO_PATH)
    src_width, src_height, total, fps = videoInfo(cap)

    # part 1 -- basic properties of the recorded video
    print("\n")
    print("PART 1 -- VIDEO PROPERTIES")
    print("\n")
    print(f"file             : {VIDEO_PATH.relative_to(BASE_DIR)}")
    print(f"frame width      : {src_width}")
    print(f"frame height     : {src_height}")
    print(f"number of frames : {total}")
    print(f"fps              : {fps:.2f}")
    print(f"duration         : {total / fps:.1f} s")
    print("\n")

    # rescales video for processing (height needed to be scaled to for aspect ratio)
    resize_scale = PROC_WIDTH / src_width
    width, height = PROC_WIDTH, int(round(src_height * resize_scale))

    model = loadDetector()
    writer = cv.VideoWriter(str(OUTPUT_VIDEO), cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # error check, like done in the labs
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open the video writer: {OUTPUT_VIDEO}")

    print(f"\ntarget class = {TARGET_CLASS}, confidence threshold = {SCORE_THRESHOLD}")
    print(f"processing at {width}x{height}\n")

    # we always start by detecting, target is never picked by hand
    state = REDETECTING
    reason = "startup"
    attempts = 0
    successes = 0
    cooldown = 0

    # initalisation
    p0 = np.empty((0, 1, 2), dtype=np.float32)
    start_count = 0
    start_spread = 0.0
    old_gray = None
    trails = None
    last_box, last_score = None, 0.0

    states, actions, track_counts, rows = [], [], [], []
    condition_images, condition_order = [], []
    frame_idx = 0

    # main loop
    while True:
        ok, source_frame = cap.read()

        if not ok:
            break

        frame = cv.resize(source_frame, (width, height))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # keep a clean copy of the raw frame for the part 1 figure
        if frame_idx in CONDITION_FRAMES:
            condition_images.append(frame.copy())
            condition_order.append(frame_idx)

        if trails is None:
            trails = np.zeros_like(frame)
        else:
            # fade out
            trails = (trails * 0.96).astype(np.uint8)

        centre = None
        displacement = (0.0, 0.0)
        scale = 0.0
        good_new, good_old = None, None
        redetected = False

        # part 3 -- tracks our points across frames
        if state == TRACKING and old_gray is not None:
            good_new, good_old, displacement = trackPoints(old_gray, gray, p0, frame.shape)

            if good_new is None:
                state = REDETECTING
                reason = "no valid tracks"
                p0 = np.empty((0, 1, 2), dtype=np.float32)
            else:
                centre = tuple(np.median(good_new, axis=0))
                p0 = good_new.reshape(-1, 1, 2)
                scale = pointSpread(good_new) / start_spread if start_spread else 0.0

                
                reliable, reason = isReliable(len(good_new), start_count, displacement)

                if not reliable:
                    state = REDETECTING

        # part 4 -- occlusion handling
        frame_state = state

        if state != TRACKING:
            if cooldown > 0:
                cooldown -= 1
                frame_state = LOST
            else:
                frame_state = REDETECTING
                attempts += 1

                box, score = detectTarget(model, frame)
                points = featuresInBox(gray, box) if box is not None else []

                if box is not None and len(points) >= MIN_TRACKS:
                    p0 = points
                    start_count = len(points)
                    start_spread = pointSpread(points.reshape(-1, 2))
                    scale = 1.0
                    centre = tuple(np.median(points.reshape(-1, 2), axis=0))
                    last_box, last_score = box, score
                    trails = np.zeros_like(frame)
                    successes += 1
                    redetected = True
                    state = TRACKING
                else:
                    p0 = np.empty((0, 1, 2), dtype=np.float32)
                    start_count = 0
                    start_spread = 0.0
                    state = LOST
                    cooldown = COOLDOWN

        # part 5 -- robot stuff
        action = robotAction(frame_state, centre, width)

        # draw everything onto the frame
        frame = drawThirds(frame)

        if frame_state == TRACKING and good_new is not None:
            frame = drawTracks(frame, trails, good_new, good_old)

        # draw the last detected box and keypoints if we are in the re detecting state
        if redetected and last_box is not None:
            frame = drawBox(frame, last_box, f"{TARGET_CLASS} {last_score:.2f}",
                            colour=(255, 0, 255), label_below=True)
            frame = drawKeypoints(frame, p0)

        if centre is not None:
            frame = drawCentre(frame, centre)

        n_tracks = len(p0)

        frame = drawOverlay(frame, frame_state, reason, frame_idx, total, n_tracks,
                            centre, displacement, scale, action, attempts, successes)

        writer.write(frame)

        states.append(frame_state)
        actions.append(action)
        track_counts.append(n_tracks)
        rows.append([
            frame_idx, n_tracks,
            f"{centre[0]:.1f}" if centre is not None else "",
            f"{centre[1]:.1f}" if centre is not None else "",
            f"{displacement[0]:.2f}", f"{displacement[1]:.2f}",
            f"{scale:.3f}" if scale else "",
            frame_state, action,
            f"{last_score:.3f}" if redetected else "",
        ])

        if frame_idx in SAVE_FRAMES:
            cv.imwrite(str(FRAMES_DIR / f"frame_{frame_idx:05d}.jpg"), frame)

        if SHOW_WINDOW:
            cv.imshow("Detection Assisted Visual Tracking", frame)

            if cv.waitKey(1) & 0xFF == 27:
                print("\nstopped early by the user")
                break

        old_gray = gray
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  frame {frame_idx}/{total}   state = {frame_state}   tracks = {n_tracks}")

    cap.release()
    writer.release()
    cv.destroyAllWindows()

    #  figure gen
    if condition_images:
        
        condition_titles = []

        for idx in condition_order:
            measured = rows[idx][6] if idx < len(rows) else ""
            suffix = f" (x{float(measured):.2f})" if measured else ""
            condition_titles.append(f"frame {idx} - {CONDITION_FRAMES[idx]}{suffix}")

        showFrames(condition_images, condition_titles, "part1_conditions.png", show=SHOW_WINDOW)

    plotValidTracks(track_counts, states)
    saveMetrics(rows)
    printSummary(states, actions, track_counts, attempts, successes)

    print(f"\nannotated video saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()