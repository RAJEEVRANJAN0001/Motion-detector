import cv2
import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()

# Load a sound file (replace with your sound file path)
sound_effect = pygame.mixer.Sound('beep-01a.wav')

# Set the threshold for motion detection
motion_threshold = 7000

cam = cv2.VideoCapture(0)

# Flag to track whether sound has been played for the current frame
sound_played = False

# Video brightness adjustment
brightness_value = 0

# Flip video feed horizontally or vertically
flip_horizontal = False
flip_vertical = False

# Initialize snapshot counter
snapshot_counter = 0

# Folder to save snapshots
output_folder = 'snapshots'
os.makedirs(output_folder, exist_ok=True)

# Flag to track whether motion detection is paused
motion_detection_paused = False

try:
    while True:
        ret, frame1 = cam.read()
        ret, frame2 = cam.read()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Reset the sound played flag for each frame
        sound_played = False

        for c in contours:
            if cv2.contourArea(c) < motion_threshold:
                continue

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(c)

            # Draw rectangle around the detected motion
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Play sound when motion is detected and sound is enabled
            if not sound_played:
                sound_effect.play()
                sound_played = True

        # Adjust video brightness
        frame1 = cv2.convertScaleAbs(frame1, alpha=1.0, beta=brightness_value)

        # Flip video feed if needed
        if flip_horizontal:
            frame1 = cv2.flip(frame1, 1)
        if flip_vertical:
            frame1 = cv2.flip(frame1, 0)

        cv2.imshow("Motion Detection", frame1)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        # Check for the 'b' key to adjust video brightness
        elif key == ord('b'):
            brightness_value += 10

        # Check for the 'h' key to flip video feed horizontally
        elif key == ord('h'):
            flip_horizontal = not flip_horizontal

        # Check for the 'v' key to flip video feed vertically
        elif key == ord('v'):
            flip_vertical = not flip_vertical

        # Check for the 'n' key to capture and save a snapshot
        elif key == ord('n'):
            snapshot_counter += 1
            snapshot_filename = os.path.join(output_folder, f'snapshot_{snapshot_counter}.jpg')
            cv2.imwrite(snapshot_filename, frame1)
            print(f'Snapshot saved: {snapshot_filename}')

        # Check for the 'p' key to pause/resume motion detection
        elif key == ord('p'):
            motion_detection_paused = not motion_detection_paused
            if motion_detection_paused:
                print('Motion detection paused.')
            else:
                print('Motion detection resumed.')

except KeyboardInterrupt:
    # Handle KeyboardInterrupt (Ctrl+C)
    pass

finally:
    # Release resources
    pygame.mixer.quit()
    cam.release()
    cv2.destroyAllWindows()
