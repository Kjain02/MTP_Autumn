import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import pims  # <-- NEW: Import PIMS library
import os
# ----------------------------------------------------------------------------
# --- USER CONFIGURATION ---
# ----------------------------------------------------------------------------

# --- Video and Duct Parameters ---
# Set the path to your video file
VIDEO_PATH = "1403Hydrogenexp4Phi_1.47_5000fps.cine"  # <-- SET THIS (can be.cine)
# Physical length of the duct in millimeters (as per the paper)
DUCT_LENGTH_MM = 1500.0  # [1]
# Frame rate of your video (3200 for the paper)
VIDEO_FPS = 5000.0       # [1]

# --- Calibration (MUST BE SET MANUALLY) ---
# Find these values by opening a frame of your video in an image editor
# The x-pixel coordinate where the flame starts (igniter)
DUCT_START_PX = 5  # <-- SET THIS
# The x-pixel coordinate where the duct ends (vent)
DUCT_END_PX = 755 # <-- SET THIS

# --- Image Processing Parameters ---
# Use Otsu's adaptive thresholding? (Recommended)
# If False, it will use MANUAL_THRESHOLD_VALUE
USE_OTSU_THRESHOLDING = True # [2]
# Manual threshold value (0-255) if not using Otsu
MANUAL_THRESHOLD_VALUE = 10
# Kernel size for morphological operations (noise removal)
MORPH_KERNEL_SIZE = 5

# --- Post-Processing Parameters ---
# Parameters for Savitzky-Golay filter [3]
# Window size (must be an odd integer). Larger = smoother.
SAVGOL_WINDOW = 25
# Polynomial order. (Must be less than window size).
SAVGOL_POLY = 3

SAVE_CONTOUR_IMAGES = True
CONTOUR_OUTPUT_FOLDER = "contour_frames"

SAVE_ORIGINAL_FRAMES = True
ORIGINAL_FRAME_FOLDER = "original_frames"

SAVE_ALL_CONTOURS_IMAGE = True
ALL_CONTOURS_OUTPUT_FOLDER = "all_contours_frames"

SAVE_PROCESSED_MASKS = True
PROCESSED_MASKS_FOLDER = "processed_masks"
# ----------------------------------------------------------------------------
# --- MAIN ANALYSIS SCRIPT ---
# ----------------------------------------------------------------------------

def analyze_flame_propagation():
    """
    Main function to process the video, extract flame data,
    and generate the required plots.
    """
    
    # --- 1. SETUP AND CALIBRATION ---
    
    # MODIFIED: Use pims.Cine to open the.cine file 
    try:
        frames = pims.Cine(VIDEO_PATH)
        print(f"Frames per second: {frames.frame_rate}")
        print(f"Total frames in video: {len(frames)}")
        print(f"Duration: {frames.get_time(-1)} seconds")
        if SAVE_CONTOUR_IMAGES:
            if not os.path.exists(CONTOUR_OUTPUT_FOLDER):
                os.makedirs(CONTOUR_OUTPUT_FOLDER)
                print(f"Created directory: {CONTOUR_OUTPUT_FOLDER}")

        if SAVE_ALL_CONTOURS_IMAGE:
            if not os.path.exists(ALL_CONTOURS_OUTPUT_FOLDER):
                os.makedirs(ALL_CONTOURS_OUTPUT_FOLDER)
                print(f"Created directory: {ALL_CONTOURS_OUTPUT_FOLDER}")


        # --- ADD THIS BLOCK ---
        if SAVE_ORIGINAL_FRAMES:
            if not os.path.exists(ORIGINAL_FRAME_FOLDER):
                os.makedirs(ORIGINAL_FRAME_FOLDER)
                print(f"Created directory: {ORIGINAL_FRAME_FOLDER}")
        # --- END ADDED BLOCK ---

        if SAVE_PROCESSED_MASKS:
            if not os.path.exists(PROCESSED_MASKS_FOLDER):
                os.makedirs(PROCESSED_MASKS_FOLDER)
                print(f"Created directory: {PROCESSED_MASKS_FOLDER}")
    except Exception as e:
        print(f"Error: Could not open video file {VIDEO_PATH} with PIMS.")
        print(f"Please ensure 'pims' is installed (pip install pims) and the file path is correct.")
        print(f"Error details: {e}")
        return

    # Calculate calibration constants
    time_step_ms = (1.0 / VIDEO_FPS) * 1000.0  # [1]
    duct_length_px = DUCT_END_PX - DUCT_START_PX
    
    if duct_length_px <= 0:
        print("Error: DUCT_END_PX must be greater than DUCT_START_PX.")
        return
        
    spatial_scale_mm_px = 1.875 #DUCT_LENGTH_MM / duct_length_px # [1]
    print(f"Video loaded. FPS: {VIDEO_FPS}, Time Step: {time_step_ms:.4f} ms/frame")
    print(f"Duct Px: {duct_length_px} px, Scale: {spatial_scale_mm_px:.4f} mm/pixel")

    # Data storage lists
    all_times_ms = []
    all_locations_px = []

    # --- 2. VIDEO PROCESSING LOOP ---
    
    # MODIFIED: Iterate through frames using the PIMS reader
    # Use enumerate to get the frame_count
    for frame_count, frame in enumerate(frames):
        
        # PIMS returns a numpy array, but we ensure it's in the right format
        # (e.g., if it's a special pims object, convert it)
        frame = np.array(frame) 

        current_time_ms = frame_count * time_step_ms

        # --- NEW: Save the original frame ---
        if SAVE_ORIGINAL_FRAMES:
            # Normalize the frame for saving as a viewable 8-bit PNG
            #.cine files are often 12-bit or 16-bit, which cv2.imwrite
            # cannot save properly as a standard PNG.
            save_frame = cv2.normalize(frame, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Construct a unique filename for each frame
            filename = os.path.join(ORIGINAL_FRAME_FOLDER, 
                                    f"original_frame_{frame_count:05d}.png")
            
            # Save the 8-bit frame
            cv2.imwrite(filename, save_frame)
        # --- END NEW ---
        
        # --- 3. CORE IMAGE PROCESSING [2] ---
        
        # MODIFIED: Handle both color (3-channel) and grayscale (2-channel).cine files
        if frame.ndim == 3:
            # If 3-channel (BGR), convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            # If 2-channel, it's already grayscale
            gray = frame
        else:
            print(f"Error: Frame {frame_count} has unexpected dimensions: {frame.shape}")
            continue
            
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarization (Thresholding) [4]
        if USE_OTSU_THRESHOLDING:
            _, binary_mask = cv2.threshold(blurred, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary_mask = cv2.threshold(blurred, MANUAL_THRESHOLD_VALUE, 255, 
                                           cv2.THRESH_BINARY)
        
        # Morphological operations (Clean up the mask) [2]
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        # Fill holes
        mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        # --- Save processed mask frames ---
        if SAVE_PROCESSED_MASKS:
            # # Save binary mask (after thresholding)
            # binary_filename = os.path.join(PROCESSED_MASKS_FOLDER, 
            #                               f"binary_mask_{frame_count:05d}.png")
            # cv2.imwrite(binary_filename, binary_mask)
            
            # # Save mask after closing (hole filling)
            # closed_filename = os.path.join(PROCESSED_MASKS_FOLDER, 
            #                               f"mask_closed_{frame_count:05d}.png")
            # cv2.imwrite(closed_filename, mask_closed)
            
            # Save final mask after opening (noise removal)
            opened_filename = os.path.join(PROCESSED_MASKS_FOLDER, 
                                           f"mask_opened_{frame_count:05d}.png")
            cv2.imwrite(opened_filename, mask_opened)
        # --- END Save processed mask frames ---
        
        # --- 4. FEATURE EXTRACTION ---

        # Find contours [5]
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)


                # --- NEW CODE: Save ALL contours image ---
        if SAVE_ALL_CONTOURS_IMAGE:
            # Create a blank black image to draw on
            all_contours_image = np.zeros_like(gray)
            # Draw all contours found in white
            cv2.drawContours(all_contours_image, contours, -1, (255), 1)
            # Construct a unique filename
            filename = os.path.join(ALL_CONTOURS_OUTPUT_FOLDER, 
                                    f"all_contours_frame_{frame_count:05d}.png")
            # Save the image
            cv2.imwrite(filename, all_contours_image)
        # --- END NEW CODE ---
            

        if contours:
            # Find the largest contour by area
            main_flame_contour = max(contours, key=cv2.contourArea)

            # --- Save contour image for debugging ---
            if SAVE_CONTOUR_IMAGES:
                # Create a blank black image to draw on (using the 'gray' frame's shape)
                contour_image = np.zeros_like(gray)
                # Draw the main contour in white
                cv2.drawContours(contour_image, [main_flame_contour], -1, (255), 2)
                # Construct a unique filename for each frame
                filename = os.path.join(CONTOUR_OUTPUT_FOLDER, f"frame_{frame_count:05d}.png")
                # Save the image
                cv2.imwrite(filename, contour_image)
            # --- Save contour image for debugging ---
            
            # Find the leading tip (max x-pixel) of this contour
            # This robustly tracks the finger tip and then the tulip cusp [1]
            # current_location_px = main_flame_contour[:, :, 0].max()

            # --- MODIFIED SECTION ---
            # Apply the y-coordinate constraint 

            # 1. Get all y-coordinates from the contour.
            #    main_flame_contour shape is (N, 1, 2). y_coords will be (N, 1).
            y_coords = main_flame_contour[:, :, 1]

            # 2. Create a boolean mask for the valid y-range.
            y_mask = (y_coords >= 135) & (y_coords <= 168)

            # 3. Get all x-coordinates.
            x_coords = main_flame_contour[:, :, 0]

            # 4. Use the y_mask to filter the x_coords.
            #    This creates a new 1D array containing ONLY the x-values
            #    where the corresponding y-value was in the valid range.
            valid_x_coords = x_coords[y_mask]

            # 5. Find the max x-coordinate *from the filtered list*
            #    We must check if the array is not empty, otherwise.max() will fail.
            if valid_x_coords.size > 0:
                # This robustly tracks the finger tip and then the tulip cusp [1]
                # *within the specified y-range*.
                current_location_px = valid_x_coords.max()
            else:
                # No points in the contour matched the y-range.
                # Set to 0 (or a value < DUCT_START_PX) to skip this frame.
                current_location_px = 0
            # --- END MODIFIED SECTION ---
            
            # Only record data if the flame is in the calibrated region
            if current_location_px >= DUCT_START_PX and ((55<=frame_count<=76) or current_location_px <= DUCT_END_PX):
                all_times_ms.append(current_time_ms)
                all_locations_px.append(current_location_px)

        # Optional: Display the processed video
        # Note: PIMS frames might be 12-bit or 16-bit. cv2.imshow expects 8-bit (0-255).
        # You may need to normalize the frame to display it correctly.
        display_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Original", display_frame)
        cv2.imshow("Mask", mask_opened)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}, Time: {current_time_ms:.2f} ms")

    # MODIFIED: No 'cap.release()' needed for PIMS
    cv2.destroyAllWindows()
    print("Video processing complete.")
    
    # --- 5. POST-PROCESSING ---
    
    if not all_times_ms:
        print("Error: No flame data was recorded. Check calibration and threshold.")
        return

    # Convert lists to NumPy arrays
    time_array = np.array(all_times_ms)
    
    # Calibrate location data (convert px to mm)
    location_array_px = np.array(all_locations_px)
    location_array_mm = (location_array_px - DUCT_START_PX) * spatial_scale_mm_px
    
    # Ensure data is within the 1500mm duct
    valid_indices = location_array_mm <= DUCT_LENGTH_MM
    time_array = time_array[valid_indices]
    location_array_mm = location_array_mm[valid_indices]
    
    if len(time_array) < SAVGOL_WINDOW:
        print(f"Error: Not enough data points ({len(time_array)}) for"
              f" Savgol filter window ({SAVGOL_WINDOW}).")
        print("Try reducing SAVGOL_WINDOW.")
        return

    # Calculate smoothed location [3]
    smooth_location_mm = savgol_filter(location_array_mm, 
                                       SAVGOL_WINDOW, SAVGOL_POLY)
    
    # Calculate velocity (m/s) using Savitzky-Golay derivative [3]
    # deriv=1 gives the 1st derivative
    # delta=time_step_ms provides the time-base
    velocity_ms = savgol_filter(location_array_mm, SAVGOL_WINDOW, SAVGOL_POLY,
                                deriv=1, delta=time_step_ms)

    print(f"Velocity: {velocity_ms}\n")
    print(f"Location: {location_array_mm}\n")
    print(f"Time: {time_array}\n")

    print("Data post-processing complete.")
    
    # --- 6. PLOTTING AND VALIDATION ---

    # Plot 1: Flame Leading Tip Location vs. Time (Fig. 4)
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, smooth_location_mm, 'b-', label='Smoothed Location')
    plt.plot(time_array, location_array_mm, 'r.', alpha=0.2, 
             label='Raw Location')
    plt.title('Flame Leading Tip Location vs. Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Flame Leading Tip Location (mm)')
    plt.grid(True)
    plt.legend()
    plt.savefig("plot_location_vs_time.png")
    print("Saved plot_location_vs_time.png")

    # Plot 2: Flame Leading Tip Velocity vs. Location (Fig. 5)
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_location_mm, velocity_ms, 'b-')
    plt.title('Flame Leading Tip Velocity vs. Location')
    plt.xlabel('Flame Leading Tip Location (mm)')
    plt.ylabel('Flame Leading Tip Velocity (m/s)')
    plt.grid(True)
    plt.savefig("plot_velocity_vs_location.png")
    print("Saved plot_velocity_vs_location.png")
    
    # --- 7. VALIDATION (Benchmarks 1 & 2) ---
    print("\n--- Validation Report ---")
    
    # Benchmark 1: Velocity Threshold [1]
    v_max = np.max(velocity_ms)
    print(f"Benchmark 1 (Velocity):")
    print(f"  > Max Flame Velocity (v_max): {v_max:.2f} m/s")
    if v_max > 31.27:
        print("  > Result: v_max is > 31.27 m/s. Paper predicts"
              " DISTORTED TULIP flame (expect velocity oscillations).")
    else:
        print("  > Result: v_max is <= 31.27 m/s. Paper predicts"
              " STABLE TULIP flame (no oscillations).")

    # Benchmark 2: Time Constant [1]
    try:
        idx_peak = np.argmax(velocity_ms)
        # Find trough (plane flame) after the peak
        idx_trough = idx_peak + np.argmin(velocity_ms[idx_peak:])
        # Find first distortion peak after the trough
        idx_distortion = idx_trough + np.argmax(velocity_ms[idx_trough:])
        
        t_plane = time_array[idx_trough]
        t_distortion = time_array[idx_distortion]
        time_delta_ms = t_distortion - t_plane
        
        print(f"\nBenchmark 2 (Timing):")
        print(f"  > Plane Flame Time (t_plane): {t_plane:.2f} ms")
        print(f"  > Initial Distortion Time (t_DTF): {t_distortion:.2f} ms")
        print(f"  > Time Interval (t_DTF - t_plane): {time_delta_ms:.2f} ms")
        print(f"  > Paper's Constant: 4.03 ms")
        if abs(time_delta_ms - 4.03) < 1.0:
            print("  > Result: SUCCESS. Calculated interval is very"
                  " close to the paper's constant.")
        else:
            print("  > Result: WARNING. Calculated interval differs"
                  " from the paper's constant.")
    except Exception as e:
        print(f"\nBenchmark 2 (Timing):")
        print(f"  > Could not perform timing analysis. Error: {e}")
        print("  > This can happen if the velocity profile is monotonic"
              " (no distortion).")
              
    plt.show()

if __name__ == "__main__":
    analyze_flame_propagation()