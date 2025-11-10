import pims  # Used for reading.cine files
import numpy as np
import matplotlib.pyplot as plt  # <-- NEW: We will use this to save the image

# --- CONFIG ---
VIDEO_PATH = "1403Hydrogenexp4Phi_1.47_5000fps.cine"  # <-- I've set this to your file
FRAME_TO_EXTRACT = 60            # <-- SET THIS (try 0 or 100 to get a clear view)
OUTPUT_IMAGE = "/Users/kartikjain/Documents/MTP/calibration_frame_60.png"
# --- END CONFIG ---

print(f"Loading {VIDEO_PATH}...")
try:
    # Use pims to open the.cine file
    frames = pims.Cine(VIDEO_PATH)
    
    if FRAME_TO_EXTRACT >= len(frames):
        print(f"Error: FRAME_TO_EXTRACT ({FRAME_TO_EXTRACT}) is out of bounds.")
        print(f"The video only has {len(frames)} frames.")
    else:
        # Get the specific frame as a numpy array
        frame = np.array(frames[FRAME_TO_EXTRACT])
        
        print(f"Frame loaded. Data type: {frame.dtype}, Shape: {frame.shape}")

        # Handle frame shape - ensure it's 2D (grayscale) or 3D with valid channels
        if frame.ndim == 3:
            # If 3D, check if it's (height, width, channels) or needs reshaping
            if frame.shape[2] > 4:
                # Unusual shape - might be transposed, try to reshape
                # Common cine format: (height, width) or (width, height)
                # If shape is (h, w, large_number), it might need to be reshaped
                print(f"Warning: Unusual frame shape {frame.shape}. Attempting to handle...")
                # Try to extract a 2D slice if possible
                if frame.shape[0] < frame.shape[2]:
                    frame = frame[:, :, 0]  # Take first channel/slice
                else:
                    frame = frame[:, :, 0]  # Take first channel/slice
            elif frame.shape[2] not in [3, 4]:
                # Not RGB/RGBA, convert to grayscale by taking first channel
                frame = frame[:, :, 0]
        
        # Ensure 2D for grayscale
        if frame.ndim == 2:
            # --- NEW SAVE METHOD ---
            # Use matplotlib.pyplot.imsave for robust saving.
            # It handles 12-bit/16-bit data and normalization automatically.
            # cmap='gray' ensures it's saved as grayscale (which it likely is)
            plt.imsave(OUTPUT_IMAGE, frame, cmap='gray')
        else:
            # If still 3D with valid channels (RGB/RGBA), save without cmap
            plt.imsave(OUTPUT_IMAGE, frame)
        
        print(f"Successfully saved frame {FRAME_TO_EXTRACT} as {OUTPUT_IMAGE}")
        print("You can now open 'calibration_frame.png' in any image editor.")

except Exception as e:
    print(f"Error: Could not read or process.cine file.")
    print(f"Make sure 'pims' and 'matplotlib' are installed.")
    print(f"Details: {e}")