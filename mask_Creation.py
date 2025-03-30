import os
import glob
import csv
import numpy as np
import cv2
from scipy.spatial import ConvexHull

def close_arc_inward_cubic(points, num_curve_points=50, frac=0.3):
    """
    Given an array of ordered points (N x 2) representing an open arc,
    close the arc by connecting the last point to the first with a cubic
    Bézier curve that leans inward (toward the centroid of the arc).

    The cubic Bézier curve is defined as:
        B(t) = (1-t)^3 * p_N + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * p_0,  t in [0,1]
    where:
        p_N is the last point,
        p_0 is the first point,
        P1 = p_N + frac*(C - p_N),
        P2 = p_0 + frac*(C - p_0),
        and C is the centroid of the arc points.
    """
    # Ensure points are float for arithmetic.
    points = points.astype(np.float32)
    p0 = points[0]
    pN = points[-1]
    
    # Compute centroid of the arc points.
    centroid = np.mean(points, axis=0)
    
    # Determine control points.
    P1 = pN + frac * (centroid - pN)
    P2 = p0 + frac * (centroid - p0)
    
    # Sample t values.
    t_vals = np.linspace(0, 1, num_curve_points)
    
    # Compute the cubic Bézier curve.
    bezier_curve = (
        ((1 - t_vals)**3)[:, np.newaxis] * pN +
        (3 * ((1 - t_vals)**2) * t_vals)[:, np.newaxis] * P1 +
        (3 * (1 - t_vals) * (t_vals**2))[:, np.newaxis] * P2 +
        ((t_vals**3))[:, np.newaxis] * p0
    )
    
    # Combine the original arc with the closing Bézier curve.
    closed_poly = np.vstack([points, bezier_curve])
    return closed_poly

def create_mask_from_txt_cubic(txt_file, shape):
    """
    Reads the pixel coordinates from a txt file, closes the open arc using
    a cubic Bézier curve that leans inward, and generates a binary mask
    by filling the resulting polygon.
    """
    points = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        x, y = int(parts[0]), int(parts[1])
                        points.append([x, y])
                    except ValueError:
                        continue
    if len(points) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    points = np.array(points)
    
    # Use the cubic Bézier method to close the arc.
    closed_poly = close_arc_inward_cubic(points, num_curve_points=50, frac=0.3)
    
    # Prepare the polygon for cv2.fillPoly (needs shape (N,1,2) and int32).
    poly_int = np.round(closed_poly).astype(np.int32).reshape((-1, 1, 2))
    
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)
    
    # Return mask as uint8 with values 0 and 255.
    return mask

# -----------------------------------------------------------
# Process dataset and save combined npy files (input & mask)
# while preserving internal folder structure.
# -----------------------------------------------------------
# Define your dataset root (update this path accordingly)
dataset_root = "/data/CME_Silhouettes/clean_files/All_Silhouettes/"
# Define output root for processed files.
output_root = "/data/CME_Silhouettes/processed_dataset_new"
os.makedirs(output_root, exist_ok=True)

# Create a list to store relative paths of processed files.
processed_files = []

# Find all npy files (assume filenames like "Image_*.npy")
npy_files = glob.glob(os.path.join(dataset_root, '**', 'Image_*.npy'), recursive=True)

for npy_file in npy_files:
    base = os.path.basename(npy_file)  # e.g., "Image_1002000_130.00.npy"
    
    # Determine the corresponding txt file.
    txt_file = os.path.join(os.path.dirname(npy_file), 
                            base.replace('Image_', 'Silhouette_ij_T_').replace('.npy', '.txt'))
    if not os.path.exists(txt_file):
        print(f"Warning: txt file for {base} not found, skipping.")
        continue

    # Load the grayscale image (expected shape: 801x801, type uint8).
    img = np.load(npy_file)  # e.g., shape (801,801), dtype=uint8

    # Create mask using the cubic Bézier closure method.
    mask = create_mask_from_txt_cubic(txt_file, (img.shape[0], img.shape[1]))  # shape (801,801), uint8 with 0/255

    # Expand mask dimensions to (H,W,1) for consistency.
    mask = np.expand_dims(mask, axis=-1)

    # Create the sample dictionary.
    sample = {'input': img, 'mask': mask}

    # Reconstruct the relative folder structure.
    relative_dir = os.path.relpath(os.path.dirname(npy_file), dataset_root)
    output_dir = os.path.join(output_root, relative_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sample in the output folder with the same base name.
    output_path = os.path.join(output_dir, base)
    np.save(output_path, sample)
    print(f"Saved {output_path}")
    
    # Save the relative path for the CSV.
    processed_files.append(os.path.relpath(output_path, output_root))

print("Dataset processing complete.")

# -----------------------------------------------------------
# Create a CSV file listing all processed files (relative to output_root)
# -----------------------------------------------------------
csv_path = os.path.join(output_root, "file_list.csv")
with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["file_path"])  # header
    for rel_path in processed_files:
        writer.writerow([rel_path])

print(f"CSV file saved to {csv_path}")



