import os
import random
import math
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys

# Training image size 
CANVAS_SIZE = 640

# Input Directories
CARDS_ROOT = Path("Card_Scans")
BACKGROUNDS_ROOT = Path("backgrounds")

# Output Directory
OUTPUT_ROOT = Path(f"dataset_pose_{CANVAS_SIZE}")

# Example Dataset Split Counts ~train 80% ~val 10% ~test 10%
SPLITS = {
    "train": 8,
    "val": 1,
    "test": 1,
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_all_image_paths(directory: Path):
    """Recursively fetch all valid image files from a directory."""
    paths = []
    if not directory.exists():
        return paths
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    return paths

def load_background(bg_path, size=CANVAS_SIZE):
    """Load and scale/crop a background image specified CANVAS_SIZE x CANVAS_SIZE."""
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    
    h, w = bg.shape[:2]

    min_dim = min(h, w)
    sy = random.randint(0, h - min_dim)
    sx = random.randint(0, w - min_dim)
    cropped = bg[sy:sy + min_dim, sx:sx + min_dim] 
    
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)

def apply_card_effects(card_bgr):
    """Apply random color/contrast jitter directly to the card image."""
    # Brightness & Contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        card_bgr = cv2.convertScaleAbs(card_bgr, alpha=alpha, beta=beta)
    return card_bgr

def add_camera_effects(image):
    """Apply environmental effects (Reflections, Blur, Noise)."""
    h, w = image.shape[:2]
    if random.random() < 0.5:
        streak_angle_deg = 45
    else:
        streak_angle_deg = random.uniform(0, 360)

    angle_rad = math.radians(streak_angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    max_length = int(math.hypot(w, h))

    # 1. Random Glare / Reflection streak across the image
    if random.random() < 0.5:
        overlay = image.copy()
        pt1 = (random.randint(0, w), random.randint(0, h))
        length = random.randint(int(max_length * 0.5), max_length)
        pt2 = (int(pt1[0] + dx * length), int(pt1[1] + dy * length))

        thickness = random.randint(40, 120)
        color = (random.randint(225, 255), random.randint(225, 255), random.randint(225, 255))
        cv2.line(overlay, pt1, pt2, color, thickness)
        
        blur_amt = random.choice([31, 51, 71])
        overlay = cv2.GaussianBlur(overlay, (blur_amt, blur_amt), 0)
        
        alpha = random.uniform(0.25, 0.90)
        image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    # Few reflection streak across the image
    for i in range(0,10):
        if random.random() < 0.2:
            overlay = image.copy()
            pt1 = (random.randint(0, w), random.randint(0, h))
            length = random.randint(int(max_length * 0.3), max_length)
            pt2 = (int(pt1[0] + dx * length), int(pt1[1] + dy * length))

            thickness = random.randint(20, 40)
            color = (random.randint(225, 255), random.randint(225, 255), random.randint(225, 255))
            cv2.line(overlay, pt1, pt2, color, thickness)

    # 2. Gaussian Blur (Out-of-focus camera simulation)
    if random.random() < 0.12:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    # 3. Sensor Noise
    if random.random() < 0.27:
        noise = np.random.normal(0, random.uniform(2, 6), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


def transform_and_place_card(card_img, bg_img, canvas_size=CANVAS_SIZE):
    """Warp card perspective, compute 4 corners, and blend onto background."""
    h_orig, w_orig = card_img.shape[:2]

    # Extract or create Alpha Channel for card cutout
    if card_img.shape[2] == 4:
        card_bgr = card_img[:, :, :3]
        card_alpha = card_img[:, :, 3]
    else:
        card_bgr = card_img
        card_alpha = np.ones((h_orig, w_orig), dtype=np.uint8) * 255

    card_bgr = apply_card_effects(card_bgr)
    #card_img, card_alpha = add_random_skin_blurs(card_bgr, card_alpha, n=3)
    
    # 1. Random Scaling (Card occupies between 40% and 70% of canvas height)
    target_h = canvas_size * random.uniform(0.40, 0.70)
    target_w = target_h * (w_orig / h_orig)
    hw, hh = target_w / 2.0, target_h / 2.0

    # 2. Calculate a strict Safe Margin (Diagonal Radius + Jitter)
    max_radius = math.hypot(hw, hh)
    jitter_max = min(target_w, target_h) * 0.10
    
    # Add a tiny 2-pixel buffer just to be absolutely safe
    safe_margin = max_radius + jitter_max + 2.0 

    # 3. Random Center Position (Guaranteed safe bounds)
    cx = random.uniform(safe_margin, canvas_size - safe_margin)
    cy = random.uniform(safe_margin, canvas_size - safe_margin)

    # 4. Define Base Unrotated Corners centered at (0,0)
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh] 
    ], dtype=np.float32)

    # 5. Rotation (+- 35 degrees)
    angle_deg = random.uniform(-35.0, 35.0)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated_corners = corners @ rot_matrix.T

    # 6. Perspective Jitter
    jitter = np.random.uniform(-jitter_max, jitter_max, size=(4, 2)).astype(np.float32)

    # Final destination points on canvas
    #expanded_corners = rotated_corners + np.array(
    #    [
    #        [-1, -1],
    #        [1, -1],
    #        [1, 1],
    #        [-1, 1],
    #    ],
    #    dtype=np.float32,
    #)
    dst_pts = rotated_corners + jitter + np.array([cx, cy], dtype=np.float32)

    src_pts = np.array([
        [0, 0],
        [w_orig, 0],
        [w_orig, h_orig],
        [0, h_orig]
    ], dtype=np.float32)

    # Homography matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_card = cv2.warpPerspective(card_bgr, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
    warped_alpha = cv2.warpPerspective(card_alpha, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)

    # 6. Drop Shadow
    shadow_mask = cv2.warpAffine(
        warped_alpha, 
        np.float32([[1, 0, random.randint(4, 10)], [0, 1, random.randint(4, 10)]]), 
        (canvas_size, canvas_size)
    )
    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    shadow_factor = 1.0 - (shadow_mask.astype(np.float32) / 255.0 * random.uniform(0.3, 0.5))[:, :, np.newaxis]
    bg_img = (bg_img.astype(np.float32) * shadow_factor).astype(np.uint8)

    # 7. Alpha Blend Card onto Background
    alpha_factor = (warped_alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]

    card_weffects = add_camera_effects(warped_card)

    final_img = (card_weffects.astype(np.float32) * alpha_factor + 
                 bg_img.astype(np.float32) * (1.0 - alpha_factor)).astype(np.uint8)

    # 8. Post-processing camera effects
    #final_img = add_camera_effects(final_img)

    return final_img, dst_pts

def generate_yolo_pose_label(dst_pts, class_id=0, size=CANVAS_SIZE):
    """Normalize corners to [0.0, 1.0] and format as YOLO Pose string."""
    norm_pts = np.clip(dst_pts / size, 0.0, 1.0)
    
    xs = norm_pts[:, 0]
    ys = norm_pts[:, 1]
    
    x_center = (xs.max() + xs.min()) / 2.0
    y_center = (ys.max() + ys.min()) / 2.0
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    
    # Format keypoints (x, y, visibility=2)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    kpts_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f} 2" for pt in norm_pts])
    
    # Combine into final label string
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kpts_str}\n"


def process_single_sample(args):
    split, idx, card_paths, bg_paths, canvas_size, output_root = args
    filename = f"pokemon_sample_{idx:06d}"

    # Pick Random Card & Background
    card_path = random.choice(card_paths)
    card_img = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)

    # Skip corrupted/unreadable cards
    while card_img is None:
        card_path = random.choice(card_paths)
        card_img = cv2.imread(str(card_path), cv2.IMREAD_UNCHANGED)

    if bg_paths:
        bg_path = random.choice(bg_paths)
        bg_img = load_background(bg_path, canvas_size)
    else:
        bg_img = np.random.randint(50, 200, (canvas_size, canvas_size, 3), dtype=np.uint8)

    # Process Image and Get Corner Points
    final_img, dst_pts = transform_and_place_card(card_img, bg_img, canvas_size)
    label_text = generate_yolo_pose_label(dst_pts, class_id=0, size=canvas_size)

    # Save Files
    out_img_path = output_root / "images" / split / f"{filename}.jpg"
    out_lbl_path = output_root / "labels" / split / f"{filename}.txt"

    cv2.imwrite(str(out_img_path), final_img)
    with open(out_lbl_path, "w") as f:
        f.write(label_text)

def main(num_workers, chunk):
    print("Scanning input image folders...")
    card_paths = get_all_image_paths(CARDS_ROOT)
    bg_paths = get_all_image_paths(BACKGROUNDS_ROOT)

    if not card_paths:
        raise FileNotFoundError(f"No card images found in '{CARDS_ROOT}'. Check folder path!")
    
    if not bg_paths:
        print(f"Warning: No background images found in '{BACKGROUNDS_ROOT}'. Generating artificial backgrounds instead.")

    print(f"Found {len(card_paths)} card images.")
    print(f"Found {len(bg_paths)} background images.")

    # Create Directory Structure
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_counter = 0
    num_threads = num_workers #multiprocessing.cpu_count()
    print(f"\n⚡ Utilizing {num_threads} CPU threads for processing.")

    for split, count in SPLITS.items():
        print(f"\nGenerating '{split}' split ({count} images)...")
        
        # Prepare task argument list
        tasks = [
            (split, img_counter + i + 1, card_paths, bg_paths, CANVAS_SIZE, OUTPUT_ROOT)
            for i in range(count)
        ]
        img_counter += count

        # Execute image creation in parallel across available CPU threads
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(process_single_sample, tasks, chunksize=chunk), total=count))

    # Generate data.yaml config for YOLO POSE
    yaml_content = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test

names:
  0: pokemon_card
  
kpt_shape: [4, 3]
"""
    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n DATASET GENERATION COMPLETE!")
    print(f"Dataset saved to: '{OUTPUT_ROOT.resolve()}'")
    print(f"Config created at: '{yaml_path.resolve()}'")

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python calibrate_thresholds.py [num_workers] [chunksize]", flush=True)
        sys.exit(1)
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else multiprocessing.cpu_count()
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    num_workers = min(num_workers, multiprocessing.cpu_count() - 1)
    main(num_workers, chunk)