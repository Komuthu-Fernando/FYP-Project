from collections import deque
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
import os
import time
import psutil
from memory_profiler import memory_usage
import logging

# Set up logging for performance metrics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directory for cropped characters
output_dir = "./cropped_final_characters"
os.makedirs(output_dir, exist_ok=True)

# Function to measure CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=0.1)

# Function to monitor performance of a specific stage
def monitor_stage(stage_name, func, *args, **kwargs):
    start_time = time.time()
    cpu_usages = []
    mem_usage = []

    # Monitor CPU and memory during execution
    def run_with_monitoring():
        while not done[0]:
            cpu_usages.append(get_cpu_usage())
            mem_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))  # Memory in MB
            time.sleep(0.01)

    done = [False]
    import threading
    monitor_thread = threading.Thread(target=run_with_monitoring)
    monitor_thread.start()

    # Execute the function
    result = func(*args, **kwargs)
    
    # Stop monitoring
    done[0] = True
    monitor_thread.join()

    end_time = time.time()
    execution_time = end_time - start_time
    avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
    max_cpu = np.max(cpu_usages) if cpu_usages else 0
    max_mem = np.max(mem_usage) if mem_usage else 0

    # Log performance metrics
    logger.info(f"Stage: {stage_name}")
    logger.info(f"  Execution Time: {execution_time:.2f} seconds")
    logger.info(f"  Average CPU Usage: {avg_cpu:.2f}%")
    logger.info(f"  Peak CPU Usage: {max_cpu:.2f}%")
    logger.info(f"  Peak Memory Usage: {max_mem:.2f} MB")

    return result, execution_time, avg_cpu, max_cpu, max_mem

# Initialize performance metrics dictionary
performance_metrics = {
    'stages': {},
    'total_time': 0,
    'total_avg_cpu': [],
    'total_max_cpu': [],
    'total_max_mem': []
}

# Load image and convert to grayscale
def load_and_convert_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' not found")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary_inverted = cv.bitwise_not(gray)
    return image, gray, binary_inverted

imageNo = 42.1
image_path = f"{imageNo}.png"
(image, gray, binary_inverted), load_time, load_avg_cpu, load_max_cpu, load_max_mem = monitor_stage(
    "Image Loading and Conversion", load_and_convert_image, image_path
)
performance_metrics['stages']['Image Loading'] = {
    'time': load_time, 'avg_cpu': load_avg_cpu, 'max_cpu': load_max_cpu, 'max_mem': load_max_mem
}
performance_metrics['total_time'] += load_time
performance_metrics['total_avg_cpu'].append(load_avg_cpu)
performance_metrics['total_max_cpu'].append(load_max_cpu)
performance_metrics['total_max_mem'].append(load_max_mem)

# Save and prepare to display dilated image
cv.imwrite("./images/dilated.png", binary_inverted)

# Get image dimensions
height, width = binary_inverted.shape

# Define white and black pixel values
WHITE = 255
BLACK = 0

# Function to count connected pixels and return their coordinates
def count_connected_pixels(image, start_y, start_x, visited, target_value):
    h, w = image.shape
    count = 0
    pixels = []
    queue = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    while queue:
        y, x = queue.popleft()
        count += 1
        pixels.append((y, x))
        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            if (0 <= new_y < h and 0 <= new_x < w and 
                not visited[new_y, new_x] and 
                image[new_y, new_x] == target_value):
                queue.append((new_y, new_x))
                visited[new_y, new_x] = True
    return count, pixels

# Function to analyze black pixel clusters
def analyze_black_pixel_clusters(image):
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    clusters = []
    
    for y in range(h):
        for x in range(w):
            if image[y, x] == BLACK and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, BLACK)
                if pixels:
                    y_coords, x_coords = zip(*pixels)
                    min_y, max_y = min(y_coords), max(y_coords)
                    min_x, max_x = min(x_coords), max(x_coords)
                    clusters.append({
                        'size': pixel_count,
                        'bounding_box': (min_x, min_y, max_x, max_y),
                        'start_pixel': (y, x),
                        'center': ((min_x + max_x) / 2, (min_y + max_y) / 2),
                        'pixels': pixels
                    })
    
    clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    average_size = sum(cluster['size'] for cluster in clusters) / len(clusters) if clusters else 0
    return clusters, average_size

# Updated function to merge vertically aligned clusters, ignoring clusters below
def merge_vertical_clusters(clusters):
    if not clusters:
        return clusters
    merged_clusters = []
    used = [False] * len(clusters)
    OVERLAP_THRESHOLD = 0.7
    
    for i, cluster in enumerate(clusters):
        if used[i]:
            continue
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        cluster_width = max_x - min_x + 1
        merged_pixels = set(cluster['pixels'])
        merged_size = cluster['size']
        merged_start_pixel = cluster['start_pixel']
        merged_center = [cluster['center'][0], cluster['center'][1]]
        count = 1
        
        best_overlap = 0
        best_j = -1
        for j, other_cluster in enumerate(clusters):
            if i == j or used[j]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            if other_min_y > max_y:
                continue
            other_width = other_max_x - other_min_x + 1
            overlap_start = max(min_x, other_min_x)
            overlap_end = min(max_x, other_max_x)
            overlap_width = max(0, overlap_end - overlap_start + 1)
            smaller_width = min(cluster_width, other_width)
            containment_ratio = overlap_width / smaller_width if smaller_width > 0 else 0
            if containment_ratio >= OVERLAP_THRESHOLD and containment_ratio > best_overlap:
                best_overlap = containment_ratio
                best_j = j
        
        if best_j != -1:
            other_cluster = clusters[best_j]
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            merged_pixels.update(other_cluster['pixels'])
            merged_size += other_cluster['size']
            min_x = min(min_x, other_min_x)
            max_x = max(max_x, other_max_x)
            min_y = min(min_y, other_min_y)
            max_y = max(max_y, other_max_y)
            merged_center[0] = (merged_center[0] * count + other_cluster['center'][0]) / (count + 1)
            merged_center[1] = (merged_center[1] * count + other_cluster['center'][1]) / (count + 1)
            count += 1
            used[best_j] = True
            print(f"Merged cluster at ({min_x}, {min_y}) with cluster at ({other_min_x}, {other_min_y})")
        
        merged_clusters.append({
            'size': merged_size,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'start_pixel': merged_start_pixel,
            'center': (merged_center[0], merged_center[1]),
            'pixels': list(merged_pixels)
        })
        used[i] = True
    
    merged_clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    return merged_clusters

# Function to merge clusters based on bounding box containment
def merge_contained_clusters(clusters, containment_threshold=0.7):
    if not clusters:
        return clusters
    merged_clusters = []
    used = [False] * len(clusters)
    
    for i, cluster in enumerate(clusters):
        if used[i]:
            continue
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        cluster_width = max_x - min_x + 1
        cluster_height = max_y - min_y + 1
        cluster_area = cluster_width * cluster_height
        merged_pixels = set(cluster['pixels'])
        merged_size = cluster['size']
        merged_start_pixel = cluster['start_pixel']
        merged_center = [cluster['center'][0], cluster['center'][1]]
        count = 1
        
        for j, other_cluster in enumerate(clusters[i+1:], start=i+1):
            if used[j]:
                continue
            other_min_x, other_min_y, other_max_x, other_max_y = other_cluster['bounding_box']
            other_width = other_max_x - other_min_x + 1
            other_height = other_max_y - other_min_y + 1
            other_area = other_width * other_height
            
            intersect_min_x = max(min_x, other_min_x)
            intersect_max_x = min(max_x, other_max_x)
            intersect_min_y = max(min_y, other_min_y)
            intersect_max_y = min(max_y, other_max_y)
            intersect_width = max(0, intersect_max_x - intersect_min_x + 1)
            intersect_height = max(0, intersect_max_y - intersect_min_y + 1)
            intersect_area = intersect_width * intersect_height
            
            smaller_area = min(cluster_area, other_area)
            containment_ratio = intersect_area / smaller_area if smaller_area > 0 else 0
            
            if containment_ratio >= containment_threshold:
                merged_pixels.update(other_cluster['pixels'])
                merged_size += other_cluster['size']
                min_x = min(min_x, other_min_x)
                max_x = max(max_x, other_max_x)
                min_y = min(min_y, other_min_y)
                max_y = max(max_y, other_max_y)
                merged_center[0] = (merged_center[0] * count + other_cluster['center'][0]) / (count + 1)
                merged_center[1] = (merged_center[1] * count + other_cluster['center'][1]) / (count + 1)
                count += 1
                used[j] = True
                print(f"Merged cluster at ({other_min_x}, {other_min_y}) "
                      f"into cluster at ({min_x}, {min_y}), "
                      f"containment ratio: {containment_ratio:.2f}")
        
        merged_clusters.append({
            'size': merged_size,
            'bounding_box': (min_x, min_y, max_x, max_y),
            'start_pixel': merged_start_pixel,
            'center': (merged_center[0], merged_center[1]),
            'pixels': list(merged_pixels)
        })
        used[i] = True
    
    merged_clusters.sort(key=lambda cluster: cluster['bounding_box'][0])
    return merged_clusters

# Function to analyze white pixel clusters within the largest black cluster
def analyze_white_clusters_within_black_cluster(image, black_cluster_pixels):
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    white_cluster_sizes = []
    maskimo = np.zeros((h, w), dtype=np.uint8)
    for y, x in black_cluster_pixels:
        maskimo[y, x] = 255
    for y, x in black_cluster_pixels:
        if image[y, x] == WHITE and not visited[y, x]:
            pixel_count, _ = count_connected_pixels(image, y, x, visited, WHITE)
            white_cluster_sizes.append(pixel_count)
    return white_cluster_sizes

# Function to create an irregularly cropped image
def create_irregular_crop(image, black_cluster_pixels):
    h, w = image.shape
    cropped_image = np.ones((h, w), dtype=np.uint8) * WHITE
    if black_cluster_pixels:
        y_coords, x_coords = zip(*black_cluster_pixels)
        min_y, max_y = min(y_coords), max(y_coords)
        min_x, max_x = 0, w - 1
        cropped_image[min_y:max_y+1, min_x:max_x+1] = image[min_y:max_y+1, min_x:max_x+1]
    return cropped_image

# Clean and save image
cleaned_image = binary_inverted
cv.imwrite("./images/cleaned_image.png", cleaned_image)

# Analyze black pixel clusters
(black_clusters, average_size), cluster_time, cluster_avg_cpu, cluster_max_cpu, cluster_max_mem = monitor_stage(
    "Black Pixel Cluster Analysis", analyze_black_pixel_clusters, cleaned_image
)
performance_metrics['stages']['Cluster Analysis'] = {
    'time': cluster_time, 'avg_cpu': cluster_avg_cpu, 'max_cpu': cluster_max_cpu, 'max_mem': cluster_max_mem
}
performance_metrics['total_time'] += cluster_time
performance_metrics['total_avg_cpu'].append(cluster_avg_cpu)
performance_metrics['total_max_cpu'].append(cluster_max_cpu)
performance_metrics['total_max_mem'].append(cluster_max_mem)

# Find largest black cluster and crop
def process_largest_cluster(image, clusters):
    if clusters:
        largest_cluster = max(clusters, key=lambda cluster: cluster['size'])
        pixels = largest_cluster['pixels']
        cropped = create_irregular_crop(image, pixels)
        print("Largest black cluster found")
        return largest_cluster, cropped
    else:
        print("\nNo black pixel clusters found in cleaned_image.")
        return None, image.copy()

(largest_black_cluster, cropped_image), crop_time, crop_avg_cpu, crop_max_cpu, crop_max_mem = monitor_stage(
    "Largest Cluster Cropping", process_largest_cluster, cleaned_image, black_clusters
)
performance_metrics['stages']['Cropping'] = {
    'time': crop_time, 'avg_cpu': crop_avg_cpu, 'max_cpu': crop_max_cpu, 'max_mem': crop_max_mem
}
performance_metrics['total_time'] += crop_time
performance_metrics['total_avg_cpu'].append(crop_avg_cpu)
performance_metrics['total_max_cpu'].append(crop_max_cpu)
performance_metrics['total_max_mem'].append(crop_max_mem)

cv.imwrite("./images/cropped_image.png", cropped_image)

# Calculate cluster statistics
def calculate_cluster_stats(clusters):
    if clusters:
        cluster_sizes = [cluster['size'] for cluster in clusters]
        largest_size = max(cluster_sizes)
        smallest_size = min(cluster_sizes)
        median_size = np.median(cluster_sizes)
        largest_cluster = max(clusters, key=lambda cluster: cluster['size'])
        print("\nBlack Pixel Cluster Statistics after scan_and_clean, cropping, vertical merging, and containment merging:")
        print(f"Largest cluster size: {largest_size} pixels")
        print(f"Smallest cluster size: {smallest_size} pixels")
        print(f"Median cluster size: {median_size:.2f} pixels")
        return largest_cluster, largest_size, smallest_size, median_size
    else:
        print("\nNo black pixel clusters found after scan_and_clean, cropping, vertical merging, and containment merging.")
        return None, 0, 0, 0

(largest_cluster, largest_size, smallest_size, median_size), stats_time, stats_avg_cpu, stats_max_cpu, stats_max_mem = monitor_stage(
    "Cluster Statistics", calculate_cluster_stats, black_clusters
)
performance_metrics['stages']['Cluster Statistics'] = {
    'time': stats_time, 'avg_cpu': stats_avg_cpu, 'max_cpu': stats_max_cpu, 'max_mem': stats_max_mem
}
performance_metrics['total_time'] += stats_time
performance_metrics['total_avg_cpu'].append(stats_avg_cpu)
performance_metrics['total_max_cpu'].append(stats_max_cpu)
performance_metrics['total_max_mem'].append(stats_max_mem)

# Function to remove noise based on connected white pixel count
def remove_noise(image, threshold):
    h, w = image.shape
    result = image.copy()
    visited = np.zeros((h, w), dtype=bool)
    changes_made = False
    for y in range(h):
        for x in range(w):
            if image[y, x] == WHITE and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, WHITE)
                print(f"White pixel cluster at ({y}, {x}): {pixel_count} pixels")
                if pixel_count < threshold:
                    for py, px in pixels:
                        result[py, px] = BLACK
                    changes_made = True
    return result, changes_made

# First Pass: Noise removal
THRESHOLD = 6400
max_iterations = 10
iteration = 0
current_image = cropped_image.copy()

def first_pass_noise_removal(image, threshold, max_iterations):
    iteration = 0
    current = image.copy()
    while iteration < max_iterations:
        print(f"\nFirst Pass - Iteration {iteration + 1}")
        denoised, changes_made = remove_noise(current, threshold)
        if not changes_made:
            print("No more noise detected in first pass, stopping iterations")
            break
        current = denoised
        iteration += 1
    return current

current_image, noise1_time, noise1_avg_cpu, noise1_max_cpu, noise1_max_mem = monitor_stage(
    "First Pass Noise Removal", first_pass_noise_removal, current_image, THRESHOLD, max_iterations
)
performance_metrics['stages']['First Pass Noise Removal'] = {
    'time': noise1_time, 'avg_cpu': noise1_avg_cpu, 'max_cpu': noise1_max_cpu, 'max_mem': noise1_max_mem
}
performance_metrics['total_time'] += noise1_time
performance_metrics['total_avg_cpu'].append(noise1_avg_cpu)
performance_metrics['total_max_cpu'].append(noise1_max_cpu)
performance_metrics['total_max_mem'].append(noise1_max_mem)

cv.imwrite("./images/denoised_image.png", current_image)

# Invert the denoised image
def invert_image(image):
    return cv.bitwise_not(image)

binary_inverted_current_image, invert_time, invert_avg_cpu, invert_max_cpu, invert_max_mem = monitor_stage(
    "Image Inversion", invert_image, current_image
)
performance_metrics['stages']['Image Inversion'] = {
    'time': invert_time, 'avg_cpu': invert_avg_cpu, 'max_cpu': invert_max_cpu, 'max_mem': invert_max_mem
}
performance_metrics['total_time'] += invert_time
performance_metrics['total_avg_cpu'].append(invert_avg_cpu)
performance_metrics['total_max_cpu'].append(invert_max_cpu)
performance_metrics['total_max_mem'].append(invert_max_mem)

cv.imwrite("./images/denoised_inverted_image.png", binary_inverted_current_image)

# Function to convert black pixels to white
def convert_black_to_white(image):
    h, w = image.shape
    result = image.copy()
    for y in range(h):
        for x in range(w):
            if image[y, x] == WHITE:
                break
            if image[y, x] == BLACK:
                result[y, x] = WHITE
    for y in range(h):
        for x in range(w-1, -1, -1):
            if image[y, x] == WHITE:
                break
            if image[y, x] == BLACK:
                result[y, x] = WHITE
    for x in range(w):
        for y in range(h):
            if image[y, x] == WHITE:
                break
            if image[y, x] == BLACK:
                result[y, x] = WHITE
    for x in range(w):
        for y in range(h-1, -1, -1):
            if image[y, x] == WHITE:
                break
            if image[y, x] == BLACK:
                result[y, x] = WHITE
    return result

# Apply black-to-white conversion
final_image, convert_time, convert_avg_cpu, convert_max_cpu, convert_max_mem = monitor_stage(
    "Black to White Conversion", convert_black_to_white, binary_inverted_current_image
)
performance_metrics['stages']['Black to White Conversion'] = {
    'time': convert_time, 'avg_cpu': convert_avg_cpu, 'max_cpu': convert_max_cpu, 'max_mem': convert_max_mem
}
performance_metrics['total_time'] += convert_time
performance_metrics['total_avg_cpu'].append(convert_avg_cpu)
performance_metrics['total_max_cpu'].append(convert_max_cpu)
performance_metrics['total_max_mem'].append(convert_max_mem)

cv.imwrite("./images/final_image.png", final_image)

# Function to remove noise based on connected black pixel count
def remove_noise_black(image, threshold):
    h, w = image.shape
    result = image.copy()
    visited = np.zeros((h, w), dtype=bool)
    changes_made = False
    for y in range(h):
        for x in range(w):
            if image[y, x] == BLACK and not visited[y, x]:
                pixel_count, pixels = count_connected_pixels(image, y, x, visited, BLACK)
                print(f"Black pixel cluster at ({y}, {x}): {pixel_count} pixels")
                if pixel_count < threshold:
                    for py, px in pixels:
                        result[py, px] = WHITE
                    changes_made = True
    return result, changes_made

# Second Pass: Noise removal
def second_pass_noise_removal(image, threshold, max_iterations):
    iteration = 0
    current = image.copy()
    while iteration < max_iterations:
        print(f"\nSecond Pass - Iteration {iteration + 1}")
        denoised, changes_made = remove_noise_black(current, threshold)
        if not changes_made:
            print("No more noise detected in second pass, stopping iterations")
            break
        current = denoised
        iteration += 1
    return current

final_image_second_pass, noise2_time, noise2_avg_cpu, noise2_max_cpu, noise2_max_mem = monitor_stage(
    "Second Pass Noise Removal", second_pass_noise_removal, final_image, THRESHOLD, max_iterations
)
performance_metrics['stages']['Second Pass Noise Removal'] = {
    'time': noise2_time, 'avg_cpu': noise2_avg_cpu, 'max_cpu': noise2_max_cpu, 'max_mem': noise2_max_mem
}
performance_metrics['total_time'] += noise2_time
performance_metrics['total_avg_cpu'].append(noise2_avg_cpu)
performance_metrics['total_max_cpu'].append(noise2_max_cpu)
performance_metrics['total_max_mem'].append(noise2_max_mem)

cv.imwrite("./images/final_image_second_pass.png", final_image_second_pass)

# Analyze black pixel clusters in final image
def process_final_clusters(image):
    clusters, avg_size = analyze_black_pixel_clusters(image)
    clusters = merge_vertical_clusters(clusters)
    clusters = merge_contained_clusters(clusters, containment_threshold=0.7)
    return clusters, avg_size

(clusters, average_size), final_cluster_time, final_cluster_avg_cpu, final_cluster_max_cpu, final_cluster_max_mem = monitor_stage(
    "Final Cluster Analysis and Merging", process_final_clusters, final_image_second_pass
)
performance_metrics['stages']['Final Cluster Analysis'] = {
    'time': final_cluster_time, 'avg_cpu': final_cluster_avg_cpu, 'max_cpu': final_cluster_max_cpu, 'max_mem': final_cluster_max_mem
}
performance_metrics['total_time'] += final_cluster_time
performance_metrics['total_avg_cpu'].append(final_cluster_avg_cpu)
performance_metrics['total_max_cpu'].append(final_cluster_max_cpu)
performance_metrics['total_max_mem'].append(final_cluster_max_mem)

# Calculate character widths and Modified Z-Scores
def calculate_z_scores(clusters):
    character_widths = []
    modified_z_scores = []
    if clusters:
        for cluster in clusters:
            min_x, min_y, max_x, max_y = cluster['bounding_box']
            char_width = max_x - min_x + 1
            character_widths.append(char_width)

        if character_widths:
            median_width = np.median(character_widths)
            absolute_deviations = [abs(width - median_width) for width in character_widths]
            mad = np.median(absolute_deviations) if absolute_deviations else 0
            if mad != 0:
                modified_z_scores = [0.6745 * abs(width - median_width) / mad for width in character_widths]
            else:
                modified_z_scores = [0] * len(character_widths)
                print("Warning: MAD is zero, setting Modified Z-Scores to 0.")
            
            print(f"\nCharacter Width Statistics (Modified Z-Score Method):")
            print(f"Character widths: {character_widths}")
            print(f"Median width: {median_width:.2f} pixels")
            print(f"MAD: {mad:.2f} pixels")
            print(f"Modified Z-Scores: {[f'{z:.2f}' for z in modified_z_scores]}")
            print(f"Unusual threshold: Modified Z-Score > 3 (split into two)")
        else:
            print("\nNo character widths to analyze.")
            median_width = mad = None
            modified_z_scores = []
    else:
        print("\nNo character clusters to analyze.")
        median_width = mad = None
        modified_z_scores = []
    return character_widths, modified_z_scores, median_width, mad

(character_widths, modified_z_scores, median_width, mad), zscore_time, zscore_avg_cpu, zscore_max_cpu, zscore_max_mem = monitor_stage(
    "Z-Score Calculation", calculate_z_scores, clusters
)
performance_metrics['stages']['Z-Score Calculation'] = {
    'time': zscore_time, 'avg_cpu': zscore_avg_cpu, 'max_cpu': zscore_max_cpu, 'max_mem': zscore_max_mem
}
performance_metrics['total_time'] += zscore_time
performance_metrics['total_avg_cpu'].append(zscore_avg_cpu)
performance_metrics['total_max_cpu'].append(zscore_max_cpu)
performance_metrics['total_max_mem'].append(zscore_max_mem)

# Process clusters: handle high outliers based on Modified Z-Scores
def process_clusters(clusters, modified_z_scores, image, output_dir, imageNo):
    final_clusters = []
    char_index = 1
    if clusters:
        for filename in os.listdir(output_dir):
            if filename.startswith(f"{imageNo}_image_character_") and filename.endswith(".png"):
                file_path = os.path.join(output_dir, filename)
                os.remove(file_path)
                print(f"Removed existing file: {file_path}")

        for idx, cluster in enumerate(clusters):
            min_x, min_y, max_x, max_y = cluster['bounding_box']
            char_width = max_x - min_x + 1
            modified_z_score = modified_z_scores[idx] if idx < len(modified_z_scores) else 0
            
            if modified_z_score > 3:
                print(f"\nDividing high outlier cluster: Width={char_width:.2f}, Modified Z-Score={modified_z_score:.2f}")
                print(f"  Bounding Box: (min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y})")
                print(f"  Size: {cluster['size']} pixels")
                
                mid_x = min_x + (max_x - min_x) // 2
                sub_pixels1 = [(y, x) for y, x in cluster['pixels'] if min_x <= x <= mid_x]
                if sub_pixels1:
                    sub_size1 = len(sub_pixels1)
                    sub_center_x1 = sum(x for y, x in sub_pixels1) / sub_size1
                    sub_center_y1 = sum(y for y, x in sub_pixels1) / sub_size1
                    final_clusters.append({
                        'size': sub_size1,
                        'bounding_box': (min_x, min_y, mid_x, max_y),
                        'start_pixel': (min_y, min_x),
                        'center': (sub_center_x1, sub_center_y1),
                        'pixels': sub_pixels1,
                        'char_index': char_index,
                        'is_sub_character': True
                    })
                    char_filename1 = os.path.join(output_dir, f"{imageNo}_image_character_{char_index}.png")
                    char_image1 = image[min_y:max_y+1, min_x:mid_x+1]
                    cv.imwrite(char_filename1, char_image1)
                    print(f"Saved sub-character {char_index} to {char_filename1}")
                    char_index += 1
                
                sub_pixels2 = [(y, x) for y, x in cluster['pixels'] if mid_x + 1 <= x <= max_x]
                if sub_pixels2:
                    sub_size2 = len(sub_pixels2)
                    sub_center_x2 = sum(x for y, x in sub_pixels2) / sub_size2
                    sub_center_y2 = sum(y for y, x in sub_pixels2) / sub_size2
                    final_clusters.append({
                        'size': sub_size2,
                        'bounding_box': (mid_x + 1, min_y, max_x, max_y),
                        'start_pixel': (min_y, mid_x + 1),
                        'center': (sub_center_x2, sub_center_y2),
                        'pixels': sub_pixels2,
                        'char_index': char_index,
                        'is_sub_character': True
                    })
                    char_filename2 = os.path.join(output_dir, f"{imageNo}_image_character_{char_index}.png")
                    char_image2 = image[min_y:max_y+1, mid_x+1:max_x+1]
                    cv.imwrite(char_filename2, char_image2)
                    print(f"Saved sub-character {char_index} to {char_filename2}")
                    char_index += 1
            else:
                cluster['char_index'] = char_index
                cluster['is_sub_character'] = False
                final_clusters.append(cluster)
                char_filename = os.path.join(output_dir, f"{imageNo}_image_character_{char_index}.png")
                char_image = image[min_y:max_y+1, min_x:max_x+1]
                cv.imwrite(char_filename, char_image)
                print(f"Saved character {char_index} to {char_filename}")
                char_index += 1
    return final_clusters, char_index

(final_clusters, char_index), process_time, process_avg_cpu, process_max_cpu, process_max_mem = monitor_stage(
    "Cluster Processing", process_clusters, clusters, modified_z_scores, final_image_second_pass, output_dir, imageNo
)
performance_metrics['stages']['Cluster Processing'] = {
    'time': process_time, 'avg_cpu': process_avg_cpu, 'max_cpu': process_max_cpu, 'max_mem': process_max_mem
}
performance_metrics['total_time'] += process_time
performance_metrics['total_avg_cpu'].append(process_avg_cpu)
performance_metrics['total_max_cpu'].append(process_max_cpu)
performance_metrics['total_max_mem'].append(process_max_mem)

# Print details for each valid character cluster
print("\nCharacter Details (Left to Right):")
for cluster in final_clusters:
    size = cluster['size']
    min_x, min_y, max_x, max_y = cluster['bounding_box']
    start_y, start_x = cluster['start_pixel']
    char_width = max_x - min_x + 1
    char_index = cluster['char_index']
    is_sub_character = cluster['is_sub_character']
    modified_z_score = modified_z_scores[clusters.index(cluster)] if modified_z_scores and cluster in clusters else 0
    print(f"Character {char_index}:")
    print(f"  Size: {size} pixels")
    print(f"  Bounding Box: (min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y})")
    print(f"  Starting Pixel: ({start_y}, {start_x})")
    print(f"  Width: {char_width} pixels")
    print(f"  Modified Z-Score: {modified_z_score:.2f}")
    print(f"  Is Sub-Character: {is_sub_character}")
    if is_sub_character:
        print(f"  ALERT: Character {char_index} is a divided sub-character")
print(f"\nTotal number of character clusters: {len(final_clusters)}")
print(f"Average size of character clusters: {average_size:.2f} pixels")

# Display images with bounding boxes
def display_results(binary_inverted, cleaned_image, cropped_image, current_image, binary_inverted_current_image, final_image, final_image_second_pass, black_clusters, largest_black_cluster, clusters, largest_cluster, final_clusters, width, height):
    plt.figure(figsize=(20, 14))
    plt.subplot(2, 4, 1)
    plt.imshow(binary_inverted, cmap="gray")
    plt.title("Binary inverted Image at First")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(cleaned_image, cmap="gray")
    plt.title("Cleaned Image (After scan_and_clean)")
    if black_clusters and largest_black_cluster:
        min_y, max_y = largest_black_cluster['bounding_box'][1], largest_black_cluster['bounding_box'][3]
        min_x, max_x = 0, width - 1
        width = max_x - min_x
        height = max_y - min_y
        rect_largest = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=2, edgecolor='green', facecolor='none', label='Region')
        plt.gca().add_patch(rect_largest)
        plt.text(min_x, min_y - 5, 'Region', color='green', fontsize=8)
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(cropped_image, cmap="gray")
    plt.title("Irregularly Cropped Largest Cluster")
    if clusters and largest_cluster:
        min_y, max_y = largest_cluster['bounding_box'][1], largest_cluster['bounding_box'][3]
        min_x, max_x = 0, width - 1
        width = max_x - min_x
        height = max_y - min_y
        rect_largest = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=2, edgecolor='blue', facecolor='none', label='Region'
        )
        plt.gca().add_patch(rect_largest)
        plt.text(min_x, min_y - 5, 'Region', color='blue', fontsize=8)
        plt.legend()
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(current_image, cmap="gray")
    plt.title("Denoised Image (First Pass)")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(binary_inverted_current_image, cmap="gray")
    plt.title("Inverted Denoised Image")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(final_image, cmap="gray")
    plt.title("Final Image (After convert_black_to_white)")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(final_image_second_pass, cmap='gray')
    plt.title("Final Image (Second Pass - Black Noise Removal)")
    for cluster in final_clusters:
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        char_index = cluster['char_index']
        is_sub_character = cluster['is_sub_character']
        width = max_x - min_x
        height = max_y - min_y
        edgecolor = 'red' if is_sub_character else 'green'
        rect = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=1, edgecolor=edgecolor, facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(min_x, min_y - 5, f'Char {char_index}', color=edgecolor, fontsize=8)
    plt.axis("off")
    plt.savefig('image_processing_results.png')

    # Create a separate figure for the labeled image
    plt.figure(figsize=(10, 7))
    plt.imshow(final_image_second_pass, cmap='gray')
    plt.title("Labeled Image with Bounding Boxes")
    for cluster in final_clusters:
        min_x, min_y, max_x, max_y = cluster['bounding_box']
        char_index = cluster['char_index']
        is_sub_character = cluster['is_sub_character']
        width = max_x - min_x
        height = max_y - min_y
        edgecolor = 'red' if is_sub_character else 'green'
        rect = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=1, edgecolor=edgecolor, facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(min_x, min_y - 5, f'Char {char_index}', color=edgecolor, fontsize=8)
    if len(final_clusters) >= 4:
        centers = [cluster['center'] for cluster in final_clusters]
        x_centers, y_centers = zip(*centers)
        tck, u = splprep([x_centers, y_centers], s=10)
        u_fine = np.linspace(0, 1, 50)
        x_spline, y_spline = splev(u_fine, tck)
        plt.plot(x_spline, y_spline, 'b-', linewidth=1, label='spline')
        plt.legend()
    plt.axis("off")
    plt.savefig('labeled_bounding_boxes_standalone.png')
    plt.close()

    plt.tight_layout()
    plt.show()

display_results(binary_inverted, cleaned_image, cropped_image, current_image, binary_inverted_current_image, final_image, final_image_second_pass, black_clusters, largest_black_cluster, clusters, largest_cluster, final_clusters, width, height)

# Log overall performance metrics
logger.info("\nOverall Performance Metrics")
logger.info(f"Total Execution Time: {performance_metrics['total_time']:.2f} seconds")
logger.info(f"Average CPU Usage (Overall): {np.mean(performance_metrics['total_avg_cpu']):.2f}%")
logger.info(f"Peak CPU Usage (Overall): {np.max(performance_metrics['total_max_cpu']):.2f}%")
logger.info(f"Peak Memory Usage (Overall): {np.max(performance_metrics['total_max_mem']):.2f} MB")

# Save performance metrics to a file
with open('performance_metrics.txt', 'w') as f:
    f.write("Performance Metrics Report\n")
    f.write("=========================\n\n")
    f.write("Stage-wise Metrics:\n")
    for stage, metrics in performance_metrics['stages'].items():
        f.write(f"Stage: {stage}\n")
        f.write(f"  Execution Time: {metrics['time']:.2f} seconds\n")
        f.write(f"  Average CPU Usage: {metrics['avg_cpu']:.2f}%\n")
        f.write(f"  Peak CPU Usage: {metrics['max_cpu']:.2f}%\n")
        f.write(f"  Peak Memory Usage: {metrics['max_mem']:.2f} MB\n")
        f.write("\n")
    f.write("Overall Metrics:\n")
    f.write(f"Total Execution Time: {performance_metrics['total_time']:.2f} seconds\n")
    f.write(f"Average CPU Usage (Overall): {np.mean(performance_metrics['total_avg_cpu']):.2f}%\n")
    f.write(f"Peak CPU Usage (Overall): {np.max(performance_metrics['total_max_cpu']):.2f}%\n")
    f.write(f"Peak Memory Usage (Overall): {np.max(performance_metrics['total_max_mem']):.2f} MB\n")