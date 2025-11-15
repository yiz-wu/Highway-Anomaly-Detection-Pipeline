import os
import glob
import csv
import json
import argparse
import re

def interpolate(value, x_points, y_points):
    """
    Linear interpolation for a single value.
    """
    if value <= x_points[0]:
        return y_points[0]
    if value >= x_points[-1]:
        return y_points[-1]

    for i in range(len(x_points) - 1):
        if x_points[i] <= value <= x_points[i+1]:
            x0, x1 = x_points[i], x_points[i+1]
            y0, y1 = y_points[i], y_points[i+1]
            if x1 == x0:
                return y0
            t = (value - x0) / float(x1 - x0)
            return y0 + t * (y1 - y0)

    return y_points[-1]

def load_gps_csv(gps_path):
    with open(gps_path, "r") as f:
        reader = csv.DictReader(f)
        gps_data = [row for row in reader]

    gps_sequences = [int(row["sequence"]) for row in gps_data]
    gps_columns = {col: [float(row[col]) for row in gps_data]
                   for col in ["x", "y", "z", "heading", "roll", "pitch"]}
    return gps_sequences, gps_columns

def load_gps_json(gps_path):
    with open(gps_path, "r") as f:
        gps_data = json.load(f)  # list of dicts expected

    gps_sequences = [int(row["sequence"]) for row in gps_data]
    gps_columns = {col: [float(row[col]) for row in gps_data]
                   for col in ["x", "y", "z", "heading", "roll", "pitch"]}
    return gps_sequences, gps_columns

def interpolate_gps_to_images(gps_path, images_folder, output_path, img_ext="png", seq_regex=None):
    """
    Interpolate GPS data to match image frames.

    Args:
        gps_path: path to GPS .csv or .json file
        images_folder: folder containing image files
        output_path: where to save interpolated GPS CSV
        img_ext: image extension (default: png)
        seq_regex: optional regex to extract sequence or timestamp from image name.
                   The first capturing group will be converted to int.
                   Example: r"(\\d{12})_Rectified_\\d+_Cam5"  → extracts 12-digit timestamp
    """
        
    # --- Load GPS data depending on extension ---
    ext = os.path.splitext(gps_path)[1].lower()
    if ext == ".csv":
        gps_sequences, gps_columns = load_gps_csv(gps_path)
    elif ext == ".json":
        gps_sequences, gps_columns = load_gps_json(gps_path)
    else:
        raise ValueError("Unsupported GPS file format. Use .csv or .json")

    # --- Collect image sequence numbers ---
    image_files = sorted(glob.glob(os.path.join(images_folder, f"*.{img_ext}")))
    image_sequences = []
    print(seq_regex)
    pattern = re.compile(seq_regex) if seq_regex else None

    for img in image_files:
        fname = os.path.splitext(os.path.basename(img))[0]
        seq = None

        if pattern is not None:
            match = pattern.search(fname)
            if match:
                try:
                    seq = int(match.group(1))
                except ValueError:
                    continue
        else:
            # fallback: assume filename itself is an integer
            try:
                seq = int(fname)
            except ValueError:
                continue

        if seq is not None:
            image_sequences.append(seq)

    if not image_sequences:
        raise ValueError("No valid image filenames with numeric or regex-matched sequence found.")

    # --- Interpolate or take exact GPS ---
    results = []
    gps_lookup = {seq: i for i, seq in enumerate(gps_sequences)}

    for seq in image_sequences:
        row = {"sequence": seq}
        if seq in gps_lookup:
            idx = gps_lookup[seq]
            for col in gps_columns:
                row[col] = gps_columns[col][idx]
        else:
            for col in gps_columns:
                row[col] = interpolate(seq, gps_sequences, gps_columns[col])
        results.append(row)

    # --- Write results to CSV ---
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sequence", "x", "y", "z", "heading", "roll", "pitch"]
        writer.writerow(header)
        for row in results:
            writer.writerow([row["sequence"], row["x"], row["y"], row["z"],
                             row["heading"], row["roll"], row["pitch"]])

    print(f"Interpolated GPS data written to {output_path}")

# ---------------- Example usage ----------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Interpolate GPS data to image sequences")
    parser.add_argument(
        "-i",
        dest="input_path",
        required=True,
        help="Path to JSON preset configuration file"
    )
    args = parser.parse_args()

    json_path = str(args.input_path)
    if not os.path.exists(json_path):
        print(f"Unable to locate file {json_path}")
        exit(1)
    with open(json_path, "r") as fp:
        preset = json.load(fp)


    dataset_cfg = preset.get("dataset", {})
    if not dataset_cfg:
        raise KeyError("Missing 'dataset' section in preset file.")

    car_gps_path = dataset_cfg.get("car_gps")
    images_folder = dataset_cfg.get("images")
    img_format = dataset_cfg.get("img_format")
    name_regex = dataset_cfg.get("name_regex")

    if not all([car_gps_path, images_folder, img_format]):
        raise ValueError("Configuration file must define 'car_gps', 'images', and 'img_format' under 'dataset'.")

    # --- Build output filename automatically ---
    base, _ = os.path.splitext(os.path.basename(car_gps_path))
    output_file = f"interpolated_{base}.csv"
    output_path = os.path.join(os.path.dirname(car_gps_path), output_file)

    # --- Run interpolation ---
    interpolate_gps_to_images(
        gps_path=car_gps_path,
        images_folder=images_folder,
        output_path=output_path,
        img_ext=img_format,
        seq_regex=name_regex
    )
