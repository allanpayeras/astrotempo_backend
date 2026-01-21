import os

from pathlib import Path
from goes2go.data import goes_timerange, goes_latest
from datetime import datetime, timedelta, timezone


GOES_DIR = "./goes_data"


def remove_empty_dirs(root_path):
    """
    Recursively remove all empty subdirectories under root_path.
    The root_path itself is not removed, even if it becomes empty.
    """
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Skip the root directory itself
        if dirpath == root_path:
            continue

        try:
            # Directory is empty if no files and no subdirectories remain
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")
        except OSError as e:
            # Covers permission errors or race conditions
            print(f"Could not remove {dirpath}: {e}")


def clean_up_goes_data(goes_dir: str, start_time: datetime, end_time: datetime):
    print(f"Cleaning GOES data in directory: {goes_dir}")

    # Get list of GOES files in the directory
    goes_path = Path(goes_dir)
    goes_files = sorted(list(goes_path.glob("**/*.nc")))

    # loop through files and verify if start observation time is within the desired range
    for file in goes_files:
        obs_time = file.stem.split("_s")[1].split("_e")[0][
            :-1
        ]  # Extract start time from filename
        obs_datetime = datetime.strptime(obs_time, "%Y%j%H%M%S").replace(
            tzinfo=timezone.utc
        )
        if not (start_time <= obs_datetime <= end_time):
            print(f"Deleting file: {file}")
            file.unlink()  # Delete the file

    remove_empty_dirs(goes_dir)


def main():
    if not Path(GOES_DIR).exists():
        Path(GOES_DIR).mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=timezone.utc)
    start_time = now - timedelta(hours=2)
    print(f"Downloading GOES data from {start_time} to {now}")

    _ = goes_timerange(
        start=start_time.replace(tzinfo=None),
        end=now.replace(tzinfo=None),
        satellite="goes19",
        product="ABI",
        domain="F",
        return_as="filelist",
        save_dir=GOES_DIR,
        max_cpus=6,
    )

    clean_up_goes_data(GOES_DIR, start_time, now)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"GOES data retrieval completed in {end_time - start_time}")
