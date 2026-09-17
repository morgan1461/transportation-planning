import pandas as pd
from pathlib import Path
import zipfile
import time
import os
import re
from datetime import date, datetime, timedelta
from calendar import monthrange

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

def busstate_processing(year, month, root_dir = "K:/AP/TTM/", current_dir = None, overlap_days = 2):
    '''
    Process all busstate zip files in the data directory and save cleaned csv files to the repo directory

    Args:
        year (str): year of busstate data to process (e.g. "25") 
            NOTE: MUST BE IN 2 DIGIT FORMAT
        month (str): month of busstate data to process (e.g. "10") 
            NOTE: MUST BE IN 2 DIGIT FORMAT
        root_dir (str): root directory path - SHOULD ONLY NEED CHANGED IF ON UNIX SYSTEM
        current_dir (str | Path | None): optional repository root override
        overlap_days (int): number of days before/after month boundary to include
            when selecting files by filename date. Helps capture cross-month spillover.

    Returns:
        None - saves cleaned csv files to repo directory "./busstate_cleaned"
    '''
    # Validate year and month inputs
    if not isinstance(year, str) or len(year) != 2 or not year.isdigit():
        raise ValueError("Year must be a 2-digit string (e.g., '25')")
    
    if not isinstance(month, str) or len(month) != 2 or not month.isdigit():
        raise ValueError("Month must be a 2-digit string (e.g., '09')")
    
    if not (1 <= int(month) <= 12):
        raise ValueError("Month must be between 01 and 12")

    if not isinstance(overlap_days, int) or overlap_days < 0:
        raise ValueError("overlap_days must be a non-negative integer")

    # convert year to full year format for later use in filtering and saving
    full_year = int("20" + year)

    #timer
    start = time.perf_counter()
    print(f"Starting busstate processing for {month}/{year}...")

    data_dir = Path(root_dir) / "Data" / "APC Data" # contains zipped raw busstate txt files
    repo_root = Path(current_dir) if current_dir is not None else REPO_ROOT
    repo_dir = repo_root / 'transportation-planning'

    # Select files by a date window around the target month so spillover data is included.
    month_num = int(month)
    month_start = date(full_year, month_num, 1)
    month_end = date(full_year, month_num, monthrange(full_year, month_num)[1])
    selection_start = month_start - timedelta(days=overlap_days)
    selection_end = month_end + timedelta(days=overlap_days)

    # list comprehension to get all busstate files within selection window - will be zipped
    # takes about 30 seconds to run for 1 month of data
    busstates = [
        path
        for path in data_dir.iterdir()
        if (file_date := extract_busstate_file_date(path.name)) is not None
        and selection_start <= file_date <= selection_end
    ]

    #print(busstates[:5]) 
    print(
        f"Found {len(busstates)} busstate files for {month}/{year} "
        f"(including +/- {overlap_days} day overlap) in '{data_dir}'"
    )

    # empty dataframe to hold busstate data
    df = pd.DataFrame()

    print(f"Unzipping and processing busstate files for {month}/{year}...")
    # loop through each busstate file, unzip, process, and add to main dataframe
    # definitely a way to make this more performant - will revist later
    for busstate in busstates:
        raw_data = unzip_busstate_to_df(busstate)
        cleaned_data = process_busstate(raw_data)
        df = pd.concat([df, cleaned_data], ignore_index=True)

    print(f"Combined dataframe has {len(df)} records for {month}/{year}")
    # print(df.head())
    # print(df.info())
    
    # print(f"Full year is {full_year} and type is {type(full_year)}")

    # sort and save
    sort_and_save(df, repo_dir, full_year)

    end = time.perf_counter()
    print(f"Finished processing busstate data for {month}/{year} in {end - start:.2f} seconds. Cleaned files saved to '{repo_dir}'")

def extract_busstate_file_date(filename):
    """
    Parse YYMMDD date from busstate filename suffix like '*250401.txt.zip'.

    Returns:
        datetime.date | None: parsed date or None if filename does not match.
    """
    match = re.search(r"(\d{6})\.txt\.zip$", filename)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%y%m%d").date()
    except ValueError:
        return None

def unzip_busstate_to_df(zip_path):
    '''
    Unzip A SINGLE busstate zip file and return a pandas dataframe
    NOTE: This assumes ONE file in the zip, which is the case for all busstate zip files. If this ever changes, we will need to modify this function to handle multiple files in the zip.
    NOTE: The first row of the csv file contains data types, so we will drop that row and reset the index.

    Parameters:
        zip_path (str): Path to the busstate zip file

    Returns:
        pandas dataframe containing all busstate data from the zip file specified 

    
    '''
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Key assumption is there is only ONE file in the zip 
        file_name = zip_ref.namelist()[0]
        with zip_ref.open(file_name) as f:
            df = (pd.read_csv(f, sep=',', low_memory=False)) # Read all columns as string to avoid dtype issues - we will convert to correct dtypes later
            df = df.drop(df.index[0]).reset_index(drop=True) # DROP first row with "data types"
    return df

def process_busstate(df):
    '''
    Clean and process the busstate dataframe by:
    - Parsing datetime columns
    - Splitting EVENT_TIME into DATE + TIME
    - Keeping only time portion for other datetime columns
    - Converting numeric columns to numeric dtype
    - Creating BOARDINGS + ALIGHTINGS columns
    - Filtering EVENT_TYPE < 17
    - Selecting relevant columns
    - Filtering missing RUN_ID / DEST_SIGN_ROUTE_TEXT
    - Sorting by DATE + EVENT_TIME

    Parameters:
        df (pandas dataframe): Raw unzipped busstate dataframe

    Returns:
        pandas dataframe: Cleaned and processed busstate dataframe
    '''
    #print("BEFORE: ", df["EVENT_TIME"].head(10))

    # cols in YYMMDDhhmmss format
    datetime_cols = [
        "EVENT_TIME",
        "ENTER_STOP_WINDOW_TIME",
        "EXIT_STOP_WINDOW_TIME",
        "TRIP_START_TIME",
        "DEPARTURE_TIME"
    ]

    # convert to datetime,
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], format="%y%m%d%H%M%S", errors="coerce")

    #print("AFTER: ", df["EVENT_TIME"].head(10))

    df["DATE"] = df["EVENT_TIME"].dt.date
    df["EVENT_TIME"] = df["EVENT_TIME"].dt.time

    # For the other datetime columns, we only care about the time portion, so we will keep only the time portion
    for col in datetime_cols[1:]:
        df[col] = df[col].dt.time

    # convert numerics from str
    numeric_cols = [
        "STOP_BACK_DOOR_ENTRY",
        "STOP_FRONT_DOOR_ENTRY",
        "STOP_FRONT_DOOR_EXIT",
        "STOP_BACK_DOOR_EXIT",
        "EVENT_TYPE"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # baordings and alightnings
    df["BOARDINGS"] = df["STOP_BACK_DOOR_ENTRY"] + df["STOP_FRONT_DOOR_ENTRY"]
    df["ALIGHTINGS"] = df["STOP_FRONT_DOOR_EXIT"] + df["STOP_BACK_DOOR_EXIT"]

    ##################################
    # not sure why this is BUT is in legacy R code so keeping for now - need to investigate later
    # comment stats this is extraneous event types that we don't care about -  will investigate later if we need to keep any of these event types
    df = df[df["EVENT_TYPE"] < 17]

    df = df[[
        "DATE", "BUS_ID", "RUN_ID", "DEST_SIGN_ROUTE_TEXT", "BLOCK_ID", "TRIP_ID",
        "ROUTE_ID", "STOP_SEQUENCE", "LATITUDE", "LONGITUDE",
        "HEADING", "OPERATOR_ID", "ODOMETER_DISTANCE", "TIMEPOINT_ID",
        "EVENT_TYPE", "EVENT_TIME", "BOARDINGS", "ALIGHTINGS", "PASSENGER_LOAD",
        "TRIP_START_TIME", "DEPARTURE_TIME", "ENTER_STOP_WINDOW_TIME", "EXIT_STOP_WINDOW_TIME"
    ]]

    # filter out missing run id and dest signs 
    df = df[(df["RUN_ID"].notna()) | (df["DEST_SIGN_ROUTE_TEXT"].notna())]

    # arrange based on date and time - also in legacy R code
    df = df.sort_values(by=["DATE", "EVENT_TIME"])

    return df

# Replicate the 'Sort and save function' save it into the repo directory
def sort_and_save(df, output_dir, year):
    """
    Save monthly busstate files by merging with existing outputs and de-duplicating.

    Parameters:
        df (pd.DataFrame): cleaned busstate dataframe with 'DATE' column
        output_dir (str): folder path to save files - defaults to repo_dir
        year (int): year to filter - defaults to 2025 for future data, can change if needed
    """
    # hardcoded month names
    month_names = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

    # ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ensure date column is datetime
    df['DATE'] = pd.to_datetime(df['DATE'])

    for month in range(1, 13):
        # Filter data for month/year
        month_df = df[(df['DATE'].dt.year == year) & (df['DATE'].dt.month == month)]

        # if no data for a month, skip saving and print message
        if month_df.empty:
            print(f"No data for month {month_names[month-1]} {year}")
            continue

        # filepath
        filepath = Path(output_dir) / f"{year}-{month_names[month-1]}-busstate.csv"

        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            existing_df['DATE'] = pd.to_datetime(existing_df['DATE'], errors='coerce')

            # Merge existing + new and remove exact duplicate rows across shared columns.
            common_cols = [col for col in month_df.columns if col in existing_df.columns]
            combined_df = pd.concat([existing_df, month_df], ignore_index=True)
            before_dedupe = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=common_cols, keep='last')
            deduped = before_dedupe - len(combined_df)
            month_df = combined_df
            print(f"Merged with existing file and removed {deduped} duplicate records for {month_names[month-1]} {year}")

        # sort by date and event time
        month_df = month_df.sort_values(by=['DATE', 'EVENT_TIME'])

        # save merged/deduped output
        month_df.to_csv(filepath, index=False)

        # output message
        print(f"Saved {len(month_df)} records for {month_names[month-1]} {year} to '{filepath}'")

# manual control
if __name__ == "__main__":
    month = str(input("Enter the 2 digit month: "))
    year = str(input("Enter the 2 digit year: "))
    busstate_processing(month=month, year=year)