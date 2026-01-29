import os
import ast
import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Dict
# from datetime import datetime
import calendar
import multiprocessing as mp
from functools import partial
import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
import gc
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import binom
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("footfall_processing.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def keep_only_observed(srs_holiday):
    # Create a dictionary to store base holiday name -> dates
    holiday_dict = {}
    holidays = srs_holiday.to_dict()

    # First pass: Group dates by base holiday name
    for date, name in holidays.items():
        base_name = name.replace(" (observed)", "")
        if base_name not in holiday_dict:
            holiday_dict[base_name] = []
        holiday_dict[base_name].append((date, name))

    # Second pass: Keep only observed date if available, otherwise keep the original
    filtered_holidays = {}
    for base_name, date_names in holiday_dict.items():
        if len(date_names) == 1:
            # Only one date for this holiday
            date, name = date_names[0]
            filtered_holidays[date] = name
        else:
            # Multiple dates - check for observed
            observed_found = False
            for date, name in date_names:
                if "(observed)" in name:
                    filtered_holidays[date] = name
                    observed_found = True
                    break
            
            # If no observed found, use the original date
            if not observed_found:
                # Find the entry without 'observed' in the name
                for date, name in date_names:
                    if "(observed)" not in name:
                        filtered_holidays[date] = name
                        break

    # Convert back to Series
    result = pd.Series(filtered_holidays)
    return result

def get_optimal_workers():
    """Determine optimal number of workers based on available RAM and CPU cores."""
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    cpu_count = multiprocessing.cpu_count()
    
    # Reserve 2GB for the main process and OS
    usable_ram = max(1, available_ram - 2)
    
    # Estimate each worker might need ~1GB (adjust based on your data)
    worker_count = min(cpu_count - 1, int(usable_ram / 1))
    
    # Always have at least 1 worker, at most CPU count - 1
    return max(1, min(worker_count, cpu_count - 1))

def write_common_placekeys(path: str, year: int, n_processes: int = None) -> Set[str]:
    """
    Identify and save placekeys that are common across all months for a given year.
    Uses parallel processing to handle large files efficiently.
    
    Args:
        path: Base directory containing the data files
        year: Year to analyze
        n_processes: Number of parallel processes to use (defaults to CPU count)
        
    Returns:
        Set of placekeys common across all months
    """
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    logger.info(f"Using {n_processes} parallel processes")
    
    # Ensure output directory exists
    output_dir = os.path.join(path, 'placekeys_census_tracts')
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store placekeys for each month
    monthly_placekeys: Dict[tuple, Set[str]] = {}
    
    # Process each month
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        pattern = f"{year}-{month_str}-01"
        matching_files = glob.glob(os.path.join(path, f'*{pattern}.csv'))
        
        if not matching_files:
            logger.warning(f"No files found for {year}-{month_str}")
            continue
        
        logger.info(f"Processing {len(matching_files)} files for {year}-{month_str}")
        
        # Process files in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_file, matching_files)
        
        # Combine results from all files
        all_placekeys = set()
        for file_placekeys in results:
            all_placekeys.update(file_placekeys)
        
        monthly_placekeys[(year, month)] = all_placekeys
        logger.info(f"Found {len(all_placekeys)} unique placekeys for {year}-{month_str}")
    
    # Find common placekeys across all months
    if not monthly_placekeys:
        logger.warning(f"No data found for year {year}")
        common_placekeys = set()
    else:
        # Start with the first month's placekeys
        months = list(monthly_placekeys.keys())
        common_placekeys = monthly_placekeys[months[0]].copy()
        
        # Find intersection with all other months
        for month_key in months[1:]:
            common_placekeys.intersection_update(monthly_placekeys[month_key])
    
    logger.info(f"Found {len(common_placekeys)} placekeys common across all months in {year}")
    
    # Write results to file
    output_file = os.path.join(output_dir, f"placekeys_{year}.txt")
    
    with open(output_file, 'w') as file:
        if not common_placekeys:
            file.write("No common placekeys found across all months")
        else:
            file.write('\n'.join(common_placekeys))
    
    logger.info(f"Wrote common placekeys to {output_file}")
    
    return common_placekeys

def process_file(file_path):
    """
    Process a single data file to extract unique placekeys.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Set of unique placekeys from this file
    """
    try:
        # Memory-efficient reading with chunks
        chunk_size = 100000  # Adjust based on your RAM
        placekeys = set()
        
        # Process file in chunks to avoid loading everything into memory
        for chunk in pd.read_csv(
            file_path, 
            usecols=["PLACEKEY", "ISO_COUNTRY_CODE", "VISITS_BY_DAY", "POI_CBG"],
            dtype={"PLACEKEY": str},
            chunksize=chunk_size
        ):
            filtered = chunk[(chunk["ISO_COUNTRY_CODE"] == "US") & 
                            (~chunk["VISITS_BY_DAY"].isna()) & (~chunk["POI_CBG"].isna()) &
                            (~chunk["PLACEKEY"].isna())]
            # chunk.dropna(subset="POI_CBG", inplace=True)
            filtered["POI_CBG"] = filtered["POI_CBG"].apply(lambda x: str(int(x)).zfill(12))
            filtered["POI_CBG_n"] = filtered["POI_CBG"].apply(lambda x: len(x))
            assert all(filtered["POI_CBG_n"] == 12)
            filtered["census_tract"] = filtered["POI_CBG"].apply(lambda x: x[:11])
            filtered["plid"] = filtered["PLACEKEY"].str.cat(filtered["census_tract"], sep="@")
            
            
            # Add unique placekeys from this chunk
            if not filtered.empty:
                placekeys.update(filtered["plid"].unique())
                
        return placekeys
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return set()
    

def extract_months(year, path, placekeys):
    """Process all months for the given year."""
    logger.info(f"Starting footfall data processing for year {year}")
    
    # Check if placekeys is a file path or a list
    if isinstance(placekeys, str) and os.path.isfile(placekeys):
        logger.info(f"Loading placekeys from file: {placekeys}")
        with open(placekeys, 'r') as f:
            placekeys = [line.strip() for line in f]
    
    logger.info(f"Processing with {len(placekeys)} placekeys")
    
    # Process each month sequentially (we're already parallelizing file processing)
    output_dir = os.path.join(path, "census_tract_aggregation", f"{year}")
    for month in range(1, 13):        
        # Save results
        month_str = str(month).zfill(2)
        output_file = os.path.join(output_dir, f"footfall_ct_{year}_{month_str}.csv")
        if os.path.exists(output_file):
            print(f"{output_file} exists")
            continue
        process_month(year, month, path, placekeys)
        
        # Force garbage collection after each month
        gc.collect()
    
    logger.info("Completed processing all months")

def process_month(year, month, path, placekeys):
    """Process all files for a given month."""
    try:
        month_str = str(month).zfill(2)
        pattern = f"{year}-{month_str}-01"
        matching_files = glob.glob(os.path.join(path, f'*{pattern}.csv'))
        
        if not matching_files:
            logger.warning(f"No files found for {year}-{month_str}")
            return
            
        logger.info(f"Processing month {year}-{month_str}, found {len(matching_files)} files")
        
        # Determine optimal number of workers
        workers = get_optimal_workers()
        logger.info(f"Using {workers} parallel workers")
        
        result_dfs = []
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(process_file_monthly_extraction, file_path, placekeys, year, month): file_path 
                for file_path in matching_files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        result_dfs.append(df)
                        logger.info(f"Successfully processed {file_path}")
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {str(e)}")
        
        if not result_dfs:
            logger.warning(f"No valid data found for month {year}-{month_str}")
            return
            
        # Concatenate all dataframes
        logger.info(f"Concatenating {len(result_dfs)} dataframes for month {year}-{month_str}")
        df_month = pd.concat(result_dfs, axis=0)
        df_month.drop_duplicates("plid", inplace=True)
        
        # Free memory
        del result_dfs
        gc.collect()
        
        # Aggregate by postal code
        logger.info(f"Aggregating data by postal code for month {year}-{month_str}")
        df_month_zip = df_month.groupby('census_tract').agg({
            'visits_by_day': combine_spend_lists,
            'census_tract': 'count'
        }).rename(columns={'census_tract': 'row_count'})
        
        # Free more memory
        del df_month
        gc.collect()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(path, "census_tract_aggregation", f"{year}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        output_file = os.path.join(output_dir, f"footfall_census_tract_{year}_{month_str}.csv")
        df_month_zip.to_csv(output_file)
        logger.info(f"Successfully wrote data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing month {year}-{month_str}: {str(e)}")

def process_file_monthly_extraction(file_path, placekeys, year, month):
    """
    Process a single file and return the processed dataframe.
    Validates and imputes visit data to match the correct number of days in the month.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    placekeys : list
        List of placekeys to filter by
    year : int
        Year of the data
    month : int
        Month of the data (1-12)
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Use chunksize for memory efficiency with large files
        chunks = []
        chunk_size = 100000  # Adjust based on your file structure
        
        # Only read the columns we need
        cols = ["PLACEKEY", "POSTAL_CODE", "ISO_COUNTRY_CODE", "VISITS_BY_DAY", "POI_CBG"]
        dtype = {"POSTAL_CODE": str}
        
        # Get expected number of days in this month
        import calendar
        expected_days = calendar.monthrange(year, month)[1]
        logger.info(f"Expected {expected_days} days for {year}-{month:02d}")
        
        # Process file in chunks
        for chunk in pd.read_csv(file_path, usecols=cols, dtype=dtype, chunksize=chunk_size):
            # Filter for US records and specific placekeys
            chunk = chunk[chunk["ISO_COUNTRY_CODE"] == "US"]
            chunk.dropna(subset=["VISITS_BY_DAY"], inplace=True)
            chunk.dropna(subset=["POI_CBG"], inplace=True)
            chunk["POI_CBG"] = chunk["POI_CBG"].apply(lambda x: str(int(x)).zfill(12))
            # chunk["POI_CBG_n"] = chunk["POI_CBG"].apply(lambda x: len(x))
            # assert all(chunk["POI_CBG_n"] == 12)
            chunk["census_tract"] = chunk["POI_CBG"].apply(lambda x: x[:11])

            chunk["plid"] = chunk["PLACEKEY"].str.cat(chunk["census_tract"], sep="@")
            chunk = chunk[chunk["plid"].isin(placekeys)]
            
            if not chunk.empty:
                # Process VISITS_BY_DAY with validation and imputation
                def validate_and_impute_days(visit_str):
                    try:
                        visits_list = ast.literal_eval(visit_str)
                        
                        # Check if the list has the correct number of days
                        if len(visits_list) == expected_days:
                            return visits_list
                        
                        # If not, impute missing days with the median value
                        if visits_list:
                            median_value = np.median(visits_list)
                        else:
                            median_value = 0
                        
                        if len(visits_list) < expected_days:
                            # Case: Missing days - add median values
                            logger.debug(f"Found {len(visits_list)} days instead of {expected_days} - imputing with median {median_value}")
                            return visits_list + [median_value] * (expected_days - len(visits_list))
                        else:
                            # Case: Too many days - truncate
                            logger.debug(f"Found {len(visits_list)} days instead of {expected_days} - truncating")
                            return visits_list[:expected_days]
                    except Exception as e:
                        logger.warning(f"Error processing visits data: {str(e)}")
                        return [0] * expected_days  # Return zeros if parsing fails
                
                # Apply the validation function
                chunk["visits_by_day"] = chunk["VISITS_BY_DAY"].apply(validate_and_impute_days)
                chunk.drop(labels="VISITS_BY_DAY", axis=1, inplace=True)
                chunks.append(chunk)
        
        # If no valid chunks, return empty DataFrame with correct columns
        if not chunks:
            return pd.DataFrame(columns=["PLACEKEY", "plid", "visits_by_day"])
        
        # Combine all chunks
        result = pd.concat(chunks, axis=0)
        
        # Log summary of imputation
        total_rows = len(result)
        logger.info(f"Processed {total_rows} rows from {file_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return pd.DataFrame(columns=["PLACEKEY", "POSTAL_CODE", "visits_by_day"])

def combine_spend_lists(series):
    """Combine multiple visit lists into a single aggregated list."""
    # Assuming this function combines lists element-wise
    # If series is empty, return an empty list
    if len(series) == 0:
        return []
    
    # Initialize with first list
    result = series.iloc[0].copy()
    
    # Add remaining lists element-wise
    for i in range(1, len(series)):
        visits = series.iloc[i]
        for j in range(min(len(result), len(visits))):
            result[j] += visits[j]
    
    return result

def write_expanded_footfall(path, year):
    pc_list = []
    for m in range(1,13):
        month = str(m).zfill(2)
        fname = f"footfall_census_tract_{year}_{month}.csv"
        df_ff = pd.read_csv(os.path.join(path, "census_tract_aggregation", f"{year}", fname), dtype={"census_tract":str})
        postal_list = df_ff["census_tract"].unique().tolist()
        pc_list.append(postal_list)
        visits = np.array(df_ff["visits_by_day"].apply(lambda x: ast.literal_eval(x)).tolist())
        ndays = visits.shape[1]
        date_cols = pd.date_range(f"{year}-{month}-01", f"{year}-{month}-{ndays}")
        visits_df = pd.DataFrame(visits, columns=date_cols)
        df_month = pd.concat([df_ff[["census_tract", "row_count"]], visits_df], axis=1)
        df_month.set_index("census_tract", inplace=True)
        df_month.sort_index(inplace=True)
        df_month.to_csv(os.path.join(path, "census_tract_aggregation", f"{year}", f"footfall_visits_expanded_{year}_{month}.csv"))

def STL_resid_analysis(stl, start_event, end_event, event_type, pad_days=21):
    resid = stl.resid
    spend_srs = stl.observed
    resid_std = (resid -resid.mean())/resid.std()
    laplace_loc, laplace_scale = stats.laplace.fit(resid_std)
    norm_loc, norm_scale = stats.norm.fit(resid_std)
    ks_laplace, p_laplace = stats.kstest(resid_std, 'laplace', args=(laplace_loc, laplace_scale))
    ks_normal, p_normal = stats.kstest(resid_std, 'norm', args=(norm_loc, norm_scale))
    event_srs = resid_std[(resid_std.index >= pd.to_datetime(start_event)) & (resid_std.index <= pd.to_datetime(end_event))]
    n_neg = sum(event_srs < 0)
    event_before = resid_std[resid_std.index < pd.to_datetime(start_event)]
    event_after = resid_std[resid_std.index > pd.to_datetime(end_event)]
    cdf_values = stats.laplace.cdf(event_srs, loc=laplace_loc, scale=laplace_scale)
    cdf_values_r = 1 - stats.laplace.cdf(event_srs, loc=laplace_loc, scale=laplace_scale)
    min_p = min(cdf_values)
    min_p_r = min(cdf_values_r)
    prep_date = pd.to_datetime(start_event, format="%Y-%m-%d")-pd.Timedelta(days=1)
    rebound_date = pd.to_datetime(end_event, format="%Y-%m-%d")+pd.Timedelta(days=1)
    counterfactual_loss = (stl.trend + stl.seasonal - stl.observed)[(resid_std.index >= pd.to_datetime(start_event)) & (resid_std.index <= pd.to_datetime(end_event))].tolist()

    if prep_date in resid_std:
        prep_resid = resid_std[prep_date]
    else:
        prep_resid = np.nan
    
    if rebound_date in resid_std:
        rebound_resid = resid_std[rebound_date]
    else:
        rebound_resid = np.nan

    event_before_list = event_before.tolist()
    event_after_list = event_after.tolist()
    event_before_list = [0] * (pad_days - len(event_before_list)) + event_before_list if len(event_before_list) < pad_days else event_before_list
    event_after_list = event_after_list + [0] * (pad_days - len(event_after_list)) if len(event_after_list) < pad_days else event_after_list

    return counterfactual_loss, [sum(cdf_values < 0.05), sum(cdf_values_r < 0.05), event_srs.shape[0], event_type, spend_srs.mean(), spend_srs.std(), 
                            prep_resid, rebound_resid, event_srs.mean(), min_p, min_p_r, p_laplace, p_normal, n_neg] + event_before_list + event_after_list


def STL_decomp(ts, zipcode, path_save, event_start, event_end, event_type, srs_holiday):
    stl = STL(ts).fit()
    ctfl_loss, resid_analysis = STL_resid_analysis(stl, event_start, event_end, event_type)
    fig = stl.plot()
    axes = fig.get_axes()
    axes[0].set_ylabel("Total spend ($)")
    plt.xticks(rotation=45)
    event_id = f"{zipcode}_{event_type}_{event_start.strftime('%Y-%m-%d')}_{event_end.strftime('%Y-%m-%d')}"
    for holiday_date in srs_holiday.index:
        if holiday_date in ts.index:
            axes[3].axvline(x=holiday_date, color='y', linestyle='--', alpha=0.5, label=srs_holiday[holiday_date])

    if event_type == "EHE":
        color = "r"
    elif event_type == "ECE":
        color = "b"
    axes[3].axvline(x=event_start, color=color, linestyle='--', alpha=0.5, label="Event Start")
    axes[3].axvline(x=event_end, color=color, linestyle='--', alpha=0.5, label="Event End")
    
    # Create a clean legend with only one entry per holiday type
    handles, labels = axes[3].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    year = event_start.year
    
    fig.suptitle(f"Zipcode - {zipcode}, Year - {year}, {event_type}")
    
    # First apply tight_layout to arrange the plots properly
    fig.tight_layout()
    
    # Then add the legend outside and adjust the figure size to make room for it
    leg = fig.legend(by_label.values(), by_label.keys(), 
                      bbox_to_anchor=(1.01, 1), 
                      loc='upper left')
    
    # Save with bbox_inches='tight' to include elements outside the plot area
    fig_save_dir = f"{path_save}/{year}"
    os.makedirs(fig_save_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_save_dir, f"{event_id}.png"), 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    return ctfl_loss, resid_analysis


