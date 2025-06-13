import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

# Import processing functions from modules
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("processing_module", "modules/01_processing.py")
    processing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processing_module)
    
    process_borehole_data = processing_module.process_borehole_data
    DEFAULT_COLUMN_MAPPINGS = processing_module.DEFAULT_COLUMN_MAPPINGS
    
except ImportError as e:
    print(f"Error importing processing module: {e}")
    print("Please ensure modules/01_processing.py exists and is accessible.")
    sys.exit(1)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(name: str = "BoreholePipeline") -> logging.Logger:
    """
    Set up a logger with colored console output.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format: [TIME] LEVEL: Message
    formatter = ColoredFormatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


def get_user_input() -> Tuple[Path, Path, bool, Dict]:
    """
    Interactive prompt for user inputs.
    
    Returns
    -------
    Tuple[Path, Path, bool, Dict]
        - Input folder path
        - Output folder path  
        - Whether to apply Savitzky-Golay filter
        - Processing parameters
    """
    print("\n" + "="*60)
    print("BOREHOLE DATA PROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Input folder
    while True:
        input_path = input("Enter the path to the folder containing raw CSV files (relative to current directory): ").strip()
        input_folder = Path(input_path)
        
        if input_folder.exists() and input_folder.is_dir():
            csv_files = list(input_folder.glob("*.csv"))
            if csv_files:
                print(f"✓ Found {len(csv_files)} CSV files in '{input_folder}'")
                break
            else:
                print(f"✗ No CSV files found in '{input_folder}'. Please try again.")
        else:
            print(f"✗ Folder '{input_folder}' does not exist. Please try again.")
    
    # Output folder
    output_path = input("\nEnter the path for processed files (relative to current directory): ").strip()
    output_folder = Path(output_path)
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output folder: '{output_folder}'")
    else:
        print(f"✓ Using existing output folder: '{output_folder}'")
    
    # Savitzky-Golay filter
    print("\n" + "-"*40)
    apply_savgol = input("Apply Savitzky-Golay smoothing filter? (y/n) [default: n]: ").strip().lower()
    apply_savgol = apply_savgol == 'y'
    
    # Processing parameters
    params = {
        'apply_savgol': apply_savgol,
        'savgol_window': 11,
        'savgol_order': 3,
        'dz_method': 'percentile95'
    }
    
    if apply_savgol:
        print("\nSavitzky-Golay filter parameters:")
        
        # Window length
        window_input = input("  Window length (odd number) [default: 11]: ").strip()
        if window_input:
            try:
                window = int(window_input)
                if window % 2 == 0:
                    window += 1
                    print(f"  → Adjusted to odd number: {window}")
                params['savgol_window'] = window
            except ValueError:
                print("  → Using default: 11")
        
        # Polynomial order
        order_input = input("  Polynomial order [default: 3]: ").strip()
        if order_input:
            try:
                params['savgol_order'] = int(order_input)
            except ValueError:
                print("  → Using default: 3")
    
    print("\n" + "="*60 + "\n")
    
    return input_folder, output_folder, apply_savgol, params


def process_single_file(input_file: Path, 
                       output_folder: Path,
                       params: Dict,
                       logger: logging.Logger) -> Dict:
    """
    Process a single CSV file through the complete pipeline.
    
    Parameters
    ----------
    input_file : Path
        Path to input CSV file
    output_folder : Path
        Path to output folder
    params : Dict
        Processing parameters
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    Dict
        Processing results and statistics
    """
    file_logger = logging.getLogger(f"BoreholePipeline.{input_file.stem}")
    file_logger.setLevel(logging.INFO)
    
    result = {
        'file': input_file.name,
        'status': 'failed',
        'error': None,
        'stats': {}
    }
    
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        file_logger.info(f"Processing {input_file.name} ({len(df)} rows)")
        
        # Process data
        df_processed, stats = process_borehole_data(
            df,
            apply_savgol=params['apply_savgol'],
            savgol_window=params['savgol_window'],
            savgol_order=params['savgol_order'],
            dz_method=params['dz_method'],
            logger=file_logger
        )
        
        # Generate output filename
        output_name = f"{input_file.stem}_processed.csv"
        output_file = output_folder / output_name
        
        # Save processed data
        df_processed.to_csv(output_file, index=False)
        
        result['status'] = 'success'
        result['stats'] = stats
        result['output_file'] = output_name
        
        file_logger.info(f"✓ Saved to {output_name}")
        
    except Exception as e:
        result['error'] = str(e)
        file_logger.error(f"✗ Error processing {input_file.name}: {e}")
    
    return result


def print_summary(results: List[Dict], processing_time: float):
    """
    Print processing summary.
    
    Parameters
    ----------
    results : List[Dict]
        List of processing results
    processing_time : float
        Total processing time in seconds
    """
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60 + "\n")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Total files processed: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    if successful:
        print("\n" + "-"*40)
        print("STATISTICS SUMMARY")
        print("-"*40)
        
        # Aggregate statistics
        total_original = sum(r['stats']['original_rows'] for r in successful)
        total_final = sum(r['stats']['final_rows'] for r in successful)
        total_removed = sum(r['stats']['negative_removed'] for r in successful)
        total_duplicates = sum(r['stats']['duplicates_found'] for r in successful)
        
        print(f"Total rows processed: {total_original:,}")
        print(f"Total rows after processing: {total_final:,}")
        print(f"Negative values removed: {total_removed:,}")
        print(f"Duplicate depths found: {total_duplicates:,}")
        
        reduction_pct = (1 - total_final / total_original) * 100
        print(f"Data reduction: {reduction_pct:.1f}%")
    
    if failed:
        print("\n" + "-"*40)
        print("FAILED FILES")
        print("-"*40)
        for r in failed:
            print(f"✗ {r['file']}: {r['error']}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logger()
    
    try:
        # Get user inputs
        input_folder, output_folder, apply_savgol, params = get_user_input()
        
        # Get list of CSV files
        csv_files = sorted(input_folder.glob("*.csv"))
        
        logger.info(f"Starting processing of {len(csv_files)} files...")
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")
        
        if apply_savgol:
            logger.info(f"Savitzky-Golay filter: ON (window={params['savgol_window']}, order={params['savgol_order']})")
        else:
            logger.info("Savitzky-Golay filter: OFF")
        
        # Process files with progress bar
        results = []
        start_time = datetime.now()
        
        with tqdm(csv_files, desc="Processing files", unit="file") as pbar:
            for csv_file in pbar:
                pbar.set_postfix_str(f"Current: {csv_file.name}")
                
                result = process_single_file(
                    csv_file, 
                    output_folder,
                    params,
                    logger
                )
                results.append(result)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        print_summary(results, processing_time)
        
        logger.info("Processing complete!")
        
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()