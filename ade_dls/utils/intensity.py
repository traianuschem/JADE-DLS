# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:08:00 2025

@author: vinci

intensity processing
"""
import pandas as pd
import matplotlib.pyplot as plt

#extraction of data required for intensity calculations
def extract_intensity(file_path):
  encodings = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1']
  
  for encoding in encodings:
    try:
      with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
        
        #extract angle
        angle = None
        for line in lines:
          if "Angle [°]       :" in line: 
            try:
              angle = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting angle from {file_path}")

        #extract MeanCR0
        meancr0 = None
        for line in lines:
          if "MeanCR0 [kHz]   :" in line: 
            try:
              meancr0 = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting MeanCR0 from {file_path}")
              
        #extract MeanCR1
        meancr1 = None
        for line in lines:
          if "MeanCR1 [kHz]   :" in line: 
            try:
              meancr1 = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting MeanCR2 from {file_path}")
              
        #extract monitor diode
        monitordiode = None
        for line in lines:
          if "Monitor Diode" in line:
            try:
             monitordiode = float(line.split()[-1]) 
             break  
            except (ValueError, IndexError):
              print(f"Error extracting monitor diode from {file_path}")    
              
        #create DataFrame
        data = {'angle [°]': [angle], 'meancr0 [kHz]': [meancr0], 'meancr1 [kHz]': [meancr1], 
                'monitordiode [cps]': [monitordiode]}
        return pd.DataFrame(data)
      
    except UnicodeDecodeError:
      if encoding == encodings[-1]:
        print(f"Failed to decode {file_path} with all attempted encodings")
        return None
      continue
    except FileNotFoundError as e:
      print(f"File not found: {file_path}. Error: {e}")
      return None
    except Exception as e:
      print(f"An error occurred while processing {file_path}: {e}")
      return None

def build_intensity_dataframe(file_paths):
    """
    Load intensity data from a list of ALV .ASC file paths and compute
    monitor-corrected, geometry-corrected mean count rates.

    Correction formula (identical to JADE-DLS 2.1 notebook):

        MeanCR_corr [kHz] = (meancr0 + meancr1) / 2
                            / (monitordiode [cps] * 1e-3)
                            * sin(angle [°])

    Explanation:
      - Average of both detector channels
      - Divide by monitor diode (converted cps → kHz via *1e-3) to
        normalise primary-beam fluctuations between measurements
      - Multiply by sin(θ) to apply the scattering-geometry correction

    Fallback (if monitordiode is unavailable): plain average * sin(θ).

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to ALV .ASC files.

    Returns
    -------
    pd.DataFrame with columns:
        filename, angle [°], meancr0 [kHz], meancr1 [kHz],
        monitordiode [cps], MeanCR_corr [kHz]
    Returns None if no files could be loaded.
    """
    import os
    import numpy as np

    frames = []
    for fp in file_paths:
        df_row = extract_intensity(fp)
        if df_row is not None:
            df_row['filename'] = os.path.basename(fp)
            frames.append(df_row)

    if not frames:
        print("build_intensity_dataframe: no intensity data could be loaded.")
        return None

    df = pd.concat(frames, ignore_index=True)

    # geometry correction factor: sin(θ)
    sin_theta = np.sin(np.radians(df['angle [°]'].fillna(90)))

    raw_mean = (df['meancr0 [kHz]'].fillna(0) + df['meancr1 [kHz]'].fillna(0)) / 2

    valid_md = df['monitordiode [cps]'].notna() & (df['monitordiode [cps]'] > 0)
    if valid_md.any():
        # monitor-diode correction: divide by monitordiode [cps] * 1e-3
        md_khz = df['monitordiode [cps]'] * 1e-3
        df['MeanCR_corr [kHz]'] = raw_mean / md_khz * sin_theta
    else:
        # fallback: no monitor correction, only geometry correction
        print("build_intensity_dataframe: monitordiode data missing — "
              "applying geometry correction only (no monitor normalisation).")
        df['MeanCR_corr [kHz]'] = raw_mean * sin_theta

    return df


#plotting mean intensities for each angle
def plot_meancr(df, x_col, y_col):
    plt.figure(figsize=(8, 4))
    plt.scatter(df[x_col], df[y_col], alpha=0.7, s=50)
    plt.xlabel('angle [°]')
    plt.ylabel('corrected mean intensity [kHz]')
    plt.title('')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    