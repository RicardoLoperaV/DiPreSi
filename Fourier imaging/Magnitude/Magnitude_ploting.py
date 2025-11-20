import sys
from pathlib import Path
# Get the repository root (2 levels up from current notebook)
repo_root = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.insert(0, str(repo_root))

# import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Template
# Definición de modulos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from imblearn.over_sampling import SMOTE

# Instantiate the Template class
Template = Template.Template()

# Build the path to the Excel file in the repository root
data_path = os.path.join(repo_root, 'Datos1_InteraccionesNIR.xlsx')
# import the data from all sheets of the Excel file


df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15 = [
    pd.read_excel(data_path, sheet_name=i) for i in range(16)
]

# If the dataframes have any missing values on the 'Tratamiento' column, delete those rows
for i, df in enumerate([df4,df10]):
    if df['Tratamiento'].isnull().any():
        df.dropna(subset=['Tratamiento'], inplace=True)

# Remove 'Fus_EH' rows from each dataframe
for i, df in enumerate([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]):
    initial_count = len(df)
    df.drop(df[df['Tratamiento'] == 'Fus_EH'].index, inplace=True)


# Lets add the column 'Sana' to all dataframes except df0
for i, df in enumerate([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15], start=0):
    df.insert(1, 'Sana', df['Tratamiento'].apply(lambda x: 1 if x == 'Control' else 0))
    df.drop(columns=['Tratamiento'], inplace=True)
    # Now if the dataframe have the column 'Planta' delete it
    if 'Planta' in df.columns:
        df.drop(columns=['Planta'], inplace=True)



# Apply SMOTE to balance the 'Sana' class in each dataframe
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)

balanced_dataframes = []
for i, df in enumerate([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]):
    print(f"\nDay {i} - Before SMOTE:")
    print(f"  Healthy: {(df['Sana'] == 1).sum()}, Unhealthy: {(df['Sana'] == 0).sum()}")
    
    # Separate features and target
    X = df.drop(columns=['Sana'])
    y = df['Sana']
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create balanced dataframe
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced.insert(0, 'Sana', y_resampled)
    
    print(f"Day {i} - After SMOTE:")
    print(f"  Healthy: {(df_balanced['Sana'] == 1).sum()}, Unhealthy: {(df_balanced['Sana'] == 0).sum()}")
    
    balanced_dataframes.append(df_balanced)

# Replace original dataframes with balanced ones
df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15 = balanced_dataframes

print("\nSMOTE balancing completed!\n")


# Function to perform 2D Fourier Transform
def spatial_2d_fourier(data_1d, grid_size='auto'):
    if grid_size == 'auto':
        # Create square grid
        size = int(np.ceil(np.sqrt(len(data_1d))))
        grid_size = (size, size)
    
    # Reshape data to 2D (pad with zeros if necessary)
    data_2d = np.zeros(grid_size)
    data_2d.flat[:len(data_1d)] = data_1d
    
    # Apply 2D FFT and shift zero frequency to center
    fft_result = fft2(data_2d)
    fft_shifted = fftshift(fft_result)
    
    # Calculate magnitude and phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    return data_2d, magnitude, phase


# Now, lets apply the 2D Fourier Transform to all the dataframes and export the results as images to a folder
output_dir = os.path.join(repo_root, 'Fourier imaging', 'Magnitude', 'Fourier_Images')
dataframes = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]

# Make a for loop to go through all dataframes
for day, df in enumerate(dataframes):
    day_dir = os.path.join(output_dir, f'Day_{day}')
    os.makedirs(day_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        sample_id = f'Sample_{idx}_{"Healthy" if row["Sana"] == 1 else "Unhealthy"}_{row["Sana"]}'

        data_1d = row.drop(['Sana']).values  # Exclude the 'Sana' column
        
        # Perform 2D Fourier Transform
        data_2d, magnitude, phase = spatial_2d_fourier(data_1d)

        # Plot and save the power spectrum without axes, ticks, labels, or margins
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes([0, 0, 1, 1])  # Create axes that fill the entire figure
        ax.imshow(np.log1p(magnitude), cmap='viridis', aspect='auto')
        ax.axis('off')
        plt.savefig(os.path.join(day_dir, f'{sample_id}_magnitude.png'), 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    dpi=100)
        plt.close()
        