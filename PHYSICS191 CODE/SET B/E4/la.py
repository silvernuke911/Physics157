import numpy as np
import pandas as pd

# Neon emission line data (wavelengths in nm, relative intensities)
neon_lines = np.array([
    (540.06, 0.05), (565.66, 0.09), (576.44, 0.19), (582.02, 0.28),
    (585.25, 0.39), (588.19, 0.46), (594.48, 0.78), (597.55, 0.91),
    (603.00, 1.00), (607.43, 0.87), (609.62, 0.79), (614.31, 0.66),
    (616.36, 0.59), (621.73, 0.50), (626.65, 0.39), (630.48, 0.28),
    (633.44, 0.19), (638.30, 0.09), (640.22, 0.05)
])

# Convert to DataFrame and save as CSV
df_neon = pd.DataFrame(neon_lines, columns=["Wavelength", "Intensity"])
df_neon.to_csv("Ne.csv", index=False)