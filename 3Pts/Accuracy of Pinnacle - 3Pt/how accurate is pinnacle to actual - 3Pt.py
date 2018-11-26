import pandas as pd
import os
import numpy as np

os.chdir('C:\\GitHub\\nba-python\\3Pts\\Accuracy of Pinnacle - 3Pt')

df = pd.read_excel('Actual Results - 3Pt - nov 25.xlsx')

df['error'] = df['#'] - df['Actual']
df['error squared'] = np.square(df['error'])

average_error_squared = df['error squared'].sum() / len(df)