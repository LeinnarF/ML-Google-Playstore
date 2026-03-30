# Data Cleaning

This notes documents the things we do for cleaning the data.

## Dataset Overview
- 2019 Google Play store apps dataset.
- Dirty data with missing values and duplicates.
- Columns included:
    - App: Name of the app.
    - Category: Category of the app.
    - Rating: Average user rating of the app (out of 5).
    - Reviews: Number of user reviews for the app.
    - Size: Size of the app.
    - Installs: Number of times the app has been installed.
    - Type: Type of the app (Free or Paid).
    - Price: Price of the app (if it's a paid app).
    - Content Rating: Age group for which the app is suitable.
    - Genres: Genre(s) of the app.
    - Last Updated: Date when the app was last updated.
    - Current Ver: Current version of the app.
    - Android Ver: Minimum Android version required to run the app.

## Pandas Initialization

Pandas provides the necessary tools to clean data.
```python
import pandas as pd
import numpy as np
```

Loading the data from a CSV
```python
df = pd.read_csv('dataset/data.csv')
```

Show missing values
```python
df.isnull().sum()
```

Show information about the data frame including the number of non-null values in each column and the type of data in each column
```python
df.info()
```

## Checking for Duplicates

First check for frequency of values in a column to see if there are any duplicates
```python
df['Column'].value_counts()
```

Remove duplicates
```python
df['Column'].drop_duplicates(inplace=True)
```

The `inplace=True` argument modifies the original data frame. If you want to create a new data frame without duplicates, you can do:
```python
new_df = df['Column'].drop_duplicates()
```

Reset the index after dropping duplicates
```python
df.reset_index(drop=True, inplace=True)
```

## Data Manipulation

In the Rating column, some of values are `NaN` or greater than 5.

Replacing invalid values
```python
df.loc[df['Rating'] > 5, 'Rating'] = np.nan
```

`df.loc` is used to access a group of rows and columns by labels or a boolean array. In this case, we are selecting rows where the 'Rating' column has values greater than 5 and setting those values to `NaN`.

Syntax:
```python
df.loc[condition, 'target_column'] = new_value
```

Where:
- `condition`: A boolean expression that specifies which rows to select.
- `'Column'`: The name of the column where you want to assign the new value.


After this, the next step we did is to align the Ratings with the Reviews. We noticed that some apps had a rating but no reviews, which is not possible. So we set the rating to zero for those apps.

```python
df.loc[df['Reviews'] == 0, 'Rating'] = 0
```

Then for the rest of the missing values in the Rating column, we can fill them with forward fill method, which propagates the last valid observation forward to the next valid observation.

```python
df['Rating'].fillna(method='ffill', inplace=True)
```

### String Manipulation

The Price columns should be numeric, but it contains the dollar sign. We can remove the dollar sign and convert the column to numeric type.

```python
df['Price'] = df['Price'].str.replace('$', '').astype(float)
```

Or without using `.astype(float)`, we can use `pd.to_numeric` to convert the column to numeric type.

```python
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
```

`errors='coerce'` will replace any non-numeric values with `NaN`.

### Combining Values in Columns

The Installs column represent the lower bound of the number of installs, but it contains the '+' sign. We can remove the '+' sign and convert the column to numeric type.

There are values that are redundant, for example `1+` and `0+`. And there is also a true zero value, which is `0`. So we combine 1+ and 0+ into a single value, which is 1, and keep the true zero value as it is.

```python
df['Installs'] = df['Installs'].str.replace({'0+', '1+'})
```

Now we can remove the '+' sign and convert the column to numeric type.

```python
df['Installs'] = df['Installs'].str.replace('+', '').astype(int)
```

### Standardizing Units

The Size column contains values in different units, such as 'M' for megabytes and 'k' for kilobytes. We can standardize the units by converting all sizes to kilobytes.

```python
def convert_size(size):
    if size.endswith('M'):
        return float(size[:-1]) * 1024  # Convert megabytes to kilobytes
    elif size.endswith('k'):
        return float(size[:-1])  # Already in kilobytes
    else:
        return np.nan  # Handle any other cases as NaN
```

Then we can apply this function to the Size column.

```python
df['Size'] = df['Size'].apply(convert_size)
```

The `apply` method is used to apply a function along an axis of the DataFrame. In this case, we are applying the `convert_size` function to each value in the 'Size' column.


### Filtering Data

In the Type column, we only want to keep the free apps. We can filter the data frame to show the inconsistent values in the Type column.

```python
inconsistent_type = df[(df['Price'] > 0) & (df['Type'] != 'Paid') | (df['Price'] == 0) & (df['Type'] != 'Free')]
```

This code filters the data frame to find rows where the Price is greater than 0 but the Type is not 'Paid', or where the Price is equal to 0 but the Type is not 'Free'. This will help us identify any inconsistencies in the Type column based on the Price column.

Resolve inconsistent type
```python
df.loc[df['Price'] > 0, 'Type'] = 'Paid'
df.loc[df['Price'] == 0, 'Type'] = 'Free'
```

This code updates the Type column based on the Price column. If the Price is greater than 0, it sets the Type to 'Paid'. If the Price is equal to 0, it sets the Type to 'Free'. This ensures that the Type column is consistent with the Price column.


## Saving the Cleaned Data

After cleaning the data, we can save the cleaned data to a new CSV file.

```python
df.to_csv('dataset/cleaned_data.csv', index=False)
```

The `index=False` argument prevents pandas from writing row indices to the CSV file, which is often desirable when saving cleaned data for further analysis.

