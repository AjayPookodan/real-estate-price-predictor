# Real Estate Price Prediction

## Project Description

A machine learning project that predicts real-estate prices using property characteristics, location, and accessibility to public amenities and transportation facilities.

## Team Members

* Ajay Balakrishnan
* Gopika CP
* Devakrishnan

## Objectives

* Analyse property prices and their influencing factors.
* Preprocess and explore real-estate datasets.
* Study the impact of location and nearby amenities on property prices.
* Train machine learning models to predict property prices.
* Develop a user-friendly application for making predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* OSMnx
* GeoPy
* Folium
* Streamlit

## Project Structure

```text
real-estate-price-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── app/
├── reports/
├── .gitignore
└── README.md
```

# MODULE 1: Data Collection and Understanding

### Dataset Overview

* **Dataset:** Gurgaon Real Estate Listings
* **Number of Records:** 10,704
* **Number of Attributes:** 67
* **Format:** CSV
* **Target Variable:** `PRICE`
* **Tools Used:** Python, Pandas, Geopy, Nominatim, Folium

### Data Source and Selection

The Gurgaon real-estate dataset was selected because it contains diverse property-related information, including listed prices, property types, locality details, area measurements, furnishing, and amenities. These attributes provide a foundation for analysing the factors that influence property prices and developing a machine learning prediction model.

* **Source URL:** To be verified
* **Author/Uploader:** To be verified
* **Licence:** To be verified
* **Download Date:** To be recorded

### Data Dictionary

A data dictionary is maintained in `reports/data_dictionary.md`. It documents all 67 attributes, including their meanings, data types, units, missing-value percentages, and relevant observations. This helps identify useful prediction features and understand the structure of the dataset before preprocessing.

### Geographic Coordinate Collection

The dataset's locality information was used to obtain geographic coordinates where suitable coordinates were unavailable. The Geopy library and Nominatim geocoding service were used to retrieve latitude and longitude for **231 unique localities**.

The results were stored in a cache to reduce repeated requests. An interactive map was generated using Folium, and approximate geographic boundaries were used to flag potentially incorrect coordinates.

**Limitation:** Coordinates represent localities rather than exact property addresses. Consequently, properties within the same locality may share coordinates, making distance calculations to amenities approximate rather than property-specific.

### Initial Data Quality Observations

Initial inspection identified several issues requiring further investigation:

* **Inconsistent price formats:** The `PRICE` column contains values such as `2.63 Cr`, `3.6 Cr`, and comma-separated amounts such as `85,000`. These must be converted into a consistent numerical format after verifying their units and transaction types.
* **Mixed data types:** Some attributes, including `FURNISHING_ATTRIBUTES`, contain mixed data types and require inspection.
* **Missing values:** Missing-value percentages are documented in the data dictionary and will guide preprocessing decisions.
* **Potentially redundant features:** Multiple price, area, identifier, and property-description fields must be examined for overlap and relevance.
* **Geographic accuracy:** Geocoded coordinates must be checked for incorrect locations before calculating distances to nearby amenities.
* **Time coverage:** Listing dates must be inspected to determine the period represented by the dataset.

### Data Storage

The original dataset is preserved in `data/raw/housing_raw.csv`. Geocoded data is stored separately in `data/raw/housing_with_coords.csv`, while the geocoding cache is maintained in `data/raw/geocode_cache.json`. The data dictionary and interactive locality map are stored in the `reports/` directory.

### Outcome

This module establishes the initial dataset and documents its structure, geographic information, and data quality considerations. The findings will guide the next stage: data preprocessing, feature engineering, and preparation of a consistent target variable for machine learning.


# MODULE 2: DATA PREPROCESSING

## Real Estate Price Prediction Using Property Characteristics and Geographical Features

---

# 1. Introduction

Data preprocessing is the second major module of the Real Estate Price Prediction project.

The purpose of this module is to transform the original raw real estate dataset into a structured, consistent, and usable dataset for exploratory data analysis (EDA) and machine learning.

The original dataset contains property listings collected from a real estate platform. Each listing includes information such as property type, locality, price, number of bedrooms, furnishing status, floor information, area measurements, and geographical details.

However, raw real estate data is not immediately suitable for machine learning. It may contain rental listings mixed with sale listings, duplicate advertisements, inconsistent price formats, missing measurements, invalid geographical coordinates, unrealistic property sizes, and columns that do not contribute directly to price prediction.

Therefore, a series of preprocessing operations was performed to improve data quality and prepare the dataset for the next stages of the project.

The final dataset produced by the current preprocessing workflow contains **7,457 property listings and 18 columns**.

The resulting data is saved in:

`data/processed/clean_v2.csv`

This file serves as the input for the next module, Exploratory Data Analysis.

---

# 2. Objectives of Module 2

The main objectives of preprocessing are:

1. Load the original real estate dataset.
2. Standardize column names and text values.
3. Separate sale listings from rental listings.
4. Remove repeated property IDs.
5. Extract latitude and longitude from geographical metadata.
6. Retain listings located within the selected Gurgaon geographical area.
7. Convert property prices into lakhs of Indian rupees.
8. Combine multiple area columns into a unified area feature.
9. Convert relevant columns into numeric data types.
10. Remove records with missing essential information.
11. Filter implausible property areas and bedroom counts.
12. Apply a price-per-square-foot sanity filter.
13. Remove listings with identical property characteristics and prices.
14. Group infrequently occurring localities.
15. Select useful features and exclude irrelevant or potentially misleading columns.
16. Save the cleaned dataset for further analysis.
17. Review the final dataset's dimensions, missing values, and descriptive statistics.

These operations are intended to improve the reliability of the data before machine learning models are trained.

---

# 3. Dataset Used

## 3.1 Original Dataset

The raw dataset is stored at:

`data/raw/housing_raw.csv`

The file contains property listings and their associated attributes.

The original dataset dimensions were:

| Property          |  Value |
| ----------------- | -----: |
| Number of rows    | 10,704 |
| Number of columns |     67 |
| Sale listings     | 10,155 |
| Rental listings   |    549 |

Each row represents a property listing in the original dataset.

The 67 columns include useful property attributes as well as additional metadata, such as image links, descriptions, identifiers, price representations, location details, and other information associated with the listings.

Not every original column is useful for predicting a property's sale price. Some columns also contain information that may be redundant, difficult to interpret, or potentially introduce data leakage.

For this reason, the original dataset was reduced to a smaller set of relevant features.

## 3.2 Final Processed Dataset

The current preprocessing script produces:

`data/processed/clean_v2.csv`

The latest reported dimensions are:

| Property             |                   Value |
| -------------------- | ----------------------: |
| Number of rows       |                   7,457 |
| Number of columns    |                      18 |
| Target variable      |           `price_lakhs` |
| Geographic features  | `latitude`, `longitude` |
| Unified area feature |             `area_sqft` |

The dataset retains property characteristics, geographical information, locality categories, and the target property price.

**Important:** The current dataset has a minimum recorded price of ₹100 lakhs. This requires further investigation because it may indicate that lower-priced sale listings were excluded by the filtering rules.

---

# 4. Preprocessing Workflow at a Glance

The following table summarizes the main stages and their recorded row counts.

| Stage | Operation                                                        | Rows Remaining |
| ----- | ---------------------------------------------------------------- | -------------: |
| 1     | Load original dataset                                            |         10,704 |
| 2     | Retain sale listings                                             |         10,155 |
| 3     | Remove duplicate property IDs                                    |         10,155 |
| 4     | Validate Gurgaon coordinates                                     |          9,672 |
| 5     | Apply area, bedroom, bathroom, and price-per-square-foot filters |          8,400 |
| 6     | Remove near-duplicate listings                                   |          7,457 |
| 7     | Select final features                                            |          7,457 |

These counts were printed by the preprocessing script.

The stages involving price conversion, area creation, and missing-value checks were also performed, but the table above records the major filtering stages for which row counts were explicitly reported.

The exact number of rows removed by each individual outlier rule was not recorded separately in the available output.

---

# 5. Detailed Explanation of Each Preprocessing Step

## Step 1: Loading the Raw Dataset

The preprocessing workflow begins by loading the original CSV file using Pandas.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/housing_raw.csv", low_memory=False)
```

### Why was this necessary?

The original CSV file is the source of all information used in the project.

Pandas loads the file into a DataFrame, allowing the data to be inspected, filtered, transformed, and saved.

The parameter `low_memory=False` helps Pandas infer column data types using the complete file rather than processing it in smaller chunks.

This is useful when a dataset contains columns with mixed data types.

The raw dataset is retained as the original source, while the preprocessing operations create a separate processed dataset.

---

## Step 2: Standardizing Column Names

The column names were standardized by:

* Removing leading and trailing spaces.
* Converting column names to lowercase.
* Replacing spaces with underscores.
* Replacing hyphens with underscores.

Example:

| Original Column Name  | Standardized Name     |
| --------------------- | --------------------- |
| `PROPERTY_TYPE`       | `property_type`       |
| `BEDROOM_NUM`         | `bedroom_num`         |
| `TOTAL_FLOOR`         | `total_floor`         |
| `PRICE_PER_UNIT_AREA` | `price_per_unit_area` |

The transformation uses operations such as:

```python
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)
```

### Why was this necessary?

Consistent column names make the code easier to write and maintain.

For example, `bedroom_num` can be accessed consistently without worrying about capitalization or spaces.

This step does not change the underlying property information. It only standardizes how the columns are named.

---

## Step 3: Filtering Sale Listings

The original dataset contains both properties offered for sale and properties offered for rent.

The `preference` column was used to distinguish between these categories.

The reported counts were:

| Listing Type | Code | Number of Listings |
| ------------ | ---- | -----------------: |
| Sale         | `S`  |             10,155 |
| Rent         | `R`  |                549 |
| Total        |      |             10,704 |

Only sale listings were retained.

The filtering operation was:

```python
df = df[df["preference"].astype(str).str.strip().str.upper() == "S"].copy()
```

### Why was this necessary?

The objective of the project is to predict the sale price of a property.

Rental prices and property sale prices represent different targets. Combining them in the same target variable would produce misleading results.

For example, a monthly rent of ₹30,000 cannot be treated as equivalent to a sale price of ₹30 lakh.

By filtering on the listing category, the dataset is restricted to the intended prediction task.

### Result

The dataset decreased from 10,704 rows to 10,155 sale listings.

The important point is that **the rent filter is based on the listing category, not on the property's price**.

However, the later preprocessing stages may have removed some genuine sale listings based on the applied price and area thresholds.

---

## Step 4: Removing Duplicate Property IDs

The `prop_id` column identifies property listings.

Duplicate property IDs were removed using:

```python
df = df.drop_duplicates(subset=["prop_id"])
```

### Why was this necessary?

Duplicate records can cause the same listing to appear multiple times during analysis or model training.

Repeated records can give certain properties more influence over the model than they should have.

Removing duplicate IDs helps ensure that each property identifier appears only once.

### Result

The row count remained at 10,155.

This indicates that no additional rows were removed at this stage, according to the reported row counts.

### Limitation

Different property IDs may still refer to the same physical property. For example, different agents might advertise the same apartment using separate listing IDs.

That issue was addressed separately in the near-duplicate removal stage.

---

## Step 5: Extracting Latitude and Longitude

Location is an important factor in real estate pricing.

Two properties with similar areas and bedroom counts may have different prices because they are located in different neighbourhoods or have different levels of access to nearby facilities.

The original dataset contained a column called `map_details`.

This column stored geographical information as text representing a dictionary.

The preprocessing script parsed this information using Python's `ast.literal_eval()`.

The following features were extracted:

* `latitude`
* `longitude`

For example, a location record might contain coordinates resembling:

```text
LATITUDE: 28.43
LONGITUDE: 77.05
```

These values were converted into numerical coordinates.

### Why use `ast.literal_eval()`?

The geographical information was stored as text rather than as an already structured Python dictionary.

`ast.literal_eval()` can convert a string representation of a Python literal into its corresponding Python object.

The script then extracts the latitude and longitude values from the resulting dictionary.

Invalid or missing coordinate information is converted into missing values.

### Why were coordinates necessary?

Coordinates are required for the planned geographical feature-engineering stage.

They can later be used to calculate distances to:

* Railway stations
* Bus stops
* Hospitals
* Schools
* Shopping centres
* Major roads and highways

These distances can become additional model features.

The coordinates therefore serve as a foundation for the location-based component of the project.

---

## Step 6: Validating the Gurgaon Geographical Area

After extracting the coordinates, the dataset was restricted to a selected geographical bounding box.

The conditions used were:

| Coordinate | Lower Bound | Upper Bound |
| ---------- | ----------: | ----------: |
| Latitude   |       28.30 |       28.55 |
| Longitude  |       76.80 |       77.20 |

A record was retained only when both coordinates fell within these ranges.

### Why was this necessary?

The dataset was intended to focus on properties in Gurgaon.

A bounding-box check removes records with coordinates outside the selected geographical area.

It also helps identify invalid coordinate values that might otherwise affect distance calculations.

### Result

The row count decreased from 10,155 to 9,672.

Therefore, 483 listings were excluded during coordinate validation.

### Important limitation

A bounding box is a rectangular geographical approximation, not an exact administrative boundary.

A coordinate can fall inside the selected rectangle without necessarily belonging to Gurgaon itself.

For a more rigorous implementation, the coordinates can later be validated against an authoritative Gurgaon boundary.

---

## Step 7: Converting Property Prices into Lakhs

The original dataset's price values were not consistently represented in a single numerical unit.

Some listings used lakh or crore notation, while other records contained numerical values.

The preprocessing workflow created a target column named:

`price_lakhs`

This column is intended to represent the property sale price in lakhs of Indian rupees.

The conversion rules are:

| Original Representation | Equivalent Price in Lakhs |
| ----------------------- | ------------------------: |
| ₹50,00,000              |                        50 |
| ₹75,00,000              |                        75 |
| ₹1 crore                |                       100 |
| ₹1.5 crore              |                       150 |
| ₹2 crore                |                       200 |
| ₹5 crore                |                       500 |

The general conversion formula for a numerical price recorded in rupees is:

$$
\text{Price in lakhs}=\frac{\text{Price in rupees}}{100000}
$$

For a value recorded in crores:

$$
\text{Price in lakhs}=\text{Price in crores}\times100
$$

### Why was this necessary?

A machine learning model requires a consistent numerical target.

The same unit must be used for every record so that the model learns meaningful relationships between property features and sale prices.

### Important issue discovered during validation

The current dataset has these reported price statistics:

| Statistic | Price in Lakhs |
| --------- | -------------: |
| Count     |          7,457 |
| Mean      |         347.22 |
| Median    |            250 |
| Minimum   |            100 |
| Maximum   |          4,500 |

The minimum of 100 lakhs means the cheapest retained property is recorded at ₹1 crore.

This is a significant validation issue because the raw dataset contains sale listings, and some genuine Gurgaon properties may be priced below ₹1 crore.

The minimum price does not by itself prove that the conversion is incorrect. It does, however, mean that the effect of the filters must be investigated.

Before model training, the original price values should be compared with `price_lakhs` to verify that the conversion is correct and that lower-priced properties were not unintentionally excluded.

---

## Step 8: Creating a Unified Property-Area Feature

The original dataset contained multiple area-related columns:

* `superbuiltup_sqft`
* `super_sqft`
* `builtup_sqft`
* `carpet_sqft`

These columns describe different measurements of property size.

However, the available values were incomplete. Some listings had a super built-up area but no carpet area, while others had different area fields populated.

The preprocessing script combined the available measurements into a single column called:

`area_sqft`

The priority order was:

1. `superbuiltup_sqft`
2. `super_sqft`
3. `builtup_sqft`
4. `carpet_sqft`

The code uses `combine_first()` to take the first non-missing value in that sequence.

Conceptually:

```python
df["area_sqft"] = (
    df["superbuiltup_sqft"]
    .combine_first(df["super_sqft"])
    .combine_first(df["builtup_sqft"])
    .combine_first(df["carpet_sqft"])
)
```

### Why was this necessary?

Having multiple area columns complicates analysis and modelling.

A unified area feature allows the project to examine the relationship between property size and sale price using a single column.

For example, the model can learn whether larger properties generally have higher prices.

### Important limitation

Carpet area, built-up area, super area, and super built-up area are not identical measurements.

Selecting the first available value does not make these definitions equivalent.

Consequently, `area_sqft` represents the best available area measurement according to the chosen priority order, rather than a perfectly standardized measurement.

This limitation should be considered when interpreting model performance.

---

## Step 9: Creating the `area_is_carpet` Feature

A binary feature named `area_is_carpet` was created.

Its purpose is to indicate whether carpet area was used as the fallback area measurement.

The logic is:

* `0`: At least one of the other three area columns was available.
* `1`: Super built-up, super area, and built-up area were all missing, so carpet area was used.

### Why was this necessary?

The source area measurement can influence comparisons between properties.

A carpet-area measurement may be substantially smaller than a super built-up measurement for the same property.

The indicator provides the model with information about which type of measurement was used as the fallback.

### Final statistics

The reported mean of `area_is_carpet` is approximately 0.255.

For a binary variable, this indicates that approximately 25.5% of the retained records used carpet area as the fallback measurement.

This feature does not convert carpet area into super built-up area. It only records which fallback was used.

---

## Step 10: Converting Columns into Numeric Data Types

The preprocessing script converted relevant property attributes into numeric values.

These included:

* `bedroom_num`
* `bathroom_num`
* `balcony_num`
* `total_floor`
* `total_landmark_count`
* `age`
* `floor_num`

The `floor_num` column received additional treatment for text descriptions such as `Ground` and `Basement`.

The script converted:

* `Ground` to `0`
* `Basement` to `-1`

Other values were converted using numeric parsing.

Values that could not be converted became missing values.

### Why was this necessary?

Numeric calculations and filtering operations require columns to contain suitable data types.

For example, the condition:

```python
df["bedroom_num"].between(1, 8)
```

requires bedroom counts to be numeric.

Similarly, area-per-bedroom calculations require numerical values for both area and bedroom count.

### Important distinction

Not every integer-valued column should be treated as a continuous numerical feature.

Some columns represent category codes rather than quantities.

For example:

* `furnish` may identify furnishing categories.
* `facing` may identify property-facing directions.
* `owntype` may identify ownership categories.
* `age` may represent age bands rather than exact property ages.

Their meanings should be confirmed from the source dataset before deciding how to encode them for machine learning.

---

## Step 11: Removing Records with Missing Essential Information

Records without essential values were excluded before the main outlier filters.

The required information included:

* `price_lakhs`
* `area_sqft`
* `bedroom_num`
* `bathroom_num`

### Why was this necessary?

These fields are required for the current preprocessing rules and model objective.

Without a valid target price, a record cannot be used as a supervised training example.

Without a usable area or bedroom count, the property cannot be evaluated using the current area and room plausibility checks.

The preprocessing script therefore removed records missing these essential fields.

Other missing values, such as missing balcony or floor information, were not automatically grounds for removing the entire property.

This helps preserve useful records while allowing less essential missing values to be handled later.

---

## Step 12: Filtering Unrealistic Property Areas

The first area filter retained properties satisfying:

$$
400 \leq \text{area\_sqft} \leq 10000
$$

This means that properties with recorded areas below 400 square feet or above 10,000 square feet were excluded.

### Why was this necessary?

Extremely small or extremely large measurements may indicate:

* Incorrectly entered values
* Inconsistent measurement units
* Data extraction errors
* Unusual properties that require additional inspection

Removing implausible values can help reduce the influence of measurement errors on the model.

### Limitation

The thresholds are assumptions used for this project, not universal real estate standards.

A genuine property can fall outside the chosen range.

Therefore, the filter should be described as a plausibility filter rather than proof that every excluded property was invalid.

---

## Step 13: Filtering Bedroom Counts

The dataset was restricted to properties with bedroom counts between 1 and 8.

The condition was:

$$
1 \leq \text{bedroom\_num} \leq 8
$$

### Why was this necessary?

The original dataset contained records with unusually large bedroom counts.

Such records may represent data errors, unusual property types, or listings that require separate treatment.

The chosen filter retains conventional residential configurations while removing records outside the selected range.

### Result

The final dataset's reported bedroom statistics are:

| Statistic | Bedrooms |
| --------- | -------: |
| Minimum   |        1 |
| Median    |        3 |
| Mean      |     3.38 |
| Maximum   |        8 |

The final dataset therefore contains properties with one to eight recorded bedrooms.

---

## Step 14: Filtering Area per Bedroom

A further plausibility check was applied to the relationship between area and bedroom count.

The condition was:

$$
\frac{\text{area\_sqft}}{\text{bedroom\_num}} \geq 250
$$

In other words, each property needed at least 250 square feet of recorded area per bedroom.

### Why was this necessary?

A property with a very small recorded area and a high bedroom count may indicate inconsistent data.

For example, a 500-square-foot property with four bedrooms would have only 125 square feet per bedroom.

The filter removes such combinations because they may represent incorrect area measurements or unusual listings.

### Limitation

Area per bedroom does not account for the entire layout, including bathrooms, kitchens, corridors, and living areas.

The threshold is therefore a rough plausibility check rather than an exact rule for determining whether a property is genuine.

---

## Step 15: Filtering Bathroom Counts

The next condition was:

$$
\text{bathroom\_num} \leq \text{bedroom\_num} + 2
$$

For example:

| Bedrooms | Maximum Bathrooms Allowed |
| -------: | ------------------------: |
|        1 |                         3 |
|        2 |                         4 |
|        3 |                         5 |
|        4 |                         6 |
|        5 |                         7 |

Listings that exceeded this condition were excluded.

### Why was this necessary?

Extremely high bathroom counts relative to bedroom counts may indicate incorrect entries or inconsistent listing information.

The rule removes some unusual combinations before the model is trained.

### Limitation

The condition is a heuristic and does not guarantee that every retained bedroom-bathroom combination is valid.

It also does not guarantee that every excluded combination is incorrect.

---

## Step 16: Filtering Price per Square Foot

A helper value was calculated for each property:

$$
\text{Price per square foot}
=
\frac{\text{Price in lakhs}\times100000}
{\text{Area in square feet}}
$$

This produces a price-per-square-foot value in Indian rupees.

The selected range was:

$$
3000 \leq \text{Price per square foot} \leq 60000
$$

Listings below ₹3,000 per square foot or above ₹60,000 per square foot were excluded.

### Why was this necessary?

A property's sale price should be reasonably consistent with its recorded area.

An unusually high price per square foot can indicate a price-entry error, an incorrect area measurement, or a special property that differs substantially from the rest of the dataset.

An unusually low price per square foot may indicate a similar issue.

This filter was applied to reduce extreme price-area combinations.

### Important consequence

This filter can also exclude genuine properties.

For example, a low-priced property with a large recorded area might be valid even if its price per square foot falls below the chosen minimum.

Because this filter depends on the target price, its impact on the final target distribution must be examined carefully.

The reported minimum price of ₹100 lakhs suggests that the filtering rules may have disproportionately excluded lower-priced listings.

This should be investigated before the dataset is treated as representative of the full sale-property market.

### Data leakage consideration

Price per square foot is directly calculated using the target price.

Therefore, it is suitable for data-quality checks but should not be used as a model input when predicting an unknown sale price.

The preprocessing script does not retain this helper feature in the final selected columns.

---

## Step 17: Removing Near-Duplicate Listings

Duplicate property IDs do not capture every possible duplicate listing.

Different agents may advertise the same property with different identifiers.

To reduce this issue, the script identifies duplicate records based on a combination of:

* `locality`
* `area_sqft`
* `bedroom_num`
* `bathroom_num`
* `price_lakhs`

Records matching across all these fields are reduced to one.

### Why was this necessary?

Near-duplicate listings can cause the same property to appear multiple times in the dataset.

If repeated versions of the same property appear in both the training and testing datasets, the model may appear more accurate than it would be on genuinely unseen properties.

Reducing duplicates helps limit this problem.

### Result

The row count decreased from 8,400 to 7,457.

A total of 943 additional records were removed during this stage.

### Limitation

This method identifies records with identical values across the selected fields.

It cannot reliably identify properties with slightly different areas, prices, or room counts.

A more advanced deduplication process could use normalized addresses, coordinates, property identifiers, and approximate matching.

---

## Step 18: Grouping Rare Localities

The dataset contains many different locality names.

A locality with only a handful of records may not provide enough examples for a machine learning model to learn a reliable price pattern.

To address this, the script creates a new column called:

`locality_grouped`

The grouping rule is:

* If a locality has at least 10 listings, retain its name.
* If a locality has fewer than 10 listings, assign it to `Other`.

### Why was this necessary?

Locality is an important property-price feature, but many unique locality names can create a high-cardinality categorical variable.

Grouping rare categories reduces the number of distinct values and helps make categorical encoding more manageable.

The common locality categories remain available for analysis, while rare categories are grouped together.

### Important distinction

The original `locality` column is used to create the grouped feature, but it is not retained in the final dataset.

Instead, `locality_grouped` is used as the locality feature.

The grouped locality labels are not numerical values. They should be encoded appropriately when the machine learning pipeline is constructed.

---

## Step 19: Selecting the Final Columns

The original dataset contains 67 columns, but only a subset is needed for the intended model.

The preprocessing script selects 18 columns.

These columns include:

* Property identification
* Geographic coordinates
* Property type
* Ownership category
* Bedroom and bathroom counts
* Balcony count
* Furnishing and facing categories
* Property age category
* Floor information
* Landmark count
* Unified property area
* Area-source indicator
* Grouped locality
* Target property price

The script excludes fields such as image URLs, descriptions, original price representations, original area columns, and unnecessary metadata.

The `brokerage` column is also excluded because its meaning and relationship with property price are uncertain.

### Why was this necessary?

Keeping unnecessary columns can complicate the modelling process and introduce irrelevant information.

Some original columns may be identifiers, free-text descriptions, or alternate representations of features already retained elsewhere.

Other columns may be derived from the target price and create data leakage.

Selecting a smaller set of relevant columns produces a cleaner dataset for subsequent analysis.

### Role of `prop_id`

The `prop_id` column remains in the processed CSV for traceability.

It is not intended to be used as a predictive feature.

When constructing the model's input matrix, `prop_id` should be excluded.

---

# 6. Understanding the Final 18 Columns

The final dataset contains the following columns.

| Column                 | Data Meaning                               | Intended Role                                  |
| ---------------------- | ------------------------------------------ | ---------------------------------------------- |
| `prop_id`              | Unique property listing identifier         | Traceability only                              |
| `latitude`             | North-south geographical coordinate        | Location feature                               |
| `longitude`            | East-west geographical coordinate          | Location feature                               |
| `property_type`        | Property category                          | Categorical feature                            |
| `owntype`              | Ownership category code                    | Categorical feature                            |
| `bedroom_num`          | Number of bedrooms                         | Numerical feature                              |
| `bathroom_num`         | Number of bathrooms                        | Numerical feature                              |
| `balcony_num`          | Number of balconies                        | Numerical feature                              |
| `furnish`              | Furnishing category code                   | Categorical feature                            |
| `facing`               | Property-facing category code              | Categorical feature                            |
| `age`                  | Property-age category code                 | Categorical feature if it represents age bands |
| `floor_num`            | Floor on which the property is located     | Numerical feature, subject to validation       |
| `total_floor`          | Total number of floors                     | Numerical feature                              |
| `total_landmark_count` | Recorded number of landmarks               | Potential location-related feature             |
| `area_sqft`            | Unified property-area measurement          | Numerical feature                              |
| `area_is_carpet`       | Indicates use of carpet area as a fallback | Binary feature                                 |
| `locality_grouped`     | Locality with rare categories grouped      | Categorical feature                            |
| `price_lakhs`          | Property price in lakhs of rupees          | Target variable                                |

## 6.1 Target Variable

The target variable is:

`price_lakhs`

The objective of the machine learning model is to predict this value using the remaining appropriate property and location features.

For example, the model may learn relationships between sale price and:

* Property area
* Number of bedrooms
* Number of bathrooms
* Property type
* Locality
* Geographical location
* Building and furnishing characteristics

## 6.2 Input Features

The input features are the information available to the model when a property price is to be predicted.

The target price itself must never be included in the input matrix.

The property identifier is also excluded because it identifies a listing rather than describing a meaningful property characteristic.

Any features derived directly from the target price must also be excluded.

---

# 7. Final Dataset Statistics

The latest reported descriptive statistics provide an overview of the processed data.

## 7.1 Property Prices

| Statistic          |        Value |
| ------------------ | -----------: |
| Number of listings |        7,457 |
| Mean price         | ₹347.22 lakh |
| Median price       |    ₹250 lakh |
| Standard deviation | ₹307.58 lakh |
| Minimum price      |    ₹100 lakh |
| Maximum price      |  ₹4,500 lakh |

The mean price is higher than the median price.

This suggests that the price distribution is right-skewed, meaning that a relatively small number of expensive properties raise the average.

The maximum recorded price is ₹45 crore.

The minimum recorded price is ₹1 crore and must be investigated because lower-priced sale listings may have been excluded by the preprocessing rules.

## 7.2 Property Area

| Statistic          |            Value |
| ------------------ | ---------------: |
| Number of listings |            7,457 |
| Mean area          | 2,501.80 sq. ft. |
| Median area        |    2,217 sq. ft. |
| Standard deviation | 1,109.58 sq. ft. |
| Minimum area       |      594 sq. ft. |
| Maximum area       |   10,000 sq. ft. |

The mean area is greater than the median area, suggesting some right-skewness in property sizes.

The minimum and maximum values reflect the area filters applied during preprocessing.

## 7.3 Bedrooms

| Statistic | Value |
| --------- | ----: |
| Mean      |  3.38 |
| Median    |     3 |
| Minimum   |     1 |
| Maximum   |     8 |

Most retained properties have approximately three or four bedrooms.

## 7.4 Geographical Coordinates

| Statistic | Latitude | Longitude |
| --------- | -------: | --------: |
| Mean      |  28.4380 |   77.0406 |
| Minimum   |  28.3509 |   76.8108 |
| Maximum   |  28.5348 |   77.1915 |

The coordinate values fall within the selected bounding box.

These values will be used later for geographical analysis and the development of amenity-related features.

---

# 8. Missing-Value Analysis

The latest missing-value output showed three missing values in `balcony_num`.

The other reported columns had no missing values.

The missing-value check is important because missing data can cause errors during analysis and machine learning.

For this project, missing values can be handled in the modelling pipeline.

A suitable approach is:

* Use median imputation for numerical columns.
* Use an `Unknown` category or most-frequent-value imputation for categorical columns.
* Review missing floor information according to the meaning of the source values.

Imputation should be fitted using the training data rather than the entire dataset to avoid information leaking from the test set into model training.

The three missing balcony values should therefore be handled before or within the model pipeline.

---

# 9. Data Leakage Prevention

Data leakage occurs when the model receives information during training that would not legitimately be available when predicting an unknown property's price.

The preprocessing module addresses several potential leakage risks.

## 9.1 Target Separation

The `price_lakhs` column is the target variable.

It must be separated from the input features before model training.

## 9.2 Property Identifier

The `prop_id` column is retained for traceability but should not be included in the model's input features.

## 9.3 Price-Derived Columns

The original price-related columns and the calculated price-per-square-foot helper are excluded from the final feature set.

Price per square foot directly uses the target price in its calculation. Including it as an input feature would provide the model with information about the value it is supposed to predict.

## 9.4 Brokerage

The `brokerage` column is excluded because the meaning of its values is uncertain and it may be related to the property's price.

## 9.5 Train-Test Separation

Preprocessing operations that learn parameters from data, such as imputing missing values, scaling numerical features, and encoding categories, should be fitted on the training set only.

This helps ensure that the test set remains independent for model evaluation.

---

# 10. Visual Validation

The preprocessing workflow also includes a comparison of property-area distributions before and after filtering.

The before-filtering plot uses the available raw `superbuiltup_sqft` values.

The after-filtering plot uses the unified `area_sqft` values from the final processed dataset.

These plots help identify differences in the distributions and assess the effect of the area filters.

However, the two plots do not contain exactly the same observations or necessarily the same area definitions.

The raw plot uses only non-missing super built-up area values, while the final plot includes area measurements selected from multiple columns.

Therefore, differences between the plots cannot be attributed exclusively to the filtering rules.

For a more accurate before-and-after comparison, the same unified-area construction should be applied to the raw dataset before plotting the distribution before filtering.

---

# 11. Limitations of the Current Preprocessing

Although the dataset has been substantially reduced and standardized, several limitations remain.

## 11.1 Minimum Price Requires Investigation

The minimum price in the final dataset is ₹100 lakhs.

This may indicate that the applied filters disproportionately excluded lower-priced properties.

The original sale/rent separation uses the `preference` column, but the final minimum price must be verified against the raw sale listings.

The price conversion and price-per-square-foot filtering rules should be reviewed before final model training.

## 11.2 Different Area Definitions

The unified `area_sqft` column can contain super built-up, super area, built-up, or carpet-area measurements.

These definitions differ, so the column is not perfectly standardized.

The `area_is_carpet` feature records whether carpet area was used as a fallback, but it does not eliminate all measurement differences.

## 11.3 Heuristic Outlier Filters

The selected thresholds for area, bedrooms, bathrooms, and price per square foot are project-specific assumptions.

They help remove implausible records but may also remove valid properties.

The effect of each rule should be reviewed and documented.

## 11.4 Duplicate Listings

The current near-duplicate removal method relies on matching selected property characteristics.

It may not identify every repeated property if some fields differ slightly.

## 11.5 Category Interpretation

Columns such as `owntype`, `furnish`, `facing`, and `age` may contain category codes.

Their meanings should be verified from the source dataset before they are encoded for machine learning.

## 11.6 Landmark Count

The `total_landmark_count` column is retained, but its exact definition and calculation should be confirmed.

A count of landmarks is not necessarily equivalent to accessibility. Two properties can have the same landmark count while being very different distances from important facilities.

## 11.7 Amenity Distances Have Not Yet Been Added

The current processed dataset contains latitude and longitude, but it does not yet contain verified distances to railway stations, bus stops, hospitals, schools, shopping centres, or highways.

Those features belong to a later feature-engineering stage.

---

# 12. Output File and Reproducibility

The current output file is:

`data/processed/clean_v2.csv`

The preprocessing code is maintained in:

`src/preprocess.py`

The script rebuilds the processed dataset from the raw CSV file.

Running the script from the project root with:

```bash
python src/preprocess.py
```

executes the preprocessing workflow, provided that the script's file paths are configured relative to the project root.

The script also prints a row-count log that records the number of rows remaining after the principal filtering stages.

This log is useful for documenting data reduction and explaining how the original dataset was transformed into the final dataset.

The original raw file should remain unchanged so that the preprocessing workflow can be repeated when the rules are revised.

---

# 13. Summary of What Was Accomplished

The preprocessing module transformed the original real estate dataset into a smaller and more structured dataset suitable for further analysis.

The main accomplishments were:

1. Loaded and standardized the raw dataset.
2. Filtered the records to retain sale listings.
3. Removed duplicate property identifiers.
4. Extracted latitude and longitude from geographical metadata.
5. Restricted the data to a selected Gurgaon geographical bounding box.
6. Created a numerical price target expressed in lakhs.
7. Combined multiple area columns into a unified area feature.
8. Created an indicator for records that use carpet area as a fallback.
9. Converted important property attributes to numeric types.
10. Removed records missing essential information.
11. Applied area, bedroom, bathroom, and price-per-square-foot filters.
12. Removed records matching the selected near-duplicate criteria.
13. Grouped infrequent localities into `Other`.
14. Selected 18 columns for the processed dataset.
15. Examined the final price, area, bedroom, and geographical statistics.
16. Saved the output to `data/processed/clean_v2.csv`.

The final reported dataset contains **7,457 rows and 18 columns**, including geographical coordinates and the target variable `price_lakhs`.

The output provides a foundation for exploratory data analysis and subsequent model development.

However, the minimum-price cutoff and the differing area definitions require further attention before the dataset can be considered fully validated.

---

# 14. Conclusion

Module 2 established the data preparation workflow for the Real Estate Price Prediction project.

The raw dataset contained 10,704 listings across 67 columns. Through sale-only filtering, geographical validation, data-type conversion, property plausibility checks, price-per-square-foot filtering, and duplicate removal, the dataset was reduced to 7,457 records.

The final dataset retains property characteristics, locality information, geographical coordinates, a unified area measurement, and the sale-price target.

These features provide a starting point for studying how property characteristics and location relate to sale prices.

The next module will focus on **Exploratory Data Analysis (EDA)**, where the distributions of prices and property characteristics will be examined, relationships between features will be investigated, and patterns across localities will be explored.

Before model training, the price conversion, minimum-price cutoff, remaining missing values, and area-measurement differences should be reviewed to ensure that the model is trained on appropriately validated data.

# Module 3: Exploratory Data Analysis (EDA)

## Real Estate Price Prediction Using Machine Learning

---

## 1. Introduction

Exploratory Data Analysis (EDA) is an important stage in a machine learning project. Before training a model, it is necessary to understand the dataset, examine the distributions of variables, identify relationships between features, detect possible data-quality issues, and determine which features may help predict the target variable.

In this project, EDA is performed on a cleaned real-estate dataset containing property listings for Gurgaon, India.

The objective is to investigate the relationship between property sale prices and characteristics such as:

* Property area.
* Number of bedrooms.
* Number of bathrooms.
* Property type.
* Furnishing status.
* Property age.
* Facing direction.
* Ownership type.
* Locality.
* Geographic coordinates.

The analysis also examines how prices vary across different localities and whether the spatial distribution of property listings reveals geographical patterns.

The findings from EDA will guide the next stage of the project, where additional location-related features will be investigated and machine learning models will be developed.

### Main objectives

The objectives of this stage are:

1. Load and verify the preprocessed dataset.
2. Understand the structure, data types, and missing values.
3. Examine the statistical characteristics of the dataset.
4. Analyse the distribution of property prices.
5. Investigate whether a logarithmic transformation is appropriate for the target.
6. Study correlations between numeric features.
7. Examine the relationship between price, area, bedrooms, and property type.
8. Compare property prices across localities.
9. Visualise property listings geographically.
10. Identify potential multicollinearity, categorical-encoding requirements, and data-quality issues.
11. Prepare a set of observations to guide subsequent feature engineering and model development.

---

## 2. Location of the EDA Files

The EDA analysis is performed in the following notebook:

`notebooks/03_eda.ipynb`

The notebook loads the dataset produced during the preprocessing stage.

### Input dataset

`data/processed/clean_v2.csv`

This file contains the cleaned property listings used for EDA.

The dataset contains 7,457 rows and 18 columns before the additional EDA helper columns are created.

The notebook creates two additional columns for analysis:

* `price_per_sqft`
* `log_price`

Therefore, the DataFrame is expected to contain 7,457 rows and 20 columns after these helper columns have been created, provided no rows have been added or removed.

### Output directory

Figures are saved in:

`reports/figures/`

The interactive geographical map is saved in:

`reports/price_map.html`

These outputs provide visual evidence of the analysis and can be included in the final project report.

---

## 3. Dataset Overview

### 3.1 Dataset description

The dataset contains property listings describing residential properties offered for sale in Gurgaon.

Each row represents a property listing, while each column describes an attribute of that listing.

The available information includes property characteristics, location information, geographic coordinates, and the sale price.

The cleaned dataset used in EDA has the following dimensions:

| Attribute                                      | Value                         |
| ---------------------------------------------- | ----------------------------- |
| Dataset file                                   | `clean_v2.csv`                |
| Location                                       | `data/processed/clean_v2.csv` |
| Number of listings                             | 7,457                         |
| Original number of columns in the cleaned file | 18                            |
| Additional EDA helper columns                  | 2                             |
| Expected DataFrame columns after setup         | 20                            |
| Target variable                                | `price_lakhs`                 |
| Geographic focus                               | Gurgaon, India                |

The 7,457-row count refers to the cleaned dataset used for EDA, not the original raw dataset.

### 3.2 Important columns

The principal columns used during EDA are described below.

| Column                 | Description                                        | Role in EDA                            |
| ---------------------- | -------------------------------------------------- | -------------------------------------- |
| `prop_id`              | Property listing identifier                        | Identification and traceability        |
| `latitude`             | Latitude of the listing                            | Geographic analysis                    |
| `longitude`            | Longitude of the listing                           | Geographic analysis                    |
| `property_type`        | Type of property                                   | Comparison between property categories |
| `owntype`              | Encoded ownership-type category                    | Categorical analysis                   |
| `bedroom_num`          | Number of bedrooms                                 | Relationship with price                |
| `bathroom_num`         | Number of bathrooms                                | Relationship with price                |
| `balcony_num`          | Number of balconies                                | Property-characteristic analysis       |
| `furnish`              | Encoded furnishing category                        | Categorical analysis                   |
| `facing`               | Encoded facing-direction category                  | Categorical analysis                   |
| `age`                  | Encoded property-age category                      | Categorical analysis                   |
| `floor_num`            | Floor number                                       | Property-characteristic analysis       |
| `total_floor`          | Total number of floors                             | Property-characteristic analysis       |
| `total_landmark_count` | Count of landmarks represented in the dataset      | Exploratory feature analysis           |
| `area_sqft`            | Selected property area in square feet              | Relationship with price                |
| `area_is_carpet`       | Indicator associated with the selected area source | Area-quality analysis                  |
| `locality_grouped`     | Grouped locality category                          | Locality-level price analysis          |
| `price_lakhs`          | Sale price in lakhs of Indian rupees               | Target variable                        |
| `price_per_sqft`       | Derived price per square foot                      | Exploratory analysis only              |
| `log_price`            | Log-transformed sale price                         | Distribution and correlation analysis  |

**Note:** The meanings of the encoded values in `owntype`, `furnish`, `facing`, and `age` must be interpreted using the original dataset's category definitions. The numeric codes should not automatically be interpreted as continuous measurements.

The exact meaning and calculation of `total_landmark_count` should likewise follow the preprocessing documentation.

### 3.3 Target variable

The target variable is:

`price_lakhs`

It represents the sale price of a property in lakhs of Indian rupees.

For example:

* 100 lakhs = ₹1 crore.
* 250 lakhs = ₹2.5 crore.
* 500 lakhs = ₹5 crore.

The prediction task is to estimate this sale price using the available property and location characteristics.

During EDA, a logarithmic version of the target is also created to investigate the effect of price skewness.

---

# 4. Part A — EDA Setup and Dataset Verification

## 4.1 Loading the dataset

The first step is to load the cleaned dataset into a pandas DataFrame.

The notebook uses:

```python
df = pd.read_csv("../data/processed/clean_v2.csv")
```

The relative path works when the notebook's working directory is `notebooks`.

Loading the data into a DataFrame allows the dataset to be inspected and analysed using pandas, NumPy, Matplotlib, and Seaborn.

## 4.2 Creating the figure directory

The notebook creates the directory used to store the generated figures:

```python
os.makedirs("../reports/figures", exist_ok=True)
```

The `exist_ok=True` argument prevents an error if the directory already exists.

The figures generated during EDA are saved in this directory so they can be reused in the project report without having to recreate them manually.

## 4.3 Checking the dataset dimensions

The dataset dimensions are checked using:

```python
print(df.shape)
```

Expected output:

```text
(7457, 18)
```

This is the initial shape before the two EDA helper columns are created.

The shape is useful for confirming that the correct dataset has been loaded.

If the dimensions differ, the input file or the preprocessing output should be checked before continuing.

## 4.4 Inspecting the first rows

The first five records are inspected using:

```python
display(df.head())
```

This provides a quick overview of the values stored in each column.

It helps verify that:

* The expected columns are present.
* Prices and areas appear in their intended units.
* Property categories are populated.
* Geographic coordinates are available.
* The dataset was loaded with the expected structure.

============================================================
PART A — DATASET VERIFICATION
============================================================

Dataset shape:
(7457, 20)

First five rows:
prop_id	latitude	longitude	property_type	owntype	bedroom_num	bathroom_num	balcony_num	furnish	facing	age	floor_num	total_floor	total_landmark_count	area_sqft	area_is_carpet	locality_grouped	price_lakhs	price_per_sqft	log_price
0	O71585872	28.406534	76.962792	Residential Apartment	1	4.0	4	4.0	4	1	2	14.0	26.0	10.0	3434.0	0	Sector 84 Gurgaon	263.0	7658.707047	5.575949
1	T71585466	28.388907	76.951328	Residential Apartment	1	4.0	4	3.0	4	1	1	7.0	29.0	9.0	2870.0	0	Sector 81 Gurgaon	360.0	12543.554007	5.888878
2	I71493958	28.518677	77.015115	Residential Apartment	1	3.0	3	3.0	4	0	1	14.0	28.0	8.0	2802.0	0	Sector 112 Gurgaon	385.0	13740.185582	5.955837
3	X71361930	28.478440	76.986110	Residential Apartment	1	3.0	4	4.0	4	2	1	4.0	27.0	14.0	2290.0	0	Sector 104 Gurgaon	195.0	8515.283843	5.278115
4	N71483500	28.416400	77.012620	Residential Apartment	1	2.0	2	3.0	2	3	6	12.0	30.0	13.0	1400.0	0	Sector 74 Gurgaon	162.0	11571.428571	5.093750


Figure A1: First five records of the cleaned real-estate dataset.

## 4.5 Checking data types

The data types are inspected using:

```python
df.info()
```

This reveals the number of non-null values and the data type of each column.

Data-type inspection is important because a column may contain numeric-looking values that should actually be treated as categories.

For example, the values in `furnish`, `facing`, `owntype`, and `age` are encoded categories. Treating these values as ordinary continuous measurements could lead to misleading conclusions during correlation analysis.

## 4.6 Checking missing values

Missing values are counted using:

```python
print(df.isnull().sum())
```

The result identifies columns that contain missing data.

A missing value does not always require a row to be deleted. The appropriate treatment depends on the feature, the number of missing values, and the purpose of the analysis.

During the EDA setup, three missing values in `balcony_num` are filled using the median.

The median is a measure of central tendency that is less affected by unusually high or low values than the mean.

The operation is:

```python
df["balcony_num"] = df["balcony_num"].fillna(
    df["balcony_num"].median()
)
```

This fills missing balcony counts with the median of the available values.

It is a practical treatment for exploratory analysis, although the final modelling pipeline should handle missing values in a way that is fitted only on the training data.

## 4.7 Converting encoded categories for EDA

The following columns are converted to strings:

```python
cat_codes = ["owntype", "furnish", "facing", "age"]

df[cat_codes] = df[cat_codes].astype(str)
```

This conversion makes it clear that these columns represent category codes rather than ordinary continuous numeric variables.

For example, a furnishing code of `2` is not necessarily twice as much furnishing as a code of `1`.

Similarly, a facing-direction code should not be interpreted as a numerical magnitude.

Converting these columns to strings helps prevent them from being accidentally included in analyses that assume continuous numerical meaning.

The conversion does not decode the categories into their original descriptions. That requires the dataset's category definitions.

## 4.8 Creating EDA helper columns

Two additional columns are created for exploratory analysis.

### A. Price per square foot

```python
df["price_per_sqft"] = (
    df["price_lakhs"] * 100000 / df["area_sqft"]
)
```

The price is stored in lakhs, so it is multiplied by 100,000 to convert it to rupees.

The result is divided by the property area in square feet.

The formula is:

$$
\text{Price per Sq Ft}
=
\frac{\text{Price in Lakhs}\times100000}
{\text{Area in Sq Ft}}
$$

This derived measure makes it possible to compare property prices relative to their areas.

A larger property may have a higher total price but a lower price per square foot. Therefore, total price and price per square foot provide different perspectives on the market.

**Important:** `price_per_sqft` is calculated directly from the target variable and area. It is useful for EDA but must not be used as a predictor when training a model to estimate the same property's sale price. Doing so would introduce target leakage.

### B. Log-transformed price

```python
df["log_price"] = np.log1p(df["price_lakhs"])
```

The `np.log1p()` function calculates:

$$
\text{log\_price}=\ln(1+\text{price\_lakhs})
$$

This transformation is investigated because real-estate prices can be strongly right-skewed.

A small number of expensive properties can stretch the distribution toward higher prices, making the distribution difficult to visualise and potentially influencing some modelling methods.

The logarithmic transformation compresses large values more strongly than small values.

It does not remove the expensive properties. Instead, it changes the scale used to represent their prices.

The target remains `price_lakhs` in the original cleaned dataset. `log_price` is an additional EDA variable used to investigate an alternative target scale.

## 4.9 Verifying the prepared DataFrame

After creating the helper columns, the notebook should verify the DataFrame again:

```python
print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
```

Expected shape:

```text
(7457, 20)
```

This assumes the notebook is using the original 7,457 rows and has created both helper columns.

Data types:
prop_id                     str
latitude                float64
longitude               float64
property_type               str
owntype                     str
bedroom_num             float64
bathroom_num              int64
balcony_num             float64
furnish                     str
facing                      str
age                         str
floor_num               float64
total_floor             float64
total_landmark_count    float64
area_sqft               float64
area_is_carpet            int64
locality_grouped            str
price_lakhs             float64
price_per_sqft          float64
log_price               float64
dtype: object

Missing values:
...


Insert the actual output of the final shape, missing-value count, and data-type checks.

Recommended caption:

**Table A1: Dataset dimensions, missing values, and data types after EDA setup.**

---

# 5. Part B — Descriptive Statistical Analysis

## 5.1 Purpose

Descriptive statistics provide a numerical summary of the dataset.

Before studying relationships between features, it is useful to understand the typical values, spread, and range of the numerical variables.

This step examines properties such as price, area, bedrooms, bathrooms, and other numeric attributes.

## 5.2 Generating descriptive statistics

The notebook uses:

```python
display(df.describe().T)
```

The `describe()` function calculates summary statistics for numerical columns.

The `.T` operation transposes the result so that each variable appears as a row, making the table easier to read.

The resulting statistics include:

| Statistic          | Meaning                            |
| ------------------ | ---------------------------------- |
| Count              | Number of non-missing observations |
| Mean               | Arithmetic average                 |
| Standard deviation | Measure of spread around the mean  |
| Minimum            | Smallest observed value            |
| 25%                | First quartile                     |
| 50%                | Median                             |
| 75%                | Third quartile                     |
| Maximum            | Largest observed value             |

The mean and median provide different views of the typical property.

When the mean is substantially larger than the median, this can indicate a right-skewed distribution in which relatively expensive properties influence the average.

The standard deviation and quartiles help describe how widely property prices and sizes vary.

## 5.3 Examining property types

The distribution of property types is examined using:

```python
display(df["property_type"].value_counts())
```

This shows how many listings belong to each property category.

Property types can have different pricing patterns because apartments, builder floors, houses, and land may differ in size, facilities, ownership arrangements, and market demand.

However, the number of listings in each category must be considered when interpreting comparisons.

A category containing very few records may produce an unstable estimate of its typical price.

## 5.4 Examining encoded categories

The notebook also checks the frequency of the coded categories:

```python
for col in cat_codes:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))
```

This helps identify common and rare category codes.

The original category definitions should be consulted before interpreting what each code means.

A category code occurring frequently indicates that it is common in the dataset, but frequency alone does not establish that the category increases or decreases prices.


============================================================
PART B — DESCRIPTIVE STATISTICS
============================================================
count	mean	std	min	25%	50%	75%	max
latitude	7457.0	28.437972	0.041522	28.350930	28.406140	28.425181	28.472552	28.534810
longitude	7457.0	77.040553	0.052859	76.810760	76.997421	77.054558	77.084474	77.191500
bedroom_num	7457.0	3.378839	0.768965	1.000000	3.000000	3.000000	4.000000	8.000000
bathroom_num	7457.0	3.635778	1.020496	0.000000	3.000000	4.000000	4.000000	9.000000
balcony_num	7457.0	2.982433	0.875781	0.000000	2.000000	3.000000	4.000000	4.000000
floor_num	6703.0	7.519171	6.602827	0.000000	2.000000	5.000000	11.000000	48.000000
total_floor	7455.0	15.211133	11.785841	0.000000	4.000000	14.000000	24.000000	51.000000
total_landmark_count	7455.0	22.166063	16.108926	1.000000	10.000000	14.000000	44.000000	50.000000
area_sqft	7457.0	2501.795235	1109.578017	594.000000	1777.000000	2217.000000	2872.000000	10000.000000
area_is_carpet	7457.0	0.255062	0.435925	0.000000	0.000000	0.000000	1.000000	1.000000
price_lakhs	7457.0	347.218318	307.582371	100.000000	175.000000	250.000000	405.000000	4500.000000
price_per_sqft	7457.0	13144.320574	6337.273591	3395.061728	9090.909091	11626.898048	15000.000000	57547.587428
log_price	7457.0	5.631945	0.614008	4.615121	5.170484	5.525453	6.006353	8.412055


Insert the actual output of `df.describe().T`.

Recommended caption:

**Table B1: Descriptive statistics of the cleaned real-estate dataset.**

Property type counts:
property_type
Residential Apartment        4750
Independent/Builder Floor    2274
Independent House/Villa       433
Name: count, dtype: int64

Category counts: owntype
owntype
1    7375
2      43
3      31
4       8
Name: count, dtype: int64

Category counts: furnish
furnish
4    5764
2     940
1     509
0     244
Name: count, dtype: int64

Category counts: facing
facing
5    2500
3    1728
1    1045
0     833
4     468
6     269
7     262
2     181
8     171
Name: count, dtype: int64

Category counts: age
age
1    2447
6    2059
5    1255
2    1157
3     538
0       1
Name: count, dtype: int64

Insert the property-type counts and the category-code frequency tables.

Recommended caption:

**Table B2: Distribution of property types and encoded categorical variables.**

## 5.5 Main observations to record

After examining the tables, record:

1. The mean and median sale price.
2. The minimum and maximum sale price.
3. The mean and median property area.
4. The range of bedroom and bathroom counts.
5. The most common property type.
6. Any property categories containing relatively few listings.
7. Any unusual or unexpected category values.

These observations establish a numerical baseline for the subsequent visual analysis.

---

# 6. Part C — Analysis of Price Distributions

## 6.1 Purpose

Price distributions help explain how property prices are spread across the dataset.

The analysis examines three measures:

1. Sale price in lakhs.
2. Log-transformed sale price.
3. Price per square foot.

Histograms are used to understand the shape of each distribution.

This is particularly important because real-estate prices often vary considerably, even within the same city.

## 6.2 Distribution of original sale prices

The first histogram uses `price_lakhs`.

The x-axis represents sale price in lakhs, while the y-axis represents the number of listings within each price interval.

This plot helps reveal whether most properties fall within a relatively narrow price range or whether the dataset contains a long tail of expensive properties.

A right-skewed distribution typically has most observations concentrated at lower values and a tail extending toward higher values.

Such a pattern can occur when a dataset contains many moderately priced properties and a smaller number of luxury properties.

The histogram is saved as part of:

`reports/figures/01_price_distribution.png`

## 6.3 Distribution of log-transformed prices

The second histogram uses `log_price`.

The transformation compresses the larger price values, allowing the distribution to be examined on a different scale.

The purpose is not to assume that the transformed distribution will be normal. Instead, it allows the analyst to assess whether the transformation produces a more balanced distribution.

If the original prices are strongly right-skewed and the log-transformed values are less skewed, this supports evaluating a logarithmic target during model development.

A histogram alone cannot establish which target produces the best predictive model. That decision must also consider model evaluation results.

## 6.4 Distribution of price per square foot

The third histogram uses `price_per_sqft`.

This measure expresses the price relative to the property area.

It can reveal whether the dataset contains properties with unusually high or low prices relative to their size.

Such observations may arise from genuine market differences, property characteristics, location, or data-quality issues.

A high price per square foot should not automatically be considered an error.

For example, a relatively small property in a premium locality may have a high price per square foot while still having a lower total price than a large property elsewhere.

## 6.5 Comparing skewness and central tendency

The notebook examines skewness and compares means with medians.

Skewness measures the asymmetry of a distribution.

A positive skewness value generally indicates a longer right tail, while a value closer to zero indicates less asymmetry. A value near zero does not, by itself, prove that a distribution is normal.

Comparing the mean and median helps assess the influence of high-priced properties.

The following code can be used to generate a compact summary:

```python
distribution_summary = pd.DataFrame({
    "mean": [
        df["price_lakhs"].mean(),
        df["log_price"].mean(),
        df["price_per_sqft"].mean()
    ],
    "median": [
        df["price_lakhs"].median(),
        df["log_price"].median(),
        df["price_per_sqft"].median()
    ],
    "skewness": [
        df["price_lakhs"].skew(),
        df["log_price"].skew(),
        df["price_per_sqft"].skew()
    ]
}, index=[
    "price_lakhs",
    "log_price",
    "price_per_sqft"
])

display(distribution_summary.round(3))
```

<img width="3173" height="768" alt="image" src="https://github.com/user-attachments/assets/cbc7010a-c6bb-4709-8199-4386bcb51253" />



Insert the image:

`reports/figures/01_price_distribution.png`

Recommended caption:

**Figure C1: Distribution of original sale prices, log-transformed prices, and price per square foot.**


mean	median	skewness
price_lakhs	347.218	250.000	4.203
log_price	5.632	5.525	0.769
price_per_sqft	13144.321	11626.898	2.211


Insert the output of `distribution_summary`.

Recommended caption:

**Table C1: Mean, median, and skewness of the three price-related variables.**

## 6.6 Interpretation and target decision

Use the actual histogram and summary statistics to answer:

* Is the original price distribution right-skewed?
* Does the log transformation reduce the observed skewness?
* Is the mean higher than the median?
* Does price per square foot contain a long tail?
* Is there evidence that a logarithmic target is worth evaluating?

### Preliminary conclusion

A logarithmic target is worth considering when the original price distribution is strongly right-skewed and the transformation improves the behaviour of the target.

However, this EDA stage does not establish that the logarithmic target will outperform the original target in every model.

The decision will be evaluated during model development.

---

# 7. Part D — Correlation Analysis

## 7.1 Purpose

Correlation analysis measures the strength and direction of relationships between numerical variables.

In this project, correlation analysis is used to investigate:

* How area relates to property price.
* Whether bedroom and bathroom counts are associated with price.
* Whether other numerical features have relationships with the target.
* Whether some predictors are strongly related to one another.

The last point is important because highly related predictors may contain overlapping information.

## 7.2 Selecting appropriate columns

The correlation analysis uses:

```python
num_cols = [
    "log_price",
    "area_sqft",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_is_carpet",
    "latitude",
    "longitude"
]
```

The encoded categorical variables `owntype`, `furnish`, `facing`, and `age` are excluded from this Pearson correlation analysis because their numeric codes do not necessarily represent meaningful numerical distances.

The `price_per_sqft` column is also excluded.

This is because price per square foot is calculated directly from the target price and property area. Including it in target-correlation analysis could create a misleading interpretation of its relationship with price.

## 7.3 Checking missing values

Before calculating the correlation matrix, the notebook checks the selected columns for missing values:

```python
print(df[num_cols].isnull().sum())
```

Some columns may contain missing observations.

In the current dataset, earlier checks found:

* `floor_num`: 754 missing values.
* `total_floor`: 2 missing values.
* `total_landmark_count`: 2 missing values.

The exact values should be confirmed using the current notebook output.

Pandas calculates each pairwise correlation using observations available for that pair. Therefore, the entire dataset does not need to be imputed or reduced to complete cases solely to create the correlation matrix.

However, the varying number of observations used for different pairs should be kept in mind when interpreting the results.

## 7.4 Calculating the correlation matrix

The Pearson correlation matrix is calculated using:

```python
corr = df[num_cols].corr(method="pearson")
```

Pearson correlation ranges from -1 to +1.

| Correlation | General interpretation              |
| ----------- | ----------------------------------- |
| Close to +1 | Strong positive linear relationship |
| Close to 0  | Weak or absent linear relationship  |
| Close to -1 | Strong negative linear relationship |

A positive correlation means that larger values of one variable tend to be associated with larger values of the other.

A negative correlation means that larger values of one variable tend to be associated with smaller values of the other.

Correlation does not establish causation, and a low Pearson correlation does not rule out a nonlinear relationship.

## 7.5 Creating the correlation heatmap

The notebook displays the correlation matrix as a heatmap.

The upper triangle is masked to avoid displaying the same pairwise correlations twice.

The resulting figure is saved as:

`reports/figures/02_correlation_heatmap.png`

<img width="2087" height="1779" alt="image" src="https://github.com/user-attachments/assets/a78c7c0e-51c3-4a07-81a4-4bdca8af9af6" />


Insert the image:

`reports/figures/02_correlation_heatmap.png`

Recommended caption:

**Figure D1: Pearson correlation heatmap of the selected numerical features and log-transformed sale price.**

## 7.6 Identifying features correlated with the target

The notebook sorts the correlations with `log_price`:

```python
display(
    corr["log_price"]
    .drop("log_price")
    .sort_values(ascending=False)
)
```

This produces a ranked list of the selected numerical variables based on their Pearson correlation with log-transformed price.

The result helps identify variables with stronger positive or negative linear associations with the target.

Correlation with log_price:
area_sqft               0.752432
bathroom_num            0.639590
bedroom_num             0.623382
longitude               0.469302
floor_num               0.189461
balcony_num             0.188906
latitude                0.157510
total_landmark_count    0.098751
total_floor             0.089147
area_is_carpet         -0.015366
Name: log_price, dtype: float64

Predictor correlations:
area_sqft	bedroom_num	bathroom_num
area_sqft	1.000	0.657	0.658
bedroom_num	0.657	1.000	0.807
bathroom_num	0.658	0.807	1.000


Insert the actual sorted correlation output.

Recommended caption:

**Table D1: Pearson correlations between log-transformed price and the selected numerical features.**

Do not state that area, bathrooms, or bedrooms have the strongest correlations unless the actual output supports that conclusion.

## 7.7 Checking potential multicollinearity

Multicollinearity occurs when two or more predictors are strongly related to one another.

In this project, area, bedroom count, and bathroom count are worth examining because larger properties may have more bedrooms and bathrooms.

The following code identifies pairs with an absolute Pearson correlation greater than 0.8:

```python
predictor_corr = df[
    [
        "area_sqft",
        "bedroom_num",
        "bathroom_num",
        "balcony_num",
        "floor_num",
        "total_floor",
        "total_landmark_count"
    ]
].corr()

threshold = 0.8

for i in range(len(predictor_corr.columns)):
    for j in range(i + 1, len(predictor_corr.columns)):

        value = predictor_corr.iloc[i, j]

        if pd.notna(value) and abs(value) > threshold:
            print(
                predictor_corr.columns[i],
                "and",
                predictor_corr.columns[j],
                "correlation:",
                round(value, 3)
            )
```

The threshold is a screening rule, not a universal definition of multicollinearity.

Pairs with weaker correlations may still be related, while strongly correlated predictors do not always need to be removed.

The final decision should depend on the modelling algorithm, feature meaning, validation performance, and other diagnostics.

## 7.8 Limitations of correlation analysis

The correlation matrix does not include every important property attribute.

In particular, categorical features and locality may have meaningful relationships with price that are not represented by this numeric Pearson correlation matrix.

Correlation also measures linear relationships rather than every possible pattern.

Therefore, correlation analysis is used together with the distributions, boxplots, locality summaries, and geographical maps.

---

# 8. Part E — Price Versus Area, Bedrooms, and Property Type

## 8.1 Purpose

This section investigates how property prices vary with selected property characteristics.

The analysis focuses on:

1. Property area.
2. Number of bedrooms.
3. Property type.
4. Additional categorical characteristics.

Scatterplots and boxplots are used because they provide different types of information.

A scatterplot shows the relationship between two numerical variables. A boxplot summarises the distribution of a numerical variable within different categories.

---

## 8.2 E1 — Relationship Between Price and Area

### Objective

Property area is a potential predictor of sale price.

A larger property may have a higher total sale price, but the relationship may not be perfectly linear.

The relationship can also vary across locations and property types.

### Scatterplot using the original price scale

The first scatterplot uses:

* X-axis: `area_sqft`
* Y-axis: `price_lakhs`

Each point represents a property listing.

The plot helps identify whether larger properties generally tend to have higher sale prices and whether the relationship contains unusual patterns.

A wide vertical spread of points at similar areas would indicate that properties of similar sizes can have different prices.

This is plausible in real estate because size is not the only factor influencing price.

### Scatterplot using logarithmic axes

The second scatterplot uses logarithmic scales for area and price.

This can make relationships across a wider range of property sizes and prices easier to inspect.

It can also make multiplicative patterns more visible.

The log-scaled plot should be interpreted alongside the original-scale plot because both views provide useful information.

### Saved output

The figure is saved as:

`reports/figures/03a_price_vs_area.png`

<img width="2772" height="968" alt="image" src="https://github.com/user-attachments/assets/64203eca-9171-4f7c-b683-7f9b6034c32e" />


Insert:

`reports/figures/03a_price_vs_area.png`

Recommended caption:

**Figure E1: Relationship between property area and sale price using original and logarithmic scales.**

### Interpretation

Use the actual plots to record:

* Whether sale price generally increases with area.
* Whether the relationship appears approximately linear or nonlinear.
* Whether prices vary considerably among properties with similar areas.
* Whether the logarithmic view reveals a clearer pattern.
* Whether unusual observations require further investigation.

Do not assume that area alone explains the sale price.

---

## 8.3 E2 — Relationship Between Price and Bedroom Count

### Objective

Bedroom count is a categorical-like discrete variable in this analysis.

A property with more bedrooms may be larger or intended for a different market segment. However, bedroom count does not uniquely determine property area or price.

The analysis groups properties by the number of bedrooms and compares their price distributions.

### Boxplot

The notebook creates a boxplot of `price_lakhs` for each bedroom count.

The y-axis uses a logarithmic scale to make differences easier to examine when prices vary widely.

A boxplot displays:

* The median price.
* The middle 50% of observations.
* The spread of the central observations.
* Potential outliers.

Outliers shown by a boxplot are observations that fall outside the plot's standard whisker range. They are not automatically data errors.

### Summary statistics

The notebook also calculates statistics grouped by bedroom count, including the number of listings, median price, mean price, and price range.

This allows the numerical results to be compared with the boxplot.

### Saved output

`reports/figures/03b_price_vs_bedrooms.png`

<img width="1768" height="968" alt="image" src="https://github.com/user-attachments/assets/7b7709c3-bd26-4a79-a631-5e8d1f84b156" />


Insert:

`reports/figures/03b_price_vs_bedrooms.png`

Recommended caption:

**Figure E2: Distribution of sale prices across bedroom counts.**

Predictor correlations:
area_sqft	bedroom_num	bathroom_num
area_sqft	1.000	0.657	0.658
bedroom_num	0.657	1.000	0.807
bathroom_num	0.658	0.807	1.000


Insert the actual grouped bedroom-price summary.

Recommended caption:

**Table E1: Sale-price statistics grouped by bedroom count.**

### Interpretation

Use the results to answer:

* Does median price generally increase with bedroom count?
* Do categories overlap considerably?
* Are some bedroom categories represented by relatively few listings?
* Does the mean differ substantially from the median?
* Are there unusually high or low prices within particular categories?

A general increase in median price would indicate an association in this dataset, not proof that increasing bedroom count alone causes prices to increase.

---

## 8.4 E3 — Price Differences Between Property Types

### Objective

Different property types may have different price distributions.

For example, apartments, builder floors, houses, and villas can differ in their areas, facilities, locations, and ownership arrangements.

The purpose of this analysis is to compare their observed prices.

### Boxplot

The notebook creates a boxplot of `price_lakhs` grouped by `property_type`.

Property types with fewer than 30 listings are excluded from this particular plot to reduce the risk of interpreting very small groups as representative distributions.

This plotting threshold is an EDA choice. It does not automatically mean that the excluded property types should be removed from the dataset or model.

The categories are ordered by their observed median prices, and the price axis uses a logarithmic scale.

### Summary statistics

The analysis also calculates:

* Number of listings.
* Median price.
* Mean price.
* Minimum price.
* Maximum price.

The median is useful because it is less sensitive than the mean to extremely high prices.

### Saved output

`reports/figures/03c_price_by_property_type.png`

<img width="1991" height="968" alt="image" src="https://github.com/user-attachments/assets/9eede19c-f509-4d07-8983-abe588c63c82" />


Insert:

`reports/figures/03c_price_by_property_type.png`

Recommended caption:

**Figure E3: Sale-price distributions across property types.**

<img width="1991" height="968" alt="image" src="https://github.com/user-attachments/assets/b98d4f17-458e-4bc7-b16d-2ca97d2a48f5" />


Insert the actual property-type summary table.

Recommended caption:

**Table E2: Descriptive statistics of sale prices grouped by property type.**

### Interpretation

Record:

* Which property types have the highest and lowest observed median prices.
* Whether the distributions overlap.
* Which property types contain relatively few listings.
* Whether the mean and median differ substantially.
* Whether the results suggest that property type should be retained as a categorical predictor.

The observed ranking is specific to this dataset and should not automatically be generalised to all properties in Gurgaon.

---

## 8.5 E4 — Other Categorical Characteristics

The analysis also examines additional characteristics:

* Furnishing category.
* Property-age category.
* Area-source indicator represented by `area_is_carpet`.

These features may help explain differences between listings that have similar areas or bedroom counts.

### Furnishing category

The sale-price distribution is compared across the observed furnishing categories.

This can reveal whether furnished, semi-furnished, and unfurnished properties, or their corresponding encoded categories, have different observed price distributions.

The original category definitions must be checked before assigning descriptive labels to the codes.

### Property-age category

The sale-price distribution is compared across the observed property-age categories.

Property age may be associated with condition, construction type, maintenance, or other characteristics. The plot can reveal associations but cannot establish why the differences occur.

The encoded values must be interpreted using the original dataset documentation.

### Area-source indicator

The price per square foot is compared across the values of `area_is_carpet`.

This analysis helps examine whether the source used for the selected area measurement is associated with differences in calculated price per square foot.

The meaning of the indicator should be confirmed from the preprocessing code before drawing a detailed conclusion.

### Saved output

`reports/figures/03d_other_categoricals.png`

<img width="3168" height="968" alt="image" src="https://github.com/user-attachments/assets/88e0277b-07d9-4425-aa06-e1b3abc28222" />


Insert:

`reports/figures/03d_other_categoricals.png`

Recommended caption:

**Figure E4: Exploratory comparisons of price by furnishing and property-age categories, and price per square foot by area-source indicator.**

### Overall interpretation of Part E

Part E helps determine whether property prices vary across important property characteristics.

Area, bedroom count, bathroom count, property type, furnishing, and age are plausible predictors to investigate during modelling.

However, the actual strength and usefulness of each feature must be established through the observed EDA results and subsequent model evaluation.

---

# 9. Part F — Locality-Level Price Analysis

## 9.1 Purpose

Location is an important consideration in real-estate analysis.

Properties with similar areas and room counts may have different prices because they are situated in different localities.

This section compares prices across the grouped localities represented in the dataset.

The analysis examines both:

1. Median total sale price.
2. Median price per square foot.

These measures describe different aspects of the market.

## 9.2 Grouping by locality

The notebook groups listings by `locality_grouped`.

For each locality, it calculates:

* Number of listings.
* Median sale price.
* Mean sale price.
* Median price per square foot.

A minimum of 15 listings is required for a locality to appear in the ranking.

The purpose of this threshold is to avoid ranking localities using extremely small groups, for which a few observations could dominate the result.

The `Other` category is excluded from the locality ranking because it combines less frequent localities rather than representing one specific geographical area.

These decisions affect the ranking, so the threshold and exclusion should be documented when interpreting the results.

## 9.3 Why median price is used

The median represents the middle observation after prices are sorted.

It is less sensitive than the mean to a small number of unusually expensive properties.

This makes it useful for comparing the typical observed price across localities.

The median does not remove the influence of all differences between properties. For example, one locality may contain larger homes than another.

Therefore, locality-level total price and price per square foot should be interpreted together.

## 9.4 Top and bottom localities by median sale price

The first ranking identifies localities with the highest and lowest median sale prices.

The corresponding figure is saved as:

`reports/figures/04a_locality_median_price.png`

<img width="2982" height="1168" alt="image" src="https://github.com/user-attachments/assets/210ca182-8879-4b5d-961d-be347e8b71cf" />


Insert:

`reports/figures/04a_locality_median_price.png`

Recommended caption:

**Figure F1: Ten highest- and lowest-ranked eligible localities by median sale price.**

The figure compares localities with at least 15 listings, excluding `Other`.

## 9.5 Top and bottom localities by median price per square foot

The second ranking compares localities by median price per square foot.

The figure is saved as:

`reports/figures/04b_locality_median_pps.png`

<img width="2982" height="1168" alt="image" src="https://github.com/user-attachments/assets/4fad443d-ea49-4b3a-91a9-9a1ca4d7cc62" />


Insert:

`reports/figures/04b_locality_median_pps.png`

Recommended caption:

**Figure F2: Ten highest- and lowest-ranked eligible localities by median price per square foot.**

This ranking helps compare the observed price relative to property size.

A locality can have a high median total price because its properties tend to be larger, while its median price per square foot may be less exceptional.

Conversely, smaller properties in a premium locality can have high prices per square foot without having the highest total sale prices.

## 9.6 Comparing the two rankings

The two rankings should be compared to identify localities that appear in both lists or have noticeably different positions.

Record:

* Which eligible localities have the highest observed median sale prices.
* Which have the lowest observed median sale prices.
* Which have the highest median price per square foot.
* Which have the lowest median price per square foot.
* Whether the two rankings identify similar or different localities.
* Whether the listing counts are large enough to make the comparisons informative.

Eligible localities: 96

Top 10 by median price:
listings	median_price	mean_price	median_pps
locality_grouped				
Sector 42 Gurgaon	46	2500.0	2132.63	42980.30
Sector 33 Gurgaon	24	987.5	718.92	20151.52
Sector 54 Gurgaon	216	725.0	821.53	24843.52
MG Road	18	650.5	666.83	22222.22
Sector 53 Gurgaon	81	549.0	574.06	19387.76
Sector 112 Gurgaon	56	545.5	598.80	12486.95
Sector 26 Gurgaon	24	512.5	615.00	16228.24
Sector 59 Gurgaon	142	502.5	503.30	18247.25
DLF Phase 5	100	499.5	454.25	18716.64
DLF Phase 3	67	451.0	626.57	13280.21

Bottom 10 by median price:
listings	median_price	mean_price	median_pps
locality_grouped				
Sector 95 Gurgaon	22	115.0	130.82	6791.81
Sector 68 Gurgaon	59	120.0	132.00	8976.38
Sector 89 Gurgaon	15	125.0	124.67	8569.30
Sector 37C Gurgaon	47	125.0	132.81	6916.43
Sector 103 Gurgaon	52	126.5	142.27	7811.96
Sector 71 Gurgaon	42	127.5	133.31	6936.38
Sector 93 Gurgaon	50	137.0	135.80	9017.73
Sector 107 Gurgaon	25	140.0	149.88	7514.15
Sector 82 Gurgaon	72	141.0	198.06	7266.07
Sector 90 Gurgaon	68	145.5	149.84	7683.63

Top 10 by median price per square foot:
listings	median_price	mean_price	median_pps
locality_grouped				
Sector 42 Gurgaon	46	2500.0	2132.63	42980.30
Sector 54 Gurgaon	216	725.0	821.53	24843.52
MG Road	18	650.5	666.83	22222.22
Sector 33 Gurgaon	24	987.5	718.92	20151.52
Sector 53 Gurgaon	81	549.0	574.06	19387.76
DLF Phase 5	100	499.5	454.25	18716.64
sector 58 Gurgaon	28	410.0	425.96	18499.21
Sector 59 Gurgaon	142	502.5	503.30	18247.25
Sector 62 Gurgaon	115	423.0	509.71	16781.84
Sector 108 Gurgaon	107	325.0	321.65	16561.60

Bottom 10 by median price per square foot:
listings	median_price	mean_price	median_pps
locality_grouped				
Sector 95 Gurgaon	22	115.0	130.82	6791.81
Sector 37C Gurgaon	47	125.0	132.81	6916.43
Sector 71 Gurgaon	42	127.5	133.31	6936.38
Sector 82 Gurgaon	72	141.0	198.06	7266.07
Sector 84 Gurgaon	81	160.0	175.46	7314.29
Sector 107 Gurgaon	25	140.0	149.88	7514.15
Sector 91 Gurgaon	40	162.0	172.42	7545.65
Sector 90 Gurgaon	68	145.5	149.84	7683.63
Sector 99A Gurgaon	18	150.0	146.28	7686.53
Sector 103 Gurgaon	52	126.5	142.27	7811.96


Insert the actual locality summary table, including listing count, median price, mean price, and median price per square foot.

Recommended caption:

**Table F1: Locality-level price statistics used to create the top and bottom rankings.**

Do not fill the locality names or prices in this README until the notebook outputs have been checked.

### Interpretation

Locality-level differences support investigating location as a predictor in the next stage of the project.

However, these comparisons do not isolate the effect of location from property size, property type, furnishing, or other differences between the listings.

The results establish observed differences in the dataset, not causal effects.

---

# 10. Part G — Geographic Distribution of Property Prices

## 10.1 Purpose

The locality analysis compares named locality groups. Geographic visualisation provides a complementary view by plotting the property listings using latitude and longitude.

This helps examine how listings are distributed spatially and whether properties with different prices appear in different areas.

The analysis uses two forms of visualisation:

1. A static scatterplot map.
2. An interactive Folium map.

The static plot is suitable for the project report, while the interactive map allows individual listings to be inspected.

## 10.2 Checking geographic coordinates

The notebook examines the latitude and longitude columns using descriptive statistics.

The checks include:

* Minimum latitude.
* Maximum latitude.
* Minimum longitude.
* Maximum longitude.
* Number of unique latitude values.
* Number of unique longitude values.

These checks help establish whether the coordinates fall within the expected geographical area.

Earlier checks showed coordinates within approximately:

| Coordinate |  Minimum |  Maximum |
| ---------- | -------: | -------: |
| Latitude   | 28.35093 | 28.53481 |
| Longitude  | 76.81076 | 77.19150 |

These are previously observed values and should be verified against the current dataset before being reported as final results.

## 10.3 Static geographic price map

The static map uses longitude on the x-axis and latitude on the y-axis.

Each point represents a property listing.

The first plot colours the points according to `log_price`.

The second plot colours them according to `price_per_sqft`.

This makes it possible to examine the spatial distribution of total prices and prices relative to area.

The price-per-square-foot colour scale is capped at the 98th percentile for visualisation. This prevents a small number of very high values from dominating the colour range.

This cap affects only the visual representation; it does not change the underlying price-per-square-foot values.

### Saved output

`reports/figures/05a_price_map_static.png`

<img width="2949" height="1168" alt="image" src="https://github.com/user-attachments/assets/f1e275de-fd3e-4d73-a3bf-2a30f2b09b92" />


Insert:

`reports/figures/05a_price_map_static.png`

Recommended caption:

**Figure G1: Geographic distribution of property listings coloured by log-transformed sale price and price per square foot.**

### Interpretation

Examine the figure to identify:

* Whether listings form visible geographic clusters.
* Whether higher- and lower-priced listings appear in different areas.
* Whether price-per-square-foot patterns differ from total-price patterns.
* Whether the same locations contain a wide range of prices.
* Whether the map appears to have any coordinate anomalies.

A scatterplot using coordinates is not a full street map. It displays the relative geographic positions of listings but does not automatically identify roads, neighbourhood boundaries, or specific landmarks.

## 10.4 Checking coordinate duplication

Multiple listings may share the same latitude and longitude.

This can happen when the dataset assigns one representative coordinate to a locality, when several listings refer to the same building or complex, or when coordinates are approximate.

Therefore, repeated coordinates do not automatically mean that the records are duplicates.

The notebook examines how many unique coordinate pairs occur within each grouped locality.

This check helps establish whether the coordinates distinguish individual properties or mainly represent approximate locality-level positions.

Coordinate summary:
latitude	longitude
count	7457.000000	7457.000000
mean	28.437972	77.040553
std	0.041522	0.052859
min	28.350930	76.810760
25%	28.406140	76.997421
50%	28.425181	77.054558
75%	28.472552	77.084474
max	28.534810	77.191500

Unique coordinate pairs: 1137

Localities with the fewest unique coordinate pairs:
listings	unique_coordinate_pairs
locality_grouped		
MG Road	18	2
Sector 26 Gurgaon	24	2
Sector 110 A Gurgaon	12	2
Sector 3 Gurgaon	24	2
Sector 36A Gurgaon	12	2
Sector 40 Gurgaon	21	2
Sector-33 Sohna	29	2
Sector 89 A Gurgaon	15	2
Sector 30 Gurgaon	14	3
Sector 4 Gurgaon	10	3
Sector 27 Gurgaon	10	3
Sector 33 Gurgaon	24	3
Sector 74 Gurgaon	51	3
Suncity	18	3
Sector 110 Gurgaon	28	3



Insert the actual output listing the localities with the fewest unique coordinate pairs.

Recommended caption:

**Table G1: Number of unique coordinate pairs within grouped localities.**

### Interpretation

If a locality contains many listings but only one or two unique coordinate pairs, its coordinates may not provide precise property-level locations.

If multiple localities share coordinates, this should also be investigated.

The geographical analysis must therefore consider the precision of the original coordinate data.

## 10.5 Interactive map using Folium

Folium is used to create an interactive HTML map.

Unlike the static plot, the interactive map can display additional details when the user hovers over a property marker.

The markers include information such as:

* Grouped locality.
* Sale price in lakhs.
* Area in square feet.
* Price per square foot.

The marker colours represent price per square foot.

To keep the interactive map manageable, up to 3,000 listings are sampled from the available records using a fixed random seed.

The fixed seed makes the sample reproducible when the underlying dataset remains unchanged.

The map is saved to:

`reports/price_map.html`

### Opening the map

From PowerShell:

```powershell
cd D:\real-estate-price-prediction
start reports\price_map.html
```

The map uses online background tiles, so the browser normally needs an internet connection to display the background map.

The property markers are generated from the dataset.

<img width="1063" height="667" alt="image" src="https://github.com/user-attachments/assets/6c7251cb-e45c-4b5d-a807-4a3f6ce1c9bf" />


Take a screenshot of the interactive map in your browser and insert it here.

Recommended caption:

**Figure G2: Interactive map of sampled property listings, with marker colours representing price per square foot.**

### Interpretation

Use the map to inspect:

* Geographic clusters of listings.
* Local differences in price per square foot.
* The distribution of prices within and between areas.
* Whether high-priced listings are concentrated in particular regions.
* Whether shared coordinates limit the precision of the displayed locations.

The interactive map complements the static figure but does not replace the numerical locality analysis.

If the background map does not load, the static scatterplot remains available for the report.

---

# 11. Part H — EDA Wrap-Up Analysis

## 11.1 Purpose

The final stage summarises the findings from the preceding sections.

It identifies candidate features, possible modelling concerns, and areas that require further investigation.

This summary provides a bridge between EDA and the next stage of the project.

## 11.2 Target variable

The original target is:

`price_lakhs`

The EDA also examines:

`log_price = np.log1p(price_lakhs)`

The transformation is investigated because it can reduce the influence of a strongly right-skewed price distribution.

The final target representation should be selected based on actual distribution results and subsequent model evaluation.

If the model predicts log-transformed prices, predictions must be transformed back to the original price scale before presenting estimated sale prices in lakhs.

For the transformation used here:

$$
\widehat{\text{price\_lakhs}}
=
\exp(\widehat{\text{log\_price}})-1
$$

The original units should also be used when calculating or reporting metrics intended to represent errors in lakhs or rupees.

## 11.3 Candidate important features

The EDA identifies several plausible predictors for further investigation.

| Feature            | Reason for investigation                   |
| ------------------ | ------------------------------------------ |
| `area_sqft`        | Represents the selected property area      |
| `bedroom_num`      | Describes the number of bedrooms           |
| `bathroom_num`     | Describes the number of bathrooms          |
| `property_type`    | Distinguishes property categories          |
| `locality_grouped` | Represents locality-level differences      |
| `furnish`          | Represents furnishing category             |
| `age`              | Represents property-age category           |
| `facing`           | Represents facing-direction category       |
| `owntype`          | Represents ownership-type category         |
| `latitude`         | Represents north-south geographic position |
| `longitude`        | Represents east-west geographic position   |

These are candidate features, not a final feature-selection result.

Their usefulness depends on their relationship with the target, data quality, redundancy, and performance during validation.

## 11.4 Potential multicollinearity

Area, bedroom count, and bathroom count may contain overlapping information.

For example, larger properties may tend to have more bedrooms and bathrooms.

The correlation matrix is used to screen for strongly related numerical predictors.

However, correlated predictors should not automatically be removed.

The appropriate decision depends on the model being used and whether the features provide useful predictive information.

Tree-based models and linear models can behave differently when predictors are correlated.

The issue should be reviewed during model development rather than resolved solely on the basis of a single correlation threshold.

## 11.5 Categorical features and encoding

The following features require categorical treatment or further verification of their meaning:

* `property_type`
* `furnish`
* `age`
* `facing`
* `owntype`
* `locality_grouped`

The encoding strategy should reflect the meaning of each feature.

For example, one-hot encoding can represent nominal categories without imposing an artificial numerical order.

An ordinal encoding may be appropriate only when a meaningful order exists and the category definitions support it.

The values in the encoded columns must be decoded or documented before an encoding strategy is chosen.

Encoding should be performed within the modelling pipeline where appropriate so that preprocessing is fitted on the training data rather than the full dataset.

## 11.6 Spatial importance

The locality-level rankings and geographic maps examine two different aspects of location.

The locality analysis compares typical prices across named locality groups.

The map examines the geographic distribution of the listings and their prices.

Together, these analyses help justify investigating location-related features in the next stage.

Possible follow-up work includes examining distances to relevant landmarks or amenities, provided the required data is available and the calculations can be validated.

The maps alone do not establish that location causes a particular price difference.

## 11.7 Data-quality observations

The EDA checks several potential data-quality issues:

1. Missing values in property characteristics.
2. Unusually small or large feature values.
3. Categories with very few listings.
4. Differences in the number of observations across property types.
5. Localities containing relatively few listings.
6. Repeated or approximate geographic coordinates.
7. Encoded category values whose meanings require documentation.
8. Price distributions containing extreme values.

These issues should be documented based on the actual notebook output.

A statistical outlier is not automatically an invalid record. A high price may reflect a genuine luxury property, and a repeated coordinate may represent multiple properties within the same building or locality.

Potential problems should be investigated before any records are removed or modified.

## 11.8 Final EDA observations

**[INSERT OUTPUT H1 HERE]**

Insert the printed output of Part H from the final cell in `03_eda.ipynb`.

Recommended caption:

**Table H1: Summary of candidate predictors, potential redundancy, categorical features, spatial observations, and data-quality checks.**

After inserting the output, add a short paragraph summarising the actual findings.

The paragraph should mention:

* Whether the price distribution supports evaluating a logarithmic target.
* Which features show promising relationships with price in the plots and correlation analysis.
* Which predictors may be redundant.
* Which features require categorical encoding.
* Whether coordinate duplication raises concerns about location precision.
* Any unexpected categories or data-quality issues that need follow-up.

Do not state that a feature is the strongest predictor unless the analysis or later model evaluation supports that conclusion.

---

# 12. Summary of Generated EDA Outputs

The following files are generated during the EDA stage.

| Output                                 | File path                                        | Purpose                                                      |
| -------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Price distribution plots               | `reports/figures/01_price_distribution.png`      | Compare original price, log price, and price per square foot |
| Correlation heatmap                    | `reports/figures/02_correlation_heatmap.png`     | Examine numerical relationships                              |
| Price versus area                      | `reports/figures/03a_price_vs_area.png`          | Investigate the relationship between area and price          |
| Price versus bedrooms                  | `reports/figures/03b_price_vs_bedrooms.png`      | Compare prices across bedroom counts                         |
| Price by property type                 | `reports/figures/03c_price_by_property_type.png` | Compare property-type price distributions                    |
| Other categorical comparisons          | `reports/figures/03d_other_categoricals.png`     | Examine furnishing, age, and area-source categories          |
| Locality median-price ranking          | `reports/figures/04a_locality_median_price.png`  | Compare locality-level median sale prices                    |
| Locality price-per-square-foot ranking | `reports/figures/04b_locality_median_pps.png`    | Compare locality-level median price per square foot          |
| Static geographic map                  | `reports/figures/05a_price_map_static.png`       | Visualise geographical price patterns                        |
| Interactive map                        | `reports/price_map.html`                         | Explore sampled listings interactively                       |
| EDA notebook                           | `notebooks/03_eda.ipynb`                         | Record the analysis and reproducible code                    |

The tables and screenshots inserted into this README should be generated from the actual notebook outputs.

---

# 13. Methodological Considerations

Several considerations are important when interpreting the EDA results.

## 13.1 Association does not imply causation

A relationship between property area and price does not prove that area alone causes the price difference.

Other factors, such as locality, property type, and condition, may contribute to the observed relationship.

## 13.2 Outliers require investigation

Unusually high or low values should be investigated rather than removed automatically.

An extreme price may represent a genuine property, while an unusual value may also indicate an error in the original listing.

The decision should be based on the feature's meaning and the evidence available.

## 13.3 Correlation is not a complete measure of feature importance

Pearson correlation captures linear association between numerical variables.

It does not capture every nonlinear relationship, and it does not directly measure a feature's predictive contribution to a trained model.

Model-based evaluation will be needed to assess predictive usefulness.

## 13.4 Locality summaries are influenced by property composition

Different localities may contain different property types, sizes, and furnishing categories.

Therefore, differences in median prices should not be interpreted as the isolated effect of locality.

## 13.5 Geographic coordinates may be approximate

Repeated coordinates can indicate approximate locality-level positions rather than exact property locations.

This limits the conclusions that can be drawn from the map and should be considered when developing location-based features.

## 13.6 Avoiding target leakage

The `price_per_sqft` variable is derived directly from `price_lakhs`.

It is used for descriptive analysis but should not be included as a predictor when the objective is to predict the same property's sale price.

Similarly, any feature created using the target price must be reviewed before it is included in a machine learning model.

---

# 14. Conclusion

Exploratory Data Analysis provides an initial understanding of the cleaned real-estate dataset and helps establish a foundation for machine learning.

The analysis examines the distribution of sale prices, compares original and log-transformed prices, investigates correlations between numerical features, studies the relationship between price and property characteristics, compares locality-level price statistics, and visualises the geographical distribution of listings.

The findings help identify plausible predictors, potential redundancy, categorical-encoding requirements, and data-quality concerns.

The results also highlight the importance of location as an area for further investigation.

However, EDA does not by itself determine the final feature set or guarantee predictive performance. Those decisions require validation using appropriately constructed machine learning pipelines.

The next stage will use these observations to investigate additional location-related features, prepare the modelling inputs, train candidate models, and compare their predictive performance.

---

# 15. Reproducibility Checklist

Before marking the EDA stage as complete, verify the following:

* [ ] `notebooks/03_eda.ipynb` contains all EDA steps from A to H.
* [ ] The cleaned dataset loads successfully.
* [ ] The DataFrame shape and missing values have been checked.
* [ ] Descriptive statistics have been generated.
* [ ] Price distributions have been examined.
* [ ] Original and log-transformed prices have been compared.
* [ ] The correlation heatmap has been generated.
* [ ] Potentially correlated predictors have been investigated.
* [ ] Price versus area has been plotted.
* [ ] Price versus bedroom count has been plotted.
* [ ] Price distributions across property types have been compared.
* [ ] Additional categorical comparisons have been examined.
* [ ] Locality-level price rankings have been generated.
* [ ] The static geographic map has been saved.
* [ ] Coordinate duplication has been investigated.
* [ ] The interactive map has been saved, if successfully generated.
* [ ] Part H summarises the actual findings.
* [ ] All figures referenced in this README exist in `reports/figures/`.
* [ ] The actual outputs have been inserted at the marked locations.
* [ ] The interpretation does not claim that correlation proves causation.
* [ ] `price_per_sqft` is documented as an EDA-only, target-derived variable.
* [ ] The README has been checked against the final notebook output.

**End of Module 3: Exploratory Data Analysis.**

