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

## MODULE 1: Data Collection and Understanding

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

**Next module: Module 3 — Exploratory Data Analysis (EDA).**
