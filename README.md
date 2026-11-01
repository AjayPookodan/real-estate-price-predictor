S# Real Estate Price Prediction

## Project Description

A machine learning project that predicts real-estate prices using property characteristics, location, and accessibility to public amenities and transportation facilities.

## Team Members

* Ajay Balakrishnan - 5
* Gopika CP - 38
* Devakrishnan - 30

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

## 1. Introduction

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

## 2. Objectives of Module 2

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

## 3. Dataset Used

### 3.1 Original Dataset

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

### 3.2 Final Processed Dataset

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

## 4. Preprocessing Workflow at a Glance

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

## 5. Detailed Explanation of Each Preprocessing Step

### Step 1: Loading the Raw Dataset

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

### Step 2: Standardizing Column Names

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

### Step 3: Filtering Sale Listings

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

### Step 4: Removing Duplicate Property IDs

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

### Step 5: Extracting Latitude and Longitude

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

### Step 6: Validating the Gurgaon Geographical Area

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

### Step 7: Converting Property Prices into Lakhs

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

### Step 8: Creating a Unified Property-Area Feature

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

### Step 9: Creating the `area_is_carpet` Feature

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

### Step 10: Converting Columns into Numeric Data Types

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

### Step 11: Removing Records with Missing Essential Information

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

### Step 12: Filtering Unrealistic Property Areas

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

### Step 13: Filtering Bedroom Counts

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

### Step 14: Filtering Area per Bedroom

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

### Step 15: Filtering Bathroom Counts

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

### Step 16: Filtering Price per Square Foot

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

### Step 17: Removing Near-Duplicate Listings

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

### Step 18: Grouping Rare Localities

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

### Step 19: Selecting the Final Columns

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

## 6. Understanding the Final 18 Columns

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

### 6.1 Target Variable

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

### 6.2 Input Features

The input features are the information available to the model when a property price is to be predicted.

The target price itself must never be included in the input matrix.

The property identifier is also excluded because it identifies a listing rather than describing a meaningful property characteristic.

Any features derived directly from the target price must also be excluded.

---

## 7. Final Dataset Statistics

The latest reported descriptive statistics provide an overview of the processed data.

### 7.1 Property Prices

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

### 7.2 Property Area

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

### 7.3 Bedrooms

| Statistic | Value |
| --------- | ----: |
| Mean      |  3.38 |
| Median    |     3 |
| Minimum   |     1 |
| Maximum   |     8 |

Most retained properties have approximately three or four bedrooms.

### 7.4 Geographical Coordinates

| Statistic | Latitude | Longitude |
| --------- | -------: | --------: |
| Mean      |  28.4380 |   77.0406 |
| Minimum   |  28.3509 |   76.8108 |
| Maximum   |  28.5348 |   77.1915 |

The coordinate values fall within the selected bounding box.

These values will be used later for geographical analysis and the development of amenity-related features.

---

## 8. Missing-Value Analysis

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

## 9. Data Leakage Prevention

Data leakage occurs when the model receives information during training that would not legitimately be available when predicting an unknown property's price.

The preprocessing module addresses several potential leakage risks.

### 9.1 Target Separation

The `price_lakhs` column is the target variable.

It must be separated from the input features before model training.

### 9.2 Property Identifier

The `prop_id` column is retained for traceability but should not be included in the model's input features.

### 9.3 Price-Derived Columns

The original price-related columns and the calculated price-per-square-foot helper are excluded from the final feature set.

Price per square foot directly uses the target price in its calculation. Including it as an input feature would provide the model with information about the value it is supposed to predict.

### 9.4 Brokerage

The `brokerage` column is excluded because the meaning of its values is uncertain and it may be related to the property's price.

### 9.5 Train-Test Separation

Preprocessing operations that learn parameters from data, such as imputing missing values, scaling numerical features, and encoding categories, should be fitted on the training set only.

This helps ensure that the test set remains independent for model evaluation.

---

## 10. Visual Validation

The preprocessing workflow also includes a comparison of property-area distributions before and after filtering.

The before-filtering plot uses the available raw `superbuiltup_sqft` values.

The after-filtering plot uses the unified `area_sqft` values from the final processed dataset.

These plots help identify differences in the distributions and assess the effect of the area filters.

However, the two plots do not contain exactly the same observations or necessarily the same area definitions.

The raw plot uses only non-missing super built-up area values, while the final plot includes area measurements selected from multiple columns.

Therefore, differences between the plots cannot be attributed exclusively to the filtering rules.

For a more accurate before-and-after comparison, the same unified-area construction should be applied to the raw dataset before plotting the distribution before filtering.

---

## 11. Limitations of the Current Preprocessing

Although the dataset has been substantially reduced and standardized, several limitations remain.

### 11.1 Minimum Price Requires Investigation

The minimum price in the final dataset is ₹100 lakhs.

This may indicate that the applied filters disproportionately excluded lower-priced properties.

The original sale/rent separation uses the `preference` column, but the final minimum price must be verified against the raw sale listings.

The price conversion and price-per-square-foot filtering rules should be reviewed before final model training.

### 11.2 Different Area Definitions

The unified `area_sqft` column can contain super built-up, super area, built-up, or carpet-area measurements.

These definitions differ, so the column is not perfectly standardized.

The `area_is_carpet` feature records whether carpet area was used as a fallback, but it does not eliminate all measurement differences.

### 11.3 Heuristic Outlier Filters

The selected thresholds for area, bedrooms, bathrooms, and price per square foot are project-specific assumptions.

They help remove implausible records but may also remove valid properties.

The effect of each rule should be reviewed and documented.

### 11.4 Duplicate Listings

The current near-duplicate removal method relies on matching selected property characteristics.

It may not identify every repeated property if some fields differ slightly.

### 11.5 Category Interpretation

Columns such as `owntype`, `furnish`, `facing`, and `age` may contain category codes.

Their meanings should be verified from the source dataset before they are encoded for machine learning.

### 11.6 Landmark Count

The `total_landmark_count` column is retained, but its exact definition and calculation should be confirmed.

A count of landmarks is not necessarily equivalent to accessibility. Two properties can have the same landmark count while being very different distances from important facilities.

### 11.7 Amenity Distances Have Not Yet Been Added

The current processed dataset contains latitude and longitude, but it does not yet contain verified distances to railway stations, bus stops, hospitals, schools, shopping centres, or highways.

Those features belong to a later feature-engineering stage.

---

## 12. Output File and Reproducibility

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

## 13. Summary of What Was Accomplished

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

## 14. Conclusion

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

## 4. Part A — EDA Setup and Dataset Verification

### 4.1 Loading the dataset

The first step is to load the cleaned dataset into a pandas DataFrame.

The notebook uses:

```python
df = pd.read_csv("../data/processed/clean_v2.csv")
```

The relative path works when the notebook's working directory is `notebooks`.

Loading the data into a DataFrame allows the dataset to be inspected and analysed using pandas, NumPy, Matplotlib, and Seaborn.

### 4.2 Creating the figure directory

The notebook creates the directory used to store the generated figures:

```python
os.makedirs("../reports/figures", exist_ok=True)
```

The `exist_ok=True` argument prevents an error if the directory already exists.

The figures generated during EDA are saved in this directory so they can be reused in the project report without having to recreate them manually.

### 4.3 Checking the dataset dimensions

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

### 4.4 Inspecting the first rows

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

### 4.5 Checking data types

The data types are inspected using:

```python
df.info()
```

This reveals the number of non-null values and the data type of each column.

Data-type inspection is important because a column may contain numeric-looking values that should actually be treated as categories.

For example, the values in `furnish`, `facing`, `owntype`, and `age` are encoded categories. Treating these values as ordinary continuous measurements could lead to misleading conclusions during correlation analysis.

### 4.6 Checking missing values

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

### 4.7 Converting encoded categories for EDA

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

### 4.8 Creating EDA helper columns

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

#### B. Log-transformed price

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

### 4.9 Verifying the prepared DataFrame

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

## 5. Part B — Descriptive Statistical Analysis

### 5.1 Purpose

Descriptive statistics provide a numerical summary of the dataset.

Before studying relationships between features, it is useful to understand the typical values, spread, and range of the numerical variables.

This step examines properties such as price, area, bedrooms, bathrooms, and other numeric attributes.

### 5.2 Generating descriptive statistics

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

### 5.3 Examining property types

The distribution of property types is examined using:

```python
display(df["property_type"].value_counts())
```

This shows how many listings belong to each property category.

Property types can have different pricing patterns because apartments, builder floors, houses, and land may differ in size, facilities, ownership arrangements, and market demand.

However, the number of listings in each category must be considered when interpreting comparisons.

A category containing very few records may produce an unstable estimate of its typical price.

### 5.4 Examining encoded categories

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

### 5.5 Main observations to record

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

## 6. Part C — Analysis of Price Distributions

### 6.1 Purpose

Price distributions help explain how property prices are spread across the dataset.

The analysis examines three measures:

1. Sale price in lakhs.
2. Log-transformed sale price.
3. Price per square foot.

Histograms are used to understand the shape of each distribution.

This is particularly important because real-estate prices often vary considerably, even within the same city.

### 6.2 Distribution of original sale prices

The first histogram uses `price_lakhs`.

The x-axis represents sale price in lakhs, while the y-axis represents the number of listings within each price interval.

This plot helps reveal whether most properties fall within a relatively narrow price range or whether the dataset contains a long tail of expensive properties.

A right-skewed distribution typically has most observations concentrated at lower values and a tail extending toward higher values.

Such a pattern can occur when a dataset contains many moderately priced properties and a smaller number of luxury properties.

The histogram is saved as part of:

`reports/figures/01_price_distribution.png`

### 6.3 Distribution of log-transformed prices

The second histogram uses `log_price`.

The transformation compresses the larger price values, allowing the distribution to be examined on a different scale.

The purpose is not to assume that the transformed distribution will be normal. Instead, it allows the analyst to assess whether the transformation produces a more balanced distribution.

If the original prices are strongly right-skewed and the log-transformed values are less skewed, this supports evaluating a logarithmic target during model development.

A histogram alone cannot establish which target produces the best predictive model. That decision must also consider model evaluation results.

### 6.4 Distribution of price per square foot

The third histogram uses `price_per_sqft`.

This measure expresses the price relative to the property area.

It can reveal whether the dataset contains properties with unusually high or low prices relative to their size.

Such observations may arise from genuine market differences, property characteristics, location, or data-quality issues.

A high price per square foot should not automatically be considered an error.

For example, a relatively small property in a premium locality may have a high price per square foot while still having a lower total price than a large property elsewhere.

### 6.5 Comparing skewness and central tendency

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

### 6.6 Interpretation and target decision

Use the actual histogram and summary statistics to answer:

* Is the original price distribution right-skewed?
* Does the log transformation reduce the observed skewness?
* Is the mean higher than the median?
* Does price per square foot contain a long tail?
* Is there evidence that a logarithmic target is worth evaluating?

#### Preliminary conclusion

A logarithmic target is worth considering when the original price distribution is strongly right-skewed and the transformation improves the behaviour of the target.

However, this EDA stage does not establish that the logarithmic target will outperform the original target in every model.

The decision will be evaluated during model development.

---

## 7. Part D — Correlation Analysis

### 7.1 Purpose

Correlation analysis measures the strength and direction of relationships between numerical variables.

In this project, correlation analysis is used to investigate:

* How area relates to property price.
* Whether bedroom and bathroom counts are associated with price.
* Whether other numerical features have relationships with the target.
* Whether some predictors are strongly related to one another.

The last point is important because highly related predictors may contain overlapping information.

### 7.2 Selecting appropriate columns

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

### 7.3 Checking missing values

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

### 7.4 Calculating the correlation matrix

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

### 7.5 Creating the correlation heatmap

The notebook displays the correlation matrix as a heatmap.

The upper triangle is masked to avoid displaying the same pairwise correlations twice.

The resulting figure is saved as:

`reports/figures/02_correlation_heatmap.png`

<img width="2087" height="1779" alt="image" src="https://github.com/user-attachments/assets/a78c7c0e-51c3-4a07-81a4-4bdca8af9af6" />


Insert the image:

`reports/figures/02_correlation_heatmap.png`

Recommended caption:

**Figure D1: Pearson correlation heatmap of the selected numerical features and log-transformed sale price.**

### 7.6 Identifying features correlated with the target

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

### 7.7 Checking potential multicollinearity

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

### 7.8 Limitations of correlation analysis

The correlation matrix does not include every important property attribute.

In particular, categorical features and locality may have meaningful relationships with price that are not represented by this numeric Pearson correlation matrix.

Correlation also measures linear relationships rather than every possible pattern.

Therefore, correlation analysis is used together with the distributions, boxplots, locality summaries, and geographical maps.

---

## 8. Part E — Price Versus Area, Bedrooms, and Property Type

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

## 9. Part F — Locality-Level Price Analysis

### 9.1 Purpose

Location is an important consideration in real-estate analysis.

Properties with similar areas and room counts may have different prices because they are situated in different localities.

This section compares prices across the grouped localities represented in the dataset.

The analysis examines both:

1. Median total sale price.
2. Median price per square foot.

These measures describe different aspects of the market.

### 9.2 Grouping by locality

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

### 9.3 Why median price is used

The median represents the middle observation after prices are sorted.

It is less sensitive than the mean to a small number of unusually expensive properties.

This makes it useful for comparing the typical observed price across localities.

The median does not remove the influence of all differences between properties. For example, one locality may contain larger homes than another.

Therefore, locality-level total price and price per square foot should be interpreted together.

### 9.4 Top and bottom localities by median sale price

The first ranking identifies localities with the highest and lowest median sale prices.

The corresponding figure is saved as:

`reports/figures/04a_locality_median_price.png`

<img width="2982" height="1168" alt="image" src="https://github.com/user-attachments/assets/210ca182-8879-4b5d-961d-be347e8b71cf" />


Insert:

`reports/figures/04a_locality_median_price.png`

Recommended caption:

**Figure F1: Ten highest- and lowest-ranked eligible localities by median sale price.**

The figure compares localities with at least 15 listings, excluding `Other`.

### 9.5 Top and bottom localities by median price per square foot

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

### 9.6 Comparing the two rankings

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

## 10. Part G — Geographic Distribution of Property Prices

### 10.1 Purpose

The locality analysis compares named locality groups. Geographic visualisation provides a complementary view by plotting the property listings using latitude and longitude.

This helps examine how listings are distributed spatially and whether properties with different prices appear in different areas.

The analysis uses two forms of visualisation:

1. A static scatterplot map.
2. An interactive Folium map.

The static plot is suitable for the project report, while the interactive map allows individual listings to be inspected.

### 10.2 Checking geographic coordinates

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

### 10.3 Static geographic price map

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

### 10.4 Checking coordinate duplication

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

#### Interpretation

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

#### Opening the map

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

#### Interpretation

Use the map to inspect:

* Geographic clusters of listings.
* Local differences in price per square foot.
* The distribution of prices within and between areas.
* Whether high-priced listings are concentrated in particular regions.
* Whether shared coordinates limit the precision of the displayed locations.

The interactive map complements the static figure but does not replace the numerical locality analysis.

If the background map does not load, the static scatterplot remains available for the report.

---

## 11. Part H — EDA Wrap-Up Analysis

### 11.1 Purpose

The final stage summarises the findings from the preceding sections.

It identifies candidate features, possible modelling concerns, and areas that require further investigation.

This summary provides a bridge between EDA and the next stage of the project.

### 11.2 Target variable

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

### 11.3 Candidate important features

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

### 11.4 Potential multicollinearity

Area, bedroom count, and bathroom count may contain overlapping information.

For example, larger properties may tend to have more bedrooms and bathrooms.

The correlation matrix is used to screen for strongly related numerical predictors.

However, correlated predictors should not automatically be removed.

The appropriate decision depends on the model being used and whether the features provide useful predictive information.

Tree-based models and linear models can behave differently when predictors are correlated.

The issue should be reviewed during model development rather than resolved solely on the basis of a single correlation threshold.

### 11.5 Categorical features and encoding

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

### 11.6 Spatial importance

The locality-level rankings and geographic maps examine two different aspects of location.

The locality analysis compares typical prices across named locality groups.

The map examines the geographic distribution of the listings and their prices.

Together, these analyses help justify investigating location-related features in the next stage.

Possible follow-up work includes examining distances to relevant landmarks or amenities, provided the required data is available and the calculations can be validated.

The maps alone do not establish that location causes a particular price difference.

## 11.7 Data-quality observations#

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

### 11.8 Final EDA observations

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

## 12. Summary of Generated EDA Outputs

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

## 13. Methodological Considerations

Several considerations are important when interpreting the EDA results.

### 13.1 Association does not imply causation

A relationship between property area and price does not prove that area alone causes the price difference.

Other factors, such as locality, property type, and condition, may contribute to the observed relationship.

### 13.2 Outliers require investigation

Unusually high or low values should be investigated rather than removed automatically.

An extreme price may represent a genuine property, while an unusual value may also indicate an error in the original listing.

The decision should be based on the feature's meaning and the evidence available.

### 13.3 Correlation is not a complete measure of feature importance

Pearson correlation captures linear association between numerical variables.

It does not capture every nonlinear relationship, and it does not directly measure a feature's predictive contribution to a trained model.

Model-based evaluation will be needed to assess predictive usefulness.

### 13.4 Locality summaries are influenced by property composition

Different localities may contain different property types, sizes, and furnishing categories.

Therefore, differences in median prices should not be interpreted as the isolated effect of locality.

### 13.5 Geographic coordinates may be approximate

Repeated coordinates can indicate approximate locality-level positions rather than exact property locations.

This limits the conclusions that can be drawn from the map and should be considered when developing location-based features.

### 13.6 Avoiding target leakage

The `price_per_sqft` variable is derived directly from `price_lakhs`.

It is used for descriptive analysis but should not be included as a predictor when the objective is to predict the same property's sale price.

Similarly, any feature created using the target price must be reviewed before it is included in a machine learning model.

---

## 14. Conclusion

Exploratory Data Analysis provides an initial understanding of the cleaned real-estate dataset and helps establish a foundation for machine learning.

The analysis examines the distribution of sale prices, compares original and log-transformed prices, investigates correlations between numerical features, studies the relationship between price and property characteristics, compares locality-level price statistics, and visualises the geographical distribution of listings.

The findings help identify plausible predictors, potential redundancy, categorical-encoding requirements, and data-quality concerns.

The results also highlight the importance of location as an area for further investigation.

However, EDA does not by itself determine the final feature set or guarantee predictive performance. Those decisions require validation using appropriately constructed machine learning pipelines.

The next stage will use these observations to investigate additional location-related features, prepare the modelling inputs, train candidate models, and compare their predictive performance.

---

## 15. Reproducibility Checklist

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
# MODULE 4: Location and Accessibility Analysis#

## 1. Overview

The Location and Accessibility Analysis module is a component of the **Real Estate Price Prediction** project. It investigates how the geographical location of a property and its proximity to important amenities may be associated with residential property prices in Gurugram (Gurgaon), Haryana, India.

Property prices are influenced not only by the physical characteristics of a property, such as its area, number of bedrooms, and property type, but also by its surrounding environment. Accessibility to public transport, educational institutions, healthcare facilities, shopping centres, rail infrastructure, and major roads can influence the convenience and desirability of a residential location.

This module enriches the property dataset with geographical features derived from OpenStreetMap (OSM). It then explores relationships between these features and property prices using statistical analysis and visualisation.

The module is designed to support the project's broader objective of building a data-driven real-estate price prediction system.

---

## 2. Objectives

The primary objectives of this module are:

1. Identify relevant geographical amenities surrounding residential properties.
2. Calculate the distance from each property to the nearest mapped amenity.
3. Count selected amenities within specified radii around each property.
4. Integrate the generated geographical features with the cleaned property dataset.
5. Examine the statistical relationships between geographical accessibility and property prices.
6. Investigate correlations between different accessibility features.
7. Evaluate the quality and limitations of the geographical data.
8. Generate reusable features that can be incorporated into subsequent modelling experiments.

### Research questions

This module investigates the following questions:

* Are properties closer to railway infrastructure generally associated with different prices?
* Is proximity to bus stops associated with property prices?
* Does access to schools and hospitals vary across property locations?
* Are properties surrounded by more amenities associated with higher prices?
* Are the distances to different geographical features strongly correlated?
* How much useful geographical information can be extracted from the available property coordinates and OSM data?

These questions are exploratory. A statistical relationship does not, by itself, establish causation.

---

## 3. Study Area

**Study area:** Gurugram (Gurgaon), Haryana, India.

Gurugram is a major urban and commercial centre in the National Capital Region (NCR). Residential property values can vary substantially between localities because of differences in transport connectivity, infrastructure, commercial development, and access to public services.

The analysis uses geographical coordinates associated with the properties in the cleaned dataset. Distances are calculated using a projected coordinate reference system suitable for the study area.

### Coordinate reference system

* Input coordinates: WGS 84 (`EPSG:4326`)
* Projected coordinate system: WGS 84 / UTM zone 43N (`EPSG:32643`)
* Distance unit during computation: metres
* Distance unit in the output dataset: kilometres

Projecting the coordinates before performing distance calculations avoids treating latitude and longitude as ordinary planar coordinates.

---

## 4. Data Sources

### 4.1 Property dataset

The property records are obtained from the cleaned dataset:

`data/processed/clean_v2.csv`

This dataset contains property identifiers, geographical coordinates, prices, and other property-related attributes.

The principal fields used by this module are:

| Field         | Description                              |
| ------------- | ---------------------------------------- |
| `prop_id`     | Unique property identifier               |
| `latitude`    | Property latitude                        |
| `longitude`   | Property longitude                       |
| `price_lakhs` | Property price in lakhs of Indian rupees |
| `area_sqft`   | Property area in square feet             |

The location-feature generation script requires `prop_id`, `latitude`, and `longitude`. Other fields are used when the geographical features are combined with the property dataset for price analysis.

The dataset used in the current workflow contains **7,457 property records**.

### 4.2 OpenStreetMap

The geographical amenity data are derived from OpenStreetMap (OSM), an open collaborative mapping project.

The source workflow uses an India-wide OSM extract in PBF format, obtained from Geofabrik, and locally processes the relevant geographical features.

The original extract is:

`data/external/india-latest.osm.pbf`

The large PBF file is an intermediate source used to extract the relevant features. The location-feature generation script described below reads the already-extracted GeoPackage files; it does not download or reprocess the PBF file automatically.

### 4.3 Extracted amenity datasets

The following GeoPackage files are expected in `data/external/`:

| File            | Feature category           | Purpose                                                            |
| --------------- | -------------------------- | ------------------------------------------------------------------ |
| `railway.gpkg`  | Railway infrastructure     | Measure proximity to mapped railway features                       |
| `bus.gpkg`      | Bus-related locations      | Estimate proximity to mapped bus locations                         |
| `school.gpkg`   | Schools                    | Measure school accessibility and nearby school density             |
| `hospital.gpkg` | Hospitals                  | Measure healthcare accessibility and nearby hospital density       |
| `mall.gpkg`     | Shopping centres and malls | Estimate shopping accessibility and nearby shopping-centre density |
| `highway.gpkg`  | Highway-related features   | Measure proximity to mapped highway geometries                     |
| `mainroad.gpkg` | Main roads                 | Measure proximity to mapped main-road geometries                   |

These are the expected filenames used by the script. The exact meaning of each category depends on the tags and filtering rules applied during extraction.

**Important:** A missing GeoPackage is skipped by the current script. Therefore, the generated output must be checked to confirm which amenity categories were actually loaded.

The presence of a category in OSM does not guarantee complete or uniform geographical coverage.

---

## 5. Methodology

The workflow consists of data loading, coordinate transformation, distance calculation, amenity counting, validation, statistical analysis, and output generation.

### Step 1: Load and validate property records

The cleaned property dataset is loaded from:

`data/processed/clean_v2.csv`

The script verifies that the required columns are present and that property identifiers are unique.

Latitude and longitude values are converted to numeric values and checked against their valid geographic ranges.

The property records are then converted into point geometries using GeoPandas.

### Step 2: Load amenity geometries

The extracted GeoPackage files are loaded using GeoPandas.

Each available amenity dataset is checked for a defined coordinate reference system. Invalid or empty geometries are excluded from the geographical calculations.

The geometries are transformed into `EPSG:32643` so that distances can be calculated in metres.

### Step 3: Calculate distance to the nearest amenity

For each property, the shortest distance to the nearest mapped feature in each available amenity category is calculated.

The nearest-feature calculation uses GeoPandas spatial operations.

The resulting distances are converted from metres to kilometres.

For example:

* `dist_railway_km`: distance to the nearest mapped railway feature.
* `dist_bus_km`: distance to the nearest mapped bus feature.
* `dist_school_km`: distance to the nearest mapped school feature.
* `dist_hospital_km`: distance to the nearest mapped hospital feature.
* `dist_mall_km`: distance to the nearest mapped mall feature.
* `dist_highway_km`: distance to the nearest mapped highway feature.
* `dist_mainroad_km`: distance to the nearest mapped main-road feature.

These values represent **straight-line geometric distances**, not road-network travel distances or actual journey times.

For linear and polygon geometries, the nearest-distance calculation measures the shortest distance to the mapped geometry itself.

### Step 4: Calculate amenity counts within specified radii

Distance alone does not describe the complete surrounding environment. Two properties may have similar distances to their nearest school but differ considerably in the number of schools nearby.

To capture this difference, the module calculates counts of selected amenities within two radii:

* 1 kilometre
* 2 kilometres

The categories included in the current count features are schools, hospitals, bus locations, and malls.

For non-point geometries used in these count calculations, centroids are used to represent their locations. Consequently, these counts are based on representative points rather than the full spatial extent of each geometry.

The generated features include:

| Feature          | Meaning                                        |
| ---------------- | ---------------------------------------------- |
| `n_school_1km`   | Number of mapped school features within 1 km   |
| `n_school_2km`   | Number of mapped school features within 2 km   |
| `n_hospital_1km` | Number of mapped hospital features within 1 km |
| `n_hospital_2km` | Number of mapped hospital features within 2 km |
| `n_bus_1km`      | Number of mapped bus features within 1 km      |
| `n_bus_2km`      | Number of mapped bus features within 2 km      |
| `n_mall_1km`     | Number of mapped mall features within 1 km     |
| `n_mall_2km`     | Number of mapped mall features within 2 km     |

These counts measure the density of mapped features within circular, straight-line search areas around each property.

They should not be interpreted as counts of every real-world amenity in the area because OSM coverage may be incomplete.

### Step 5: Construct the location-feature dataset

The distance and count features are combined into a table indexed by `prop_id`.

The resulting dataset is saved to:

`data/processed/location_features.csv`

The current generated dataset contains:

* 7,457 property records
* 16 columns, including the property identifier
* 15 geographical features in the observed output

The number of available features can vary if some GeoPackage files are missing or contain no usable geometries.

### Step 6: Integrate location features with property data

The location-feature dataset can be merged with the cleaned property dataset using `prop_id`.

The property identifier acts as the key connecting each geographical feature to the corresponding property record.

A one-to-one merge validation is recommended to prevent accidental duplication of property records.

The combined dataset can then be used for exploratory analysis and subsequent model experiments.

---

## 6. Feature Engineering

The module creates two principal types of geographical features.

### 6.1 Nearest-distance features

Distance features represent the proximity of each property to the nearest mapped amenity category.

Examples include:

* `dist_railway_km`
* `dist_bus_km`
* `dist_school_km`
* `dist_hospital_km`
* `dist_mall_km`
* `dist_highway_km`
* `dist_mainroad_km`

A smaller value indicates that the property is geographically closer to the nearest mapped feature.

However, a smaller distance does not necessarily imply better accessibility. For example, proximity to a major road may improve connectivity but could also be associated with traffic noise or congestion.

### 6.2 Nearby-amenity count features

Count features represent the number of mapped amenities within a specified radius.

Examples include:

* `n_school_1km`
* `n_school_2km`
* `n_hospital_1km`
* `n_hospital_2km`
* `n_bus_1km`
* `n_bus_2km`
* `n_mall_1km`
* `n_mall_2km`

Higher counts indicate a greater number of mapped features within the relevant radius, subject to the quality and completeness of the source data.

Distance and count features provide complementary information. Distance describes proximity to the nearest feature, while counts approximate the density of features around a property.

---

## 7. Validation and Quality Checks

Validation is performed to confirm that the generated feature dataset is structurally consistent and numerically valid.

### 7.1 Record and identifier validation

The generated dataset is checked for:

* The expected number of property records.
* Duplicate property identifiers.
* Missing property identifiers.
* Property identifiers that are absent from the cleaned dataset.
* Preservation of property order when required by the workflow.

The current dataset contains 7,457 rows, with no duplicate property identifiers.

A comparison with `clean_v2.csv` confirmed that all 7,457 identifiers matched in the same order.

### 7.2 Missing-value validation

The generated `location_features.csv` file was checked for missing values.

**Observed result:** 0 missing values across the generated feature table.

This confirms that the current output contains no missing values. It does not independently establish that every source feature is correctly mapped or that all amenity categories are complete.

### 7.3 Distance validation

The distance features were checked for negative values.

**Observed result:** 0 negative distances.

Negative distances would be invalid for this application.

The minimum distance to a school is zero for some records. Such values may occur when a property point overlaps a mapped school geometry or when property coordinates are approximate.

Zero distances are therefore retained rather than automatically removed.

### 7.4 Amenity-count validation

The amenity-count features were checked for negative values.

**Observed result:** 0 negative counts.

The following relationship was also tested for every available count category:

`count within 2 km >= count within 1 km`

The current output reported zero violations for schools, hospitals, bus locations, and malls.

This is an important consistency check because the 2 km search radius contains the entire 1 km search area.

### 7.5 Coordinate-quality limitations

The property dataset contains 7,457 records but only 1,137 unique latitude-longitude pairs.

Multiple property listings share identical coordinates. Some of these groups contain many listings.

This may occur because:

* Multiple listings refer to the same building or complex.
* The original source uses locality-level or approximate coordinates.
* Coordinates are rounded or reused across records.
* Several properties genuinely share the same mapped location.

Consequently, the location features cannot always distinguish individual properties within the same coordinate group.

Listings sharing coordinates will generally receive identical distance and amenity-count features when calculated from the same amenity geometries.

This limitation should be considered when interpreting the results.

---

## 8. Exploratory Data Analysis

After feature generation, the geographical features are combined with the property dataset for exploratory analysis.

### 8.1 Price per square foot

Price per square foot is calculated as:

`price_per_sqft = price_lakhs * 100000 / area_sqft`

This converts the property price from lakhs of rupees into rupees and divides it by the property area in square feet.

Price per square foot provides a normalised measure of property value that can help compare properties of different sizes.

It is used for exploratory analysis rather than as a feature when predicting the total price from the same target price.

### 8.2 Log-transformed price

A log-transformed price measure can be calculated as:

`log_price = np.log1p(price_lakhs)`

Real-estate prices are often right-skewed, with a relatively small number of expensive properties.

A logarithmic transformation can reduce the influence of extreme values and make patterns easier to examine.

### 8.3 Spearman correlation analysis

Spearman's rank correlation coefficient is used to examine monotonic relationships between geographical features and price measures.

The analysis can compare each location feature against:

* `price_per_sqft`
* `log_price`

Spearman correlation is useful when relationships are not necessarily linear or when the variables have skewed distributions.

The coefficient ranges from -1 to +1:

| Correlation | Interpretation                        |
| ----------- | ------------------------------------- |
| Close to +1 | Strong positive monotonic association |
| Close to -1 | Strong negative monotonic association |
| Close to 0  | Little or no monotonic association    |

A weak correlation is still a valid finding. It may indicate that geographical accessibility alone does not explain much variation in property prices.

Correlation does not establish causation and does not account automatically for confounding factors such as locality, property type, area, age, or other characteristics.

### 8.4 Distance-versus-price analysis

Binned plots can be used to examine how median price per square foot changes with distance to an amenity.

For example, properties can be grouped into distance intervals and the median price per square foot calculated for each interval.

Median values are often more robust to extreme property prices than means.

The resulting patterns may indicate whether proximity to an amenity is associated with higher or lower property values.

### 8.5 Amenity-density analysis

The counts of amenities within 1 km and 2 km can be compared with property prices.

This helps investigate whether the number of mapped schools, hospitals, bus locations, or malls in the surrounding area is associated with property value.

However, greater amenity density may also be associated with urban density, commercial development, and locality characteristics. The observed relationship should not automatically be attributed to amenity availability alone.

### 8.6 Correlation between distance features

A correlation matrix can be constructed using the nearest-distance features.

This analysis helps identify features that move together.

For example, distance to railway infrastructure may be correlated with distance to major roads because both reflect the spatial organisation of urban development.

Strongly correlated features may contain overlapping information, although correlation alone does not prove that either feature is unnecessary.

### 8.7 Highway and main-road distance inspection

The current output has identical summary statistics for `dist_highway_km` and `dist_mainroad_km`.

Both features have:

* Mean distance: approximately 0.022 km.
* Maximum distance: approximately 0.239 km.

These results warrant further inspection.

Possible explanations include overlapping road geometries, similar extraction rules, or a genuine spatial relationship between the two categories.

The two columns should not be removed or treated as duplicates based on summary statistics alone. Compare their row-level values and correlation before deciding how to handle them.

---

## 9. Outputs and Generated Files

The principal output is:

`data/processed/location_features.csv`

It contains the property identifier and generated geographical features.

Other outputs produced by the exploratory analysis workflow may include:

| Output                                           | Purpose                                                               |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| `reports/location_price_correlations.csv`        | Statistical associations between location features and price measures |
| `reports/distance_correlations.csv`              | Correlation matrix for nearest-distance features                      |
| `reports/figures/07_distance_vs_price.png`       | Distance-versus-price visualisation                                   |
| `reports/figures/08_amenity_counts_vs_price.png` | Amenity-count-versus-price visualisation                              |
| `reports/figures/08_distance_correlations.png`   | Distance-feature correlation heatmap                                  |

These are expected output paths from the notebook workflow. Their existence and final contents should be verified in the repository before they are referenced as completed deliverables.

The generated features can also be merged into a modelling dataset for experiments comparing models with and without geographical information.

---

## 10. Project Structure

The relevant project structure is:

```text
real-estate-price-prediction/
│
├── data/
│   ├── processed/
│   │   ├── clean_v2.csv
│   │   └── location_features.csv
│   │
│   └── external/
│       ├── india-latest.osm.pbf
│       ├── railway.gpkg
│       ├── bus.gpkg
│       ├── school.gpkg
│       ├── hospital.gpkg
│       ├── mall.gpkg
│       ├── highway.gpkg
│       └── mainroad.gpkg
│
├── notebooks/
│   └── 04_location_features.ipynb
│
├── src/
│   └── location_features.py
│
└── reports/
    ├── location_price_correlations.csv
    ├── distance_correlations.csv
    └── figures/
        ├── 07_distance_vs_price.png
        ├── 08_amenity_counts_vs_price.png
        └── 08_distance_correlations.png
```

This structure illustrates the expected organisation. Some files may be absent if their corresponding extraction or analysis steps have not been run.

The original OSM PBF file and extracted GeoPackages are source/intermediate data, not outputs generated by `location_features.py`.

---

## 11. Installation and Dependencies

The module uses Python and common geospatial and data-analysis libraries.

Principal dependencies include:

* Python
* pandas
* NumPy
* GeoPandas
* SciPy
* Shapely
* Pyogrio or another supported GeoPandas file-reading engine
* A spatial-index backend supported by GeoPandas, where required by the installed version

Install the Python dependencies in the project's virtual environment.

If the project has a `requirements.txt` file, use:

```powershell
pip install -r requirements.txt
```

Otherwise, install the required packages individually:

```powershell
pip install pandas numpy geopandas scipy shapely pyogrio
```

Geospatial dependencies may require additional system components depending on the Python environment and operating system.

---

## 12. Running the Location-Feature Generation Script

The reusable feature-generation script is:

`src/location_features.py`

It reads the cleaned property dataset and the available GeoPackage files, calculates the geographical features, validates property identifiers, and saves the resulting CSV.

### Step 1: Open the project directory

```powershell
cd D:\real-estate-price-prediction
```

### Step 2: Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Run the script

```powershell
python src/location_features.py
```

### Step 4: Verify the output

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/processed/location_features.csv'); print('Shape:',d.shape); print('Duplicate IDs:',d['prop_id'].duplicated().sum()); print(d.head())"
```

The script overwrites the existing `location_features.csv` file when it completes successfully.

**Important:** The script uses the existing GeoPackage files. It does not automatically download OSM data, extract new amenities from the PBF file, or regenerate missing GeoPackages.

If an amenity GeoPackage is missing, the script may skip that category and produce fewer features. Review the script's terminal output to confirm which categories were loaded.

---

## 13. Reproducibility

The location-feature generation process is separated from exploratory analysis so that the geographical features can be regenerated without manually rerunning every notebook cell.

Reproducibility depends on:

1. Using the same cleaned property dataset.
2. Using the same extracted OSM GeoPackages.
3. Using consistent amenity extraction and filtering rules.
4. Using compatible versions of the Python geospatial libraries.
5. Applying the same coordinate reference system and distance definitions.
6. Keeping the feature-generation script and notebook logic consistent.

Changes to the underlying OSM extract, extraction rules, property coordinates, or amenity datasets may change the resulting feature values.

For reproducible research, record the source-data date and the versions of the relevant packages where possible.

---

## 14. Limitations

The results must be interpreted in the context of the following limitations.

### 14.1 Incomplete OSM coverage

OpenStreetMap is collaboratively maintained. Some schools, hospitals, bus locations, shopping centres, and roads may be missing, incorrectly tagged, or outdated.

A low amenity count does not necessarily mean that an area lacks those amenities.

### 14.2 Straight-line distance

The calculated distances represent geometric proximity, not actual travel distance.

A school that is 500 metres away in a straight line may require a longer journey because of roads, walls, crossings, railway lines, or restricted access.

Network-based routing would be needed to estimate realistic travel distances or travel times.

### 14.3 Approximate property coordinates

Multiple listings share the same coordinates. If a listing uses a locality-level or approximate point, the calculated distance may not represent the property's precise physical location.

This is especially important when interpreting zero distances and very small distances.

### 14.4 Amenity representation

Non-point geometries used for amenity counts are represented by their centroids. This is computationally efficient but may not reflect the entire geometry of a large polygon or a long feature.

The count features therefore measure the number of represented amenity geometries within a radius, rather than necessarily the number of complete physical amenities intersecting that radius.

### 14.5 Feature-category definitions

The meaning of an OSM feature depends on the tags and filters used during extraction. The extracted railway, bus, highway, and main-road categories should be documented alongside their extraction rules.

### 14.6 Correlation is not causation

An association between distance and property price does not prove that proximity to an amenity causes a price difference.

Locality, building quality, property size, age, and market conditions may influence both location and price.

### 14.7 Spatial dependence

Nearby properties are not necessarily independent observations. Properties in the same locality or building may share geographical characteristics.

Random train-test splits may therefore overestimate predictive performance if geographically similar properties appear in both training and test sets.

A spatially aware evaluation strategy may be more appropriate when assessing the generalisation of the final prediction model to new locations.

---

## 15. Integration with the Price Prediction Model

The generated geographical features can be used to investigate whether location and accessibility improve the predictive performance of the real-estate model.

A suitable experiment is to compare:

**Baseline model**

Uses property-related features such as area, bedrooms, bathrooms, property type, age, and locality information.

**Location-enhanced model**

Uses the same baseline features plus selected geographical distance and amenity-count features.

Model performance can then be compared using appropriate regression metrics, such as:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score

Use the same data split and evaluation procedure for both models so that the comparison is meaningful.

To reduce spatial leakage, consider grouping properties by locality or using spatially separated training and test sets.

### Avoiding target leakage

The `price_per_sqft` measure is derived from the property price itself. If `price_lakhs` is the prediction target, `price_per_sqft` must not be included as an input feature because it directly incorporates the target.

It can be used for exploratory analysis, but not as a predictor of the same total price.

All preprocessing and feature selection decisions should also be made without using information from the held-out test set.

---

## 16. Future Improvements

Potential extensions include:

1. Calculate road-network travel distances instead of relying exclusively on straight-line distances.
2. Estimate travel times to important amenities.
3. Include metro stations and additional public-transport categories, where reliable data are available.
4. Improve property-coordinate accuracy where more precise locations can be obtained.
5. Record OSM extraction dates and document the exact tags used for each category.
6. Compare centroid-based counts with geometry-aware spatial counts.
7. Investigate highly correlated road-distance features.
8. Add spatial visualisations showing property prices and amenity distributions.
9. Compare baseline and location-enhanced prediction models.
10. Evaluate models using spatially separated train-test splits.
11. Explore distance bands or nonlinear transformations when the relationship between accessibility and price is not linear.
12. Examine whether the importance of accessibility differs between localities or property types.

---

## 17. Conclusion

The Location and Accessibility Analysis module extends the real-estate dataset with geographical information derived from OpenStreetMap.

It calculates nearest-feature distances and nearby-amenity counts, validates the generated data, and provides a foundation for exploring relationships between accessibility and property prices.

The current feature dataset contains 7,457 property records with no duplicate property identifiers, no missing values, no negative distances or counts, and no violations of the tested 1 km versus 2 km count relationships.

However, the results remain subject to limitations arising from OSM coverage, approximate property coordinates, straight-line distance measurements, and shared geographical locations.

The generated features provide an additional source of information for subsequent modelling experiments. Their practical value should ultimately be determined by comparing model performance with and without the geographical features using a consistent and leakage-aware evaluation strategy.



# Module 5: Feature Engineering

## Gurgaon Real Estate Price Prediction

---

## 1. Introduction

Feature engineering is one of the most important stages in a machine learning project. The quality and representation of input features can significantly affect a model's ability to learn patterns and make accurate predictions.

In this project, Module 4 focuses on transforming the cleaned Gurgaon real estate dataset into a machine-learning-ready dataset, creating meaningful features, incorporating geographical accessibility information, selecting suitable feature groups, comparing machine learning algorithms, tuning the best-performing model, and preparing the final model for deployment.

The primary objective of this module is to build a reliable real estate price prediction model that estimates property prices using property characteristics and geographical information.

A secondary objective is to investigate whether geographical accessibility features, such as distances to railway stations, schools, hospitals, malls, bus stops, highways, and main roads, improve prediction accuracy.

### Main objectives

* Create meaningful features from existing property attributes.
* Develop geographical accessibility features using OpenStreetMap-derived information.
* Investigate relationships between features and property prices.
* Identify the most useful features using feature importance analysis.
* Compare different feature combinations through ablation experiments.
* Compare machine learning algorithms.
* Tune the Random Forest model.
* Evaluate the selected model on an independent test set.
* Analyse prediction errors.
* Save the final model and its configuration for future deployment.
* Develop reusable feature-engineering functions for the application.

### Tools and technologies

| Technology                 | Purpose                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| Python                     | Data processing and model development                            |
| Pandas                     | Data manipulation and feature creation                           |
| NumPy                      | Numerical operations and logarithmic transformations             |
| Scikit-learn               | Preprocessing, model training, feature selection, and evaluation |
| Matplotlib                 | Data visualization                                               |
| Seaborn                    | Statistical visualization                                        |
| Jupyter Notebook           | Experimentation and documentation                                |
| Joblib                     | Saving and loading the trained model                             |
| JSON                       | Storing model metadata and configuration                         |
| OpenStreetMap-derived data | Geographical accessibility analysis                              |
| GeoPandas and OSMnx        | Geographical data processing                                     |
| SciPy                      | Spatial calculations and statistical analysis                    |
| Streamlit                  | Planned deployment interface                                     |

---

## 2. Dataset Preparation

## 2.1 Input dataset

The feature-engineering workflow uses the processed Gurgaon property dataset created during the earlier data-cleaning module.

The principal input files are:

```text
data/
├── processed/
│   ├── clean_v2.csv
│   ├── location_features.csv
│   └── clean_v3_location.csv
```

### `clean_v2.csv`

This file contains the cleaned property dataset after the earlier preprocessing steps.

It includes information such as:

* Property identifiers
* Latitude and longitude
* Property type
* Ownership type
* Number of bedrooms
* Number of bathrooms
* Number of balconies
* Furnishing status
* Facing direction
* Property age
* Floor information
* Property area
* Grouped locality
* Property price

The cleaned dataset contains **7,457 property records**.

### `location_features.csv`

This file contains geographical accessibility features calculated for the properties.

It contains 16 columns, including the property identifier and 15 additional columns representing distances, counts, and other accessibility-related measurements.

The file was checked for duplicate property identifiers and missing values. The final validation found:

* 7,457 rows
* No duplicate `prop_id` values
* No missing values in the accessibility feature file

### `clean_v3_location.csv`

This file combines the cleaned property dataset with the corresponding geographical accessibility features.

The property identifier, `prop_id`, was used to associate each property with its accessibility information.

The resulting dataset maintains **7,457 property records**.

---

## 2.2 Data integrity validation

Before engineering additional features, the dataset was checked to ensure that the property records and geographical features were correctly aligned.

The validation process included:

1. Checking the number of rows in the property and accessibility datasets.
2. Checking for duplicate property identifiers.
3. Checking for missing accessibility values.
4. Verifying that every property had a corresponding accessibility record.
5. Confirming that the merged dataset preserved the expected number of records.

These checks help prevent incorrect joins, duplicate records, and missing geographical information from affecting model training.

---

## 3. Basic Feature Engineering

## 3.1 Why basic feature engineering is necessary

Raw property attributes do not always express the relationships that a machine learning model needs to learn.

For example, the number of bedrooms and bathrooms can be considered separately, but their relationship may also provide useful information about a property.

Feature engineering creates additional variables from existing attributes to represent these relationships more explicitly.

In this project, the basic feature-engineering stage creates or supports the following features:

| Feature            | Description                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| `bed_bath_ratio`   | Represents the relationship between the number of bedrooms and bathrooms        |
| `total_rooms`      | Represents a combined room-count measure derived from available room attributes |
| `area_per_bedroom` | Represents the available area relative to the number of bedrooms                |
| `floor_ratio`      | Represents a normalized floor-related measure derived from floor information    |

These features provide alternative representations of the property's characteristics.

The exact mathematical definitions, including how zero values and missing values are handled, should follow the implementation in `src/features.py` or the original feature-engineering notebook.

### Example: Area per bedroom

A general representation of this feature is:

$$
\text{Area Per Bedroom} =
\frac{\text{Area in Square Feet}}{\text{Number of Bedrooms}}
$$

This can help distinguish properties with similar total areas but different bedroom configurations.

The implementation must handle zero bedroom counts safely to prevent division-by-zero errors.

### Example: Bedroom-bathroom relationship

A bedroom-to-bathroom relationship can provide information about the property's layout.

For example, two properties with the same area may differ in functionality if one has more bathrooms relative to its number of bedrooms.

The engineered feature gives the model an additional way to represent this relationship.

### Example: Floor ratio

A floor-related ratio can represent the position of a property within a building.

Its usefulness depends on the availability and interpretation of `floor_num` and `total_floor`.

Care must be taken when these values are missing or invalid.

## 3.2 Implementation

The reusable function for creating basic features is:

```python
add_basic_features(d)
```

This function is included in:

```text
src/features.py
```

It accepts a DataFrame and creates the basic engineered features according to the project's implementation.

The function was successfully imported and executed during validation.

The existing dataset contained 45 columns before and after the test invocation. This is consistent with adding features that already exist in the input data, rather than evidence that the function failed.

**Important:** Feature functions should be checked for repeatability. Running the same function twice should not unintentionally duplicate columns or change the meaning of existing features.

---

## 4. Geographical Accessibility Feature Engineering

## 4.1 Motivation

Property prices are influenced by more than the physical characteristics of a building.

Two properties with similar areas, bedroom counts, and property types may have different prices because of differences in their locations.

For example, a property located near a railway station, school, hospital, shopping mall, or major road may offer greater convenience than an otherwise similar property located far from these facilities.

To investigate this relationship, the project incorporates geographical accessibility features derived from OpenStreetMap-related geographical data.

This component is an important differentiator from a conventional real estate price prediction model that uses only property attributes.

## 4.2 Distance-based features

The following distance features were included in the accessibility dataset:

| Feature            | Meaning                               |
| ------------------ | ------------------------------------- |
| `dist_railway_km`  | Distance to a railway-related feature |
| `dist_bus_km`      | Distance to a bus-related feature     |
| `dist_school_km`   | Distance to a school                  |
| `dist_hospital_km` | Distance to a hospital                |
| `dist_mall_km`     | Distance to a mall                    |
| `dist_highway_km`  | Distance to a highway                 |
| `dist_mainroad_km` | Distance to a main road               |

These features express proximity in kilometres.

Distance features allow the model to learn whether the proximity of particular facilities is associated with property prices.

For example, the relationship between distance to a railway station and price may differ from the relationship between distance to a school and price.

These relationships should be learned from the data rather than assumed in advance.

### Distance interpretation

Generally:

* A smaller distance indicates greater proximity.
* A larger distance indicates that the relevant feature is farther away.
* A distance of zero may indicate that a property coordinate coincides with the mapped feature or that multiple records share a coordinate.

Distance values must be interpreted in the context of the geographical data and the method used to calculate them.

## 4.3 Amenity-count features

Distance alone does not describe the entire accessibility of a location.

For example, two properties may be equally close to their nearest school, but one may have several schools nearby while the other has only one.

To capture this distinction, the project includes amenity-count features.

| Feature          | Meaning                                    |
| ---------------- | ------------------------------------------ |
| `n_school_1km`   | Number of schools within 1 km              |
| `n_school_2km`   | Number of schools within 2 km              |
| `n_hospital_1km` | Number of hospitals within 1 km            |
| `n_hospital_2km` | Number of hospitals within 2 km            |
| `n_bus_1km`      | Number of bus-related features within 1 km |
| `n_bus_2km`      | Number of bus-related features within 2 km |
| `n_mall_1km`     | Number of malls within 1 km                |
| `n_mall_2km`     | Number of malls within 2 km                |

The count features represent the availability of nearby facilities.

A higher count may indicate that a location has access to a greater number of relevant facilities. However, the effect on property prices depends on the actual data and the type of amenity.

### Count validation

The count features were checked to ensure that the 1 km counts did not exceed their corresponding 2 km counts.

For example:

$$
n_{\text{school},1km}
\leq
n_{\text{school},2km}
$$

The same principle applies to the hospital, bus, and mall counts.

This is a useful consistency check because every facility within 1 km should also be within 2 km when both counts use the same definitions and geographical dataset.

## 4.4 Accessibility scores

The project also incorporates accessibility scores:

* `access_plain`
* `access_weighted`

These represent alternative ways of combining accessibility information.

A plain accessibility score can provide a combined representation of the available distance-based information.

A weighted score allows different accessibility components to contribute according to specified weights.

The precise calculation depends on the project's fitted parameters, feature columns, and weight configuration.

The implementation is handled by the following functions:

```python
fit_access_params(train, cols)

access_components(d, params)

add_access_features(d, params, weights)
```

The intended workflow is to estimate the required parameters from the training partition and apply the resulting transformation to the appropriate datasets.

This separation is important because the parameters should not be fitted using held-out test data.

## 4.5 Amenity density and total amenity counts

Additional features represent the concentration of nearby facilities.

| Feature               | Meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `amenity_density_1km` | A measure of nearby amenity density within 1 km               |
| `amenity_density_2km` | A measure of nearby amenity density within 2 km               |
| `n_total_1km`         | Total count of the amenities included in the 1 km aggregation |
| `n_total_2km`         | Total count of the amenities included in the 2 km aggregation |

Amenity density helps describe how concentrated the relevant facilities are around a property.

Total amenity counts provide an aggregate measure of nearby facilities.

These features may contain overlapping information. For example, a total amenity count and an amenity density measure may be strongly related.

Their usefulness therefore needs to be evaluated rather than assumed.

## 4.6 Accessibility classes

The project also includes an accessibility classification feature:

```text
access_class
```

This feature converts a continuous accessibility score into a categorical representation.

The implementation uses fitted bin edges:

```python
fit_bin_edges(train, score)

add_access_class(d, edges, score)
```

This makes it possible to represent accessibility using discrete classes rather than only continuous scores.

The exact class boundaries are determined by the fitted bin edges. They should be preserved when the feature transformation is reused in the application.

## 4.7 Accessibility density parameters

The density-related functions are:

```python
fit_density_params(train, count_bases)

add_density_features(d, params, count_bases)
```

These functions support fitting and applying the density transformation using the relevant count-based information.

As with the accessibility scores, the fitted parameters should be derived from training data and reused consistently.

## 4.8 Reusable accessibility feature functions

The complete set of reusable feature-engineering functions is:

| Function                                       | Purpose                                                  |
| ---------------------------------------------- | -------------------------------------------------------- |
| `add_basic_features(d)`                        | Creates basic property-derived features                  |
| `fit_access_params(train, cols)`               | Fits parameters used by the accessibility transformation |
| `access_components(d, params)`                 | Calculates accessibility components                      |
| `add_access_features(d, params, weights)`      | Adds accessibility-related features                      |
| `fit_density_params(train, count_bases)`       | Fits parameters for density-related features             |
| `add_density_features(d, params, count_bases)` | Adds density-related features                            |
| `fit_bin_edges(train, score)`                  | Determines accessibility classification boundaries       |
| `add_access_class(d, edges, score)`            | Adds the categorical accessibility class                 |

These functions are organized in:

```text
src/features.py
```

The module was successfully imported, and all eight functions were confirmed to be callable.

This makes the feature-engineering code easier to reuse outside the notebook.

---

## 5. Preparing the Machine Learning Dataset

## 5.1 Creating the modeling dataset

After combining the cleaned property information with the engineered geographical features, the modeling dataset contains:

* Original property attributes
* Geographical coordinates
* Basic engineered features
* Accessibility distances
* Amenity counts
* Accessibility scores
* Density-related features
* Accessibility classes
* Target price information

The feature-engineering notebook produced a combined DataFrame with:

$$
\boxed{7,457 \text{ rows} \times 45 \text{ columns}}
$$

The target variable is:

```text
price_lakhs
```

This represents property price in lakhs.

## 5.2 Logarithmic target transformation

Property prices commonly have a right-skewed distribution: most properties may be priced within a lower range, while a smaller number of expensive properties have substantially higher prices.

To reduce the effect of this skewness during model training, the project creates a logarithmic target:

$$
y_{\log} = \ln(1+y)
$$

where:

* \(y\) is the original property price in lakhs.
* \(y_{\log}\) is the transformed target.
* \(\ln\) is the natural logarithm.

The implementation is:

```python
import numpy as np

full["log_price"] = np.log1p(full["price_lakhs"])
```

The transformation can make the target distribution more manageable for modeling.

It also changes the scale on which errors are measured, so the final results must be interpreted carefully.

### Converting predictions back to the original scale

The model predicts `log_price`.

To convert the prediction back into lakhs, use:

$$
\hat{y} = e^{\hat{y}_{\log}} - 1
$$

In Python:

```python
predicted_price_lakhs = np.expm1(predicted_log_price)
```

The resulting value can be converted into crores by dividing by 100:

```python
predicted_price_crores = predicted_price_lakhs / 100
```

This conversion is essential when presenting predictions to users.

---

## 6. Preventing Data Leakage

Data leakage occurs when information that would not legitimately be available during model training is used to construct the inputs or fit the model.

Preventing leakage is essential for obtaining a realistic estimate of model performance.

## 6.1 Target-related columns

The following columns are not used as model inputs:

| Column           | Reason for exclusion                                                   |
| ---------------- | ---------------------------------------------------------------------- |
| `price_lakhs`    | Original target variable                                               |
| `log_price`      | Logarithmically transformed target                                     |
| `price_per_sqft` | Price-derived variable that can reveal information about the target    |
| `prop_id`        | Identifier rather than a meaningful predictive property characteristic |

The model should learn to predict price from property characteristics, not from information calculated using the price itself.

## 6.2 Training and test partitions

The dataset was divided into training and test partitions.

| Partition    | Number of records | Percentage |
| ------------ | ----------------: | ---------: |
| Training set |             5,965 |        80% |
| Test set     |             1,492 |        20% |
| Total        |             7,457 |       100% |

The split was created using a fixed random state so that the partition could be reproduced.

The training set was used for:

* Feature selection experiments
* Feature importance analysis
* Feature-group ablation
* Algorithm comparison
* Hyperparameter tuning
* Final model fitting

The test set was reserved for final evaluation.

**Important:** The held-out test set must not be used to choose features or tune hyperparameters. Its purpose is to estimate how the selected model performs on data that was not used during model selection.

## 6.3 Fitting preprocessing correctly

Categorical encoding, imputation, scaling where applicable, and other learned preprocessing operations should be fitted using training data only.

A scikit-learn `Pipeline` helps enforce this process by keeping preprocessing and model prediction together.

This is particularly important when categorical features must be encoded before they can be used by the Random Forest model.

---

## 7. Feature Group Definition

To evaluate the contribution of different feature groups, the project organizes input features into logical categories.

## 7.1 Base numerical features

The base numerical group contains seven features:

```python
BASE_NUM = [
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_sqft",
]
```

These represent fundamental property characteristics.

## 7.2 Geographical coordinates

The geographical group contains:

```python
GEO = [
    "latitude",
    "longitude",
]
```

Coordinates provide the model with the approximate geographical position of each property.

They may help the model learn spatial price patterns across Gurgaon.

## 7.3 Base categorical features

The base categorical group contains six features:

```python
BASE_CAT = [
    "property_type",
    "owntype",
    "furnish",
    "facing",
    "age",
    "locality_grouped",
]
```

These represent property classifications, ownership information, furnishing status, facing direction, age category, and grouped locality.

Although some of these variables may be stored as numbers, they represent categories rather than continuous numerical measurements.

For example, an encoded furnishing category should not automatically be interpreted as being twice as much as another category simply because its code is twice as large.

The preprocessing pipeline must therefore treat these variables as categorical features.

## 7.4 Accessibility numerical features

The accessibility numerical group contains 21 features:

```python
AMEN_NUM = [
    "dist_railway_km",
    "dist_bus_km",
    "dist_school_km",
    "dist_hospital_km",
    "dist_mall_km",
    "dist_highway_km",
    "dist_mainroad_km",
    "access_plain",
    "access_weighted",
    "n_school_1km",
    "n_school_2km",
    "n_hospital_1km",
    "n_hospital_2km",
    "n_bus_1km",
    "n_bus_2km",
    "n_mall_1km",
    "n_mall_2km",
    "amenity_density_1km",
    "amenity_density_2km",
    "n_total_1km",
    "n_total_2km",
]
```

These features represent distances, accessibility scores, counts, and amenity density.

## 7.5 Accessibility categorical features

The accessibility categorical group contains:

```python
AMEN_CAT = [
    "access_class",
]
```

This feature represents the categorical accessibility classification.

## 7.6 Base feature group

The base feature group combines the base numerical and categorical features:

$$
\text{BASE\_FEATURES}
=
\text{BASE\_NUM}
\cup
\text{BASE\_CAT}
$$

It contains:

$$
7+6=\boxed{13 \text{ features}}
$$

## 7.7 Core feature group

The core feature group adds latitude and longitude to the base features:

$$
\text{BASE\_GEO\_FEATURES}
=
\text{BASE\_FEATURES}
\cup
\text{GEO}
$$

It contains:

$$
13+2=\boxed{15 \text{ features}}
$$

This became the selected feature set for the final model.

## 7.8 Full feature group

The full feature group combines the core features with all 22 accessibility features:

$$
\text{FULL\_FEATURES}
=
\text{BASE\_GEO\_FEATURES}
\cup
\text{AMEN\_NUM}
\cup
\text{AMEN\_CAT}
$$

It contains:

$$
15+21+1=\boxed{37 \text{ features}}
$$

The 37-feature group was used to investigate whether the accessibility information improved prediction performance.

---

## 8. Feature Correlation Analysis

## 8.1 Purpose

Correlation analysis helps identify relationships between numerical variables.

Highly correlated features may contain overlapping information.

For example, a total amenity count and a density measure may be strongly related because both describe the availability of nearby facilities.

Correlation analysis can help:

* Identify potentially redundant numerical features.
* Understand relationships between features.
* Investigate associations between numerical variables and the target.
* Guide further feature-selection experiments.

## 8.2 Correlation interpretation

The Pearson correlation coefficient measures the strength and direction of a linear relationship between two variables.

Its value ranges from -1 to +1.

| Correlation | Interpretation                      |
| ----------- | ----------------------------------- |
| Close to +1 | Strong positive linear relationship |
| Close to -1 | Strong negative linear relationship |
| Close to 0  | Weak or absent linear relationship  |

A correlation coefficient close to zero does not necessarily mean that two variables are unrelated. Their relationship may be nonlinear.

Similarly, correlation does not establish causation.

## 8.3 Correlation-based feature pruning

The project investigated redundancy among numerical features.

The analysis identified candidates such as:

* `dist_mainroad_km`
* `access_plain`
* `n_total_1km`
* `n_total_2km`

These were considered in the correlation-pruning work.

However, correlation alone is not sufficient to decide whether a feature should be removed. A feature that is correlated with another feature may still contribute useful information to a nonlinear model.

The project therefore also evaluated feature combinations empirically.

---

## 9. Random Forest Feature Importance

## 9.1 Purpose

Feature importance analysis helps identify which input features are useful to a trained model.

The project used Random Forest feature-importance analysis as part of the feature-selection process.

A Random Forest consists of multiple decision trees trained on different samples and feature subsets.

Each tree learns relationships between the inputs and target. Combining many trees can capture nonlinear relationships and interactions between property characteristics.

## 9.2 Permutation feature importance

The project performed permutation feature importance analysis.

The general procedure is:

1. Train a model using the training data.
2. Measure its baseline predictive performance.
3. Randomly shuffle one feature's values.
4. Measure performance again.
5. Calculate how much performance deteriorates.
6. Repeat the procedure as appropriate to obtain a more stable estimate.

If shuffling a feature causes a substantial reduction in model performance, the model may rely on that feature for prediction.

Permutation importance is useful because it evaluates feature importance in terms of the model's actual predictive performance rather than only its internal tree statistics.

## 9.3 Limitations

Feature importance must be interpreted carefully.

* Correlated features can share predictive information.
* A feature may be useful in combination with another feature.
* Importance depends on the model and evaluation dataset.
* A low importance score does not prove that a variable is irrelevant in every possible model.
* Feature importance does not establish a causal relationship with property prices.

The analysis was therefore combined with feature-group ablation and model comparison.

---

## 10. Feature-Group Ablation Experiments

## 10.1 Purpose

Feature-group ablation measures the effect of adding or removing groups of features.

Rather than evaluating features only individually, the project compared models trained with different combinations of property, geographical, and accessibility information.

This helps answer the question:

**Do geographical coordinates and accessibility features improve property price prediction beyond the basic property attributes?**

## 10.2 Core feature set

The core set contains 15 features:

* Seven base numerical features
* Six base categorical features
* Two geographical coordinates

It includes the fundamental property information and location coordinates without the additional accessibility measurements.

## 10.3 Full accessibility feature set

The full set contains 37 features.

It includes all 15 core features plus 22 accessibility-related features.

These additional features represent:

* Distance to nearby facilities
* Amenity counts
* Accessibility scores
* Amenity density
* Total amenity counts
* Accessibility classification

## 10.4 Why ablation is important

Adding more features does not automatically improve a machine learning model.

Additional variables may introduce:

* Redundant information
* Noisy measurements
* Unnecessary model complexity
* Greater computational requirements
* Overfitting risk

Ablation experiments provide empirical evidence about whether the additional features are worthwhile.

The comparison was conducted using the training and validation process rather than using the final held-out test set to select the winning feature group.

The resulting selected feature set for the final model was the 15-feature core set.

This does not mean that geographical accessibility is unimportant. It means that, under the tested modeling configuration, the additional accessibility features were not selected for the final production model.

---

## 11. Additional Feature Validation: `area_is_carpet`

The project also investigated the binary feature:

```text
area_is_carpet
```

This variable indicates whether the reported property area is identified as carpet area.

Its distribution in the cleaned dataset was:

| Value   | Count |
| ------- | ----: |
| 0       | 5,555 |
| 1       | 1,902 |
| Missing |     0 |
| Total   | 7,457 |

The feature was added to candidate feature configurations and evaluated.

The comparison did not show a measurable improvement in Random Forest performance.

Consequently, `area_is_carpet` was excluded from the selected 15-feature final model.

This decision illustrates an important feature-engineering principle: a feature should not be retained solely because it seems intuitively useful. Its contribution should be assessed using a suitable evaluation procedure.

---

## 12. Comparing Candidate Feature Sets

The project defined candidate feature configurations to evaluate alternative representations of the data.

The main configurations were:

| Feature set                 | Number of features | Purpose                                   |
| --------------------------- | -----------------: | ----------------------------------------- |
| Base features               |                 13 | Evaluate fundamental property information |
| Core features               |                 15 | Add latitude and longitude                |
| Full accessibility features |                 37 | Evaluate all accessibility information    |

The experiments were used to determine whether the additional feature groups improved prediction performance.

The selected final feature configuration was:

```python
FINAL_CORE_FEATURES = BASE_GEO_FEATURES.copy()

FINAL_FEATURES = FINAL_CORE_FEATURES.copy()
```

The final feature list contains:

```python
FINAL_FEATURES = [
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_sqft",
    "property_type",
    "owntype",
    "furnish",
    "facing",
    "age",
    "locality_grouped",
    "latitude",
    "longitude",
]
```

The selection preserves both property characteristics and geographical location.

The full accessibility feature set remains useful for research, comparison, and potential future model improvements.

---

## 13. Comparing Machine Learning Algorithms

## 13.1 Objective

After selecting the core feature set, the project compared candidate machine learning algorithms.

The objective was to determine which model provided a suitable balance of predictive accuracy, robustness, and deployment practicality.

The target used for training was:

```text
log_price
```

The algorithm comparison was conducted using the established training and validation workflow.

## 13.2 Algorithms considered

The project planned and investigated conventional regression approaches, including:

### Linear Regression

Linear Regression models the target as a linear combination of input features.

Advantages:

* Simple to understand.
* Fast to train.
* Provides a useful baseline.
* Can help identify broad relationships.

Limitations:

* Cannot naturally capture all nonlinear relationships.
* May require additional feature transformations to represent complex interactions.

### Decision Tree Regressor

A Decision Tree learns a sequence of conditions that divide the input space into smaller regions.

Advantages:

* Can capture nonlinear relationships.
* Supports feature interactions.
* Does not require numerical features to be scaled for its splits.

Limitations:

* Can overfit the training data.
* May be sensitive to changes in the training sample.

### Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

Advantages:

* Captures nonlinear relationships.
* Can model interactions between features.
* Often performs well on structured tabular datasets.
* Reduces some of the instability associated with a single decision tree.

Limitations:

* Can require more memory and computation than simpler models.
* Is less directly interpretable than a linear model.
* May produce larger errors on unusual or extreme properties.

### Gradient Boosting Regressor

Gradient Boosting builds a sequence of models in which later models attempt to improve on the errors of earlier ones.

Advantages:

* Can capture nonlinear patterns.
* Can perform strongly on structured datasets.
* Supports iterative improvement of model predictions.

Limitations:

* Performance depends on appropriate hyperparameters.
* Requires careful validation to avoid overfitting.

## 13.3 Evaluation metrics

The models were evaluated using regression metrics.

### R-squared

$$
R^2 =
1 -
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

R-squared measures how well the predictions account for variation in the target relative to predicting its mean.

Higher values generally indicate better performance under the same evaluation conditions.

R-squared can be negative if a model performs worse than the mean-prediction baseline.

### Mean Absolute Error

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

MAE measures the average absolute difference between actual and predicted values.

It is relatively easy to interpret because it uses the same units as the target.

### Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
$$

RMSE penalizes large errors more heavily than MAE because the errors are squared before averaging.

This is useful for a real estate project because very large prediction errors may have substantial practical consequences.

### Metric interpretation across target scales

The model was trained on the logarithmically transformed target.

Therefore, metrics calculated on `log_price` and metrics calculated on the original `price_lakhs` scale measure different quantities.

They should be reported separately and should not be compared directly as if they used the same units.

---

## 14. Random Forest Hyperparameter Tuning

## 14.1 Purpose

After the algorithm comparison, the project tuned the Random Forest model.

Hyperparameters determine aspects of the model's structure and training process.

The objective was to identify a suitable configuration using the training and validation workflow.

## 14.2 Selected hyperparameters

The selected configuration was:

```python
BEST_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}
```

### Explanation of the parameters

| Parameter           | Selected value | Explanation                                                  |
| ------------------- | -------------- | ------------------------------------------------------------ |
| `n_estimators`      | 400            | Number of trees in the forest                                |
| `max_depth`         | `None`         | No explicit maximum tree depth                               |
| `min_samples_split` | 2              | Minimum number of samples required to split an internal node |
| `min_samples_leaf`  | 1              | Minimum number of samples required at a leaf                 |
| `max_features`      | `1.0`          | Each split considers all input features                      |

### Number of trees

The model uses 400 decision trees.

Increasing the number of trees can improve the stability of predictions, although it also increases training time, prediction time, and model size.

### Maximum depth

Setting `max_depth=None` allows trees to continue splitting until other stopping conditions are met.

This provides flexibility but may increase the risk of overfitting when other constraints are insufficient.

### Minimum samples for splitting

The value `min_samples_split=2` allows a node to split when it contains at least two samples, subject to the remaining tree-building conditions.

### Minimum samples per leaf

The value `min_samples_leaf=1` allows a leaf to contain a single training sample.

This makes the trees flexible enough to represent detailed patterns in the training data.

### Features considered at each split

The value `max_features=1.0` allows the split procedure to consider all available input features.

This setting was part of the selected configuration.

---

## 15. Comparing the Tuned Core Model and Accessibility Model

The project compared the selected core feature configuration with the larger accessibility configuration.

The comparison investigated whether adding the accessibility features provided enough improvement to justify the additional complexity.

The two configurations were:

| Model configuration | Input features |
| ------------------- | -------------- |
| Core model          | 15             |
| Accessibility model | 37             |

The final production model uses the 15-feature core configuration.

This decision keeps the deployed model's input requirements relatively simple.

The accessibility features remain valuable components of the research because they allow the project to investigate how proximity to amenities relates to property prices.

Future work could explore alternative accessibility definitions, improved geographical data, additional amenities, and different model architectures.

---

## 16. Final Model Training

## 16.1 Final feature configuration

The final model uses the 15 core features.

These features are divided into two groups for preprocessing.

### Numerical features

```python
FINAL_NUM_FEATURES = [
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_sqft",
    "latitude",
    "longitude",
]
```

There are nine numerical features.

### Categorical features

```python
FINAL_CAT_FEATURES = [
    "property_type",
    "owntype",
    "furnish",
    "facing",
    "age",
    "locality_grouped",
]
```

There are six categorical features.

Therefore:

$$
9+6=\boxed{15 \text{ input features}}
$$

The numerical and categorical feature lists together match the final model's expected input columns.

## 16.2 Preprocessing pipeline

The final model is stored as a scikit-learn `Pipeline`.

A pipeline combines the preprocessing operations with the trained estimator.

This is important because the same transformations must be applied during both training and prediction.

For example, if categorical variables are encoded during training, new property inputs must be encoded using the same fitted transformation.

The application should not independently invent new category codes or use a different encoding procedure.

## 16.3 Final training procedure

The final model was fitted on all 5,965 records in the original training partition.

The 1,492 held-out test records were reserved for final evaluation.

This allows the final model to use the entire training partition after the model-selection process is complete while preserving an independent test set for reporting performance.

---

## 17. Final Model Evaluation

## 17.1 Held-out test results

The final Random Forest model was evaluated on the 1,492-record held-out test set.

The observed metrics were:

### Metrics on the logarithmic target scale

| Metric    |   Result |
| --------- | -------: |
| R-squared | 0.912288 |
| MAE       | 0.118397 |
| RMSE      | 0.182717 |

### Metrics on the original property-price scale

| Metric    |        Result |
| --------- | ------------: |
| R-squared |      0.902345 |
| MAE       |  ₹43.34 lakhs |
| RMSE      | ₹100.15 lakhs |

The original-scale metrics are particularly useful when explaining the model to users because they express errors in the same units as property prices.

## 17.2 Interpretation of R-squared

The original-scale R-squared value is approximately:

$$
R^2 = 0.9023
$$

This indicates that the model performs substantially better than a simple mean-prediction baseline on the held-out test data under the R-squared metric.

It does not mean that every prediction is 90.23% accurate.

R-squared is a measure of performance relative to variation in the target, not a percentage accuracy score for individual property predictions.

## 17.3 Interpretation of MAE

The original-scale MAE is approximately:

$$
MAE = ₹43.34 \text{ lakhs}
$$

This means that the average absolute difference between the predicted and actual property prices is approximately ₹43.34 lakhs across the test records.

It does not mean that every prediction is wrong by this amount. Individual errors may be much smaller or much larger.

## 17.4 Interpretation of RMSE

The original-scale RMSE is approximately:

$$
RMSE = ₹100.15 \text{ lakhs}
$$

RMSE is larger than MAE because it gives more weight to large errors.

The difference between the two metrics suggests that some predictions have substantially larger errors than the typical absolute error.

This is particularly relevant in real estate, where properties vary greatly in price and a small number of expensive properties can produce large absolute errors.

---

## 18. Prediction Error Analysis

## 18.1 Purpose

After evaluating the model, the project investigated its prediction errors.

Error analysis is important because aggregate metrics alone do not explain where a model performs well or poorly.

For example, a model may perform well for moderately priced properties but produce large errors for luxury properties.

Understanding these patterns helps identify areas for future improvement.

## 18.2 Actual versus predicted prices

The error-analysis results were:

| Measure                |          Result |
| ---------------------- | --------------: |
| Mean actual price      |   ₹350.70 lakhs |
| Mean predicted price   |   ₹344.85 lakhs |
| Mean signed error      |    -₹5.85 lakhs |
| Median absolute error  |    ₹20.31 lakhs |
| Maximum absolute error | ₹1,880.23 lakhs |

The mean signed error was calculated using the convention:

$$
\text{Signed Error}
=
\text{Predicted Price}
-
\text{Actual Price}
$$

Under this convention, a negative value indicates underprediction.

The mean signed error of approximately -₹5.85 lakhs indicates that the model's average prediction was slightly below the average actual price.

However, a small average signed error does not guarantee that individual predictions are close to their actual values. Overpredictions and underpredictions can offset one another.

## 18.3 Median absolute error

The median absolute error was approximately ₹20.31 lakhs.

This means that half of the absolute prediction errors were no greater than approximately ₹20.31 lakhs, while the other half were at least this large.

The median is less sensitive to extreme errors than the mean.

This explains why the median absolute error can be substantially lower than the mean absolute error.

## 18.4 Large errors for expensive properties

The analysis found that prediction errors tended to increase for higher-priced properties, particularly for properties above ₹10 crore.

This is an important observation.

Expensive properties may have unusual characteristics, unique locations, luxury amenities, or other factors that are not fully represented in the available dataset.

Because the model is trained on historical examples, it may have difficulty predicting unusual properties when similar examples are limited.

Large errors should be investigated, but they should not automatically be treated as evidence that the corresponding records are invalid.

Removing expensive properties merely because the model predicts them poorly could introduce selection bias and reduce the model's usefulness for high-value properties.

## 18.5 Future improvements suggested by error analysis

Potential improvements include:

* Collecting more examples of high-value properties.
* Investigating whether the most expensive listings contain unusual data patterns.
* Evaluating prediction errors across price ranges.
* Exploring additional location and property-quality features.
* Comparing alternative model families.
* Reporting uncertainty alongside point predictions.
* Testing whether separate models are useful for different price segments.

These are potential future improvements rather than completed steps.

---

## 19. Saving the Final Model

## 19.1 Model file

The trained model was saved to:

```text
models/final_price_model.joblib
```

The file contains the fitted scikit-learn pipeline, including its trained preprocessing components and Random Forest estimator.

The file size was approximately 193.8 MB.

The large file size reflects the stored model and its associated fitted objects.

## 19.2 Model metadata

The accompanying metadata file is:

```text
models/final_model_metadata.json
```

This file is intended to preserve information about the model and its configuration.

Keeping metadata separate from the serialized model makes it easier to document the model's feature schema, target, and evaluation results.

## 19.3 Reloading the model

The model was loaded successfully using Joblib:

```python
import joblib

model = joblib.load(
    r"D:\real-estate-price-prediction\models\final_price_model.joblib"
)
```

The loaded object was confirmed to be a scikit-learn `Pipeline`.

The model's expected feature count was verified:

```text
Expected feature count: 15
```

The expected input features were:

```python
[
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_sqft",
    "property_type",
    "owntype",
    "furnish",
    "facing",
    "age",
    "locality_grouped",
    "latitude",
    "longitude",
]
```

## 19.4 Prediction consistency verification

The reloaded model was compared against the original model's predictions.

The maximum observed prediction difference was approximately:

$$
5.33 \times 10^{-15}
$$

The predictions also passed the numerical equivalence check.

This confirms that the saved model reproduces the original model's predictions to floating-point precision for the tested inputs.

This verification is important because a model that saves and reloads correctly is easier to deploy reliably.

---

## 20. Preparing the Modeling Dataset for Reuse

The feature-engineering workflow also prepares a modeling dataset containing the final selected features and target information.

The intended output is:

```text
data/processed/model_data.csv
```

The file contains:

* `prop_id`
* The 15 final model features
* `price_lakhs`
* `log_price`
* `split`

The `split` column identifies whether a record belongs to the training or test partition.

Expected dimensions:

$$
\boxed{7,457 \text{ rows} \times 19 \text{ columns}}
$$

The expected partition counts are:

| Split | Records |
| ----- | ------: |
| Train |   5,965 |
| Test  |   1,492 |
| Total |   7,457 |

This dataset is useful for reproducing experiments and inspecting the final model inputs.

**Important:** Although the modeling dataset contains target columns for evaluation and analysis, `price_lakhs` and `log_price` must never be included in the prediction input features.

The `split` column is also metadata, not a model input.

The existence of the output file should be verified separately when documenting the final project files.

---

## 21. Reusable Feature-Engineering Module

## 21.1 Why a separate Python module is useful

Initially, feature-engineering functions are often developed and tested in a notebook.

However, notebooks are primarily intended for experimentation and analysis.

A deployed application should be able to reuse the required functions without depending on the notebook's execution history.

For this reason, the project organizes the feature-engineering functions into:

```text
src/features.py
```

This allows the functions to be imported into other Python files.

## 21.2 Available functions

The module contains eight functions:

```python
from src.features import (
    add_basic_features,
    fit_access_params,
    access_components,
    add_access_features,
    fit_density_params,
    add_density_features,
    fit_bin_edges,
    add_access_class,
)
```

All eight functions were successfully imported during validation.

The module path was confirmed as:

```text
D:\real-estate-price-prediction\src\features.py
```

## 21.3 Import-path handling

During development, the notebook initially returned:

```text
ModuleNotFoundError: No module named 'src'
```

This occurred because Python could not locate the project package from the notebook's current import path.

The problem was resolved by adding the project directory to `sys.path`:

```python
import sys
from pathlib import Path

PROJECT_DIR = Path(r"D:\real-estate-price-prediction")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
```

The imports then succeeded.

This is a development-environment issue rather than evidence of a problem with the feature-engineering functions themselves.

For the deployed application, imports should be configured relative to the project structure and execution environment.

---

## 22. Important Distinction: Research Features Versus Deployment Features

One of the main outcomes of this module is the distinction between the full research dataset and the selected model's actual inputs.

The research dataset contains 45 columns, including:

* Original property attributes
* Target variables
* Basic engineered features
* Accessibility distances
* Amenity counts
* Accessibility scores
* Density features
* Accessibility classes

The full candidate feature set contains 37 input features.

However, the selected final model uses only 15 input features.

This distinction is important for the deployment interface.

The final model does not require the application to calculate all 37 candidate features before making a prediction.

Instead, the application must provide the following 15 features:

1. Bedroom count
2. Bathroom count
3. Balcony count
4. Floor number
5. Total floors
6. Total landmark count
7. Area in square feet
8. Property type
9. Ownership type
10. Furnishing status
11. Facing direction
12. Property age category
13. Grouped locality
14. Latitude
15. Longitude

The deployed model's preprocessing pipeline should then handle the input transformations it was trained to perform.

This reduces the complexity of the prediction interface and avoids calculating unnecessary accessibility features for the selected model.

---

## 23. Final Project Directory Structure

The relevant project structure is organized as follows:

```text
real-estate-price-prediction/
│
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│       ├── clean_v2.csv
│       ├── location_features.csv
│       ├── clean_v3_location.csv
│       └── model_data.csv
│
├── notebooks/
│   ├── 03_eda.ipynb
│   ├── 04_location_features.ipynb
│   └── 05_feature_engineering.ipynb
│
├── src/
│   └── features.py
│
├── models/
│   ├── final_price_model.joblib
│   └── final_model_metadata.json
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   └── ...
│
└── README.md
```

The listed notebook and model filenames reflect the project's current workflow. Files such as `model_data.csv`, `streamlit_app.py`, or other report outputs should be included in the final repository only after their creation and validation have been confirmed.

---

## 24. Challenges and Lessons Learned

## 24.1 More features do not always mean better predictions

One of the main lessons from the project is that adding features does not guarantee better model performance.

The accessibility feature set increased the number of candidate inputs from 15 to 37.

However, the final model selected the smaller core feature set.

This demonstrates why feature engineering should be combined with empirical evaluation.

## 24.2 Feature meaning matters

Some variables are numerical representations of categories.

Treating category codes as continuous numerical measurements can introduce misleading relationships.

The project therefore separates numerical and categorical features before preprocessing.

## 24.3 Geographical data requires validation

Geographical features depend on the quality and completeness of the underlying map data.

Distances and amenity counts should be checked for:

* Missing values
* Negative distances or counts
* Incorrect coordinate alignment
* Duplicate property identifiers
* Unexpected zero distances
* Inconsistent counts across distance ranges

These checks help identify possible data-quality issues before modeling.

## 24.4 Data leakage must be prevented

Price-derived features can make a model appear more accurate than it really is.

The project excludes the target variables and price-derived columns from the model inputs.

The held-out test partition is also reserved for final evaluation.

## 24.5 Error analysis is essential

A strong overall R-squared value does not guarantee reliable predictions for every property.

The prediction-error analysis revealed larger errors for high-priced properties.

This shows why a model should be evaluated using several metrics and examined across different price ranges.

## 24.6 Model reproducibility matters

Saving the trained pipeline and confirming that it produces consistent predictions after reloading are important steps toward reliable deployment.

The project also separates reusable feature-engineering functions from the notebook.

These practices make the workflow easier to maintain and reproduce.

---

## 25. Final Outcomes of Module 4

The feature-engineering module produced the following major outcomes.

| Component                              | Outcome                                                       |
| -------------------------------------- | ------------------------------------------------------------- |
| Modeling dataset                       | 7,457 property records                                        |
| Train/test split                       | 5,965 training records and 1,492 test records                 |
| Target transformation                  | `log_price = log1p(price_lakhs)`                              |
| Basic feature engineering              | Property-derived features created                             |
| Geographical feature engineering       | Distances, counts, accessibility scores, and density features |
| Accessibility classification           | `access_class` generated using fitted bin edges               |
| Correlation analysis                   | Numerical relationships and redundancy investigated           |
| Feature importance                     | Random Forest permutation importance performed                |
| Feature ablation                       | Core and full accessibility feature groups compared           |
| Additional feature validation          | `area_is_carpet` tested and excluded from the selected model  |
| Final input configuration              | 15 core features                                              |
| Candidate accessibility configuration  | 37 features                                                   |
| Selected algorithm                     | Random Forest Regressor                                       |
| Selected number of trees               | 400                                                           |
| Test R-squared on log scale            | 0.912288                                                      |
| Test MAE on log scale                  | 0.118397                                                      |
| Test RMSE on log scale                 | 0.182717                                                      |
| Test R-squared on original price scale | 0.902345                                                      |
| Test MAE on original price scale       | ₹43.34 lakhs                                                  |
| Test RMSE on original price scale      | ₹100.15 lakhs                                                 |
| Saved model                            | `models/final_price_model.joblib`                             |
| Model metadata                         | `models/final_model_metadata.json`                            |
| Reusable feature functions             | Eight functions in `src/features.py`                          |
| Model reload verification              | Passed for the tested inputs                                  |

---

## 26. Conclusion

Module 4 transformed the cleaned Gurgaon real estate dataset into a machine-learning-ready representation and established a reproducible modeling workflow.

The work included basic property feature engineering, geographical accessibility analysis, amenity-count creation, accessibility scoring, feature correlation analysis, permutation feature importance, feature-group ablation, candidate feature comparison, algorithm evaluation, Random Forest hyperparameter tuning, final model evaluation, and prediction-error analysis.

Although the project developed a comprehensive set of 37 candidate input features, the selected final model uses 15 core features consisting of property characteristics, categorical attributes, and geographical coordinates.

The final Random Forest pipeline achieved an R-squared value of approximately 0.9023 on the original property-price scale, with an MAE of approximately ₹43.34 lakhs and an RMSE of approximately ₹100.15 lakhs on the held-out test set.

The model was saved and successfully reloaded, and its predictions were verified for numerical consistency. The eight feature-engineering functions were also organized into a reusable Python module.

These results establish the foundation for the next stage: developing a Streamlit application that accepts property details and uses the saved model to estimate a property's price.

The model remains an estimation tool rather than a substitute for professional property valuation. Its predictions are limited by the quality, coverage, and representativeness of the available dataset, especially for unusual and high-value properties.


# Module 6 —  Model Training

## 1. Introduction

The Model Training module is a major component of the Real Estate Price Prediction project. Its purpose is to develop, train, compare, optimize, and evaluate machine learning models that predict residential property prices in Gurgaon, India.

The module uses the cleaned and feature-engineered dataset produced during the earlier stages of the project. It applies data preprocessing, regression algorithms, cross-validation, hyperparameter tuning, and performance evaluation to identify a suitable model for predicting property prices.

Four regression algorithms are investigated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

The models are evaluated using statistical performance metrics, including the coefficient of determination (R²), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

The selected model is saved together with its preprocessing pipeline so that the trained model can subsequently be integrated into the Streamlit application.

The main objective is not simply to achieve a high training score, but to develop a model that generalizes well to previously unseen properties.

---

## 2. Objectives of the Module

The primary objectives of the Model Training module are:

* Load the processed property dataset and verify the training and testing partitions.
* Identify numerical and categorical input features.
* Build a reusable preprocessing pipeline.
* Handle missing values systematically.
* Convert categorical variables into a numerical representation suitable for machine learning.
* Train multiple regression algorithms and compare their performance.
* Establish a baseline using cross-validation.
* Tune model hyperparameters to improve predictive performance.
* Evaluate the tuned models using consistent validation procedures.
* Select an appropriate final model.
* Transform predicted prices back to their original scale.
* Save the trained models, hyperparameters, evaluation results, and metadata.
* Investigate whether additional features improve model performance.
* Prepare the training workflow for reuse outside the Jupyter notebook.

---

## 3. Dataset Overview

### 3.1 Source Dataset

The module uses the processed dataset:

`data/processed/model_data.csv`

This dataset was generated during the earlier data-cleaning, exploratory data analysis, location-feature extraction, and feature-engineering stages.

The processed dataset contains **7,457 property records and 19 columns**.

### 3.2 Dataset Dimensions

| Property                   | Value |
| -------------------------- | ----: |
| Total property records     | 7,457 |
| Total columns              |    19 |
| Training records           | 5,965 |
| Testing records            | 1,492 |
| Numerical input features   |     9 |
| Categorical input features |     6 |
| Total input features       |    15 |

The dataset includes the input features, target variables, property identifier, and train/test split information.

### 3.3 Important Columns

| Column                 | Description                              |
| ---------------------- | ---------------------------------------- |
| `prop_id`              | Unique property identifier               |
| `bedroom_num`          | Number of bedrooms                       |
| `bathroom_num`         | Number of bathrooms                      |
| `balcony_num`          | Number of balconies                      |
| `floor_num`            | Floor number of the property             |
| `total_floor`          | Total number of floors in the building   |
| `total_landmark_count` | Aggregate count of nearby landmarks      |
| `area_sqft`            | Property area in square feet             |
| `latitude`             | Geographic latitude                      |
| `longitude`            | Geographic longitude                     |
| `property_type`        | Type of residential property             |
| `owntype`              | Ownership category                       |
| `furnish`              | Furnishing category                      |
| `facing`               | Property-facing direction                |
| `age`                  | Property age category                    |
| `locality_grouped`     | Grouped locality information             |
| `price_lakhs`          | Actual property price in lakhs           |
| `log_price`            | Log-transformed property price           |
| `split`                | Indicates training or testing membership |

The property identifier and split column are not used as predictive features.

---

## Part A: Setup and Data Verification

## 4. Purpose

Part A establishes the foundation for the model-training workflow. Before training any model, it is necessary to verify the processed dataset, feature configuration, target variable, and train/test partition.

This step ensures that the training process uses the expected data and that the evaluation results can be reproduced.

## 4.1 Loading the Dataset

The processed dataset is loaded into a Pandas DataFrame.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/model_data.csv")

print("Dataset shape:", df.shape)
print(df.head())
```

The expected dataset shape is:

`(7457, 19)`

The first few rows are inspected to confirm that the columns contain the expected values.

## 4.2 Loading the Feature Configuration

The numerical and categorical feature lists are maintained in:

`models/feature_config.json`

The configuration defines which columns are used as model inputs.

The numerical features are:

```python
numeric_features = [
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "floor_num",
    "total_floor",
    "total_landmark_count",
    "area_sqft",
    "latitude",
    "longitude",
]
```

The categorical features are:

```python
categorical_features = [
    "property_type",
    "owntype",
    "furnish",
    "facing",
    "age",
    "locality_grouped",
]
```

Separating features by data type allows the preprocessing pipeline to apply the appropriate transformation to each group.

## 4.3 Separating Training and Testing Data

The dataset contains a `split` column that identifies whether each record belongs to the training or testing set.

The data is separated using this column rather than generating a new random split during every training run.

```python
train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

y_train = train_df["log_price"].copy()
y_test = test_df["log_price"].copy()
```

The resulting dimensions are:

* `X_train`: 5,965 rows × 15 columns
* `X_test`: 1,492 rows × 15 columns
* `y_train`: 5,965 target values
* `y_test`: 1,492 target values

The training set is used to fit the models. The testing set is reserved for evaluating the final model on previously unseen observations.

## 4.4 Avoiding Data Leakage

Data leakage occurs when information that would not legitimately be available during prediction influences model training.

The following precautions are important:

* The actual price is excluded from the input feature matrix.
* `log_price` is used as the target during log-target model training.
* `price_lakhs` is not included as an input feature.
* The property identifier is excluded from the model inputs.
* The split column is used only to separate the data.
* Imputation and scaling parameters are learned from the training data rather than the complete dataset.
* The test set is not used to fit the preprocessing pipeline or tune the model.

These practices help ensure that the reported performance is a meaningful estimate of how the model may perform on unseen properties.

---

# Part B: Build the Preprocessing Pipeline

## 5. Purpose

Machine learning algorithms generally require input data to be represented in a suitable numerical format.

However, the real-estate dataset contains both numerical and categorical variables. Some values are missing, and categorical values cannot be passed directly to most regression algorithms.

Part B creates a preprocessing pipeline to handle these requirements consistently.

The pipeline uses:

1. Numerical missing-value imputation.
2. Numerical scaling.
3. Categorical missing-value imputation.
4. One-hot encoding of categorical variables.

A `ColumnTransformer` combines these operations into a single preprocessing object.

## 5.1 Numerical Feature Processing

Numerical features include property size, number of rooms, floor information, and geographic coordinates.

### Median Imputation

Missing numerical values are replaced using the median calculated from the corresponding feature in the training data.

The median is less sensitive to extreme values than the mean and provides a practical method for filling missing numerical observations.

For example, missing `floor_num` values are imputed using the training-set median.

Missing floor numbers are not automatically replaced with zero because zero could represent an actual numerical value rather than missing information.

### Standardization

Numerical features are standardized using `StandardScaler`.

Standardization transforms a numerical variable approximately according to:

$$
z = \frac{x-\mu}{\sigma}
$$

where:

* \(x\) is the original value.
* \(\mu\) is the training-set mean.
* \(\sigma\) is the training-set standard deviation.
* \(z\) is the standardized value.

Scaling can help models such as Linear Regression operate more effectively when features have very different numerical ranges.

Although tree-based models generally do not require scaling, a shared preprocessing structure simplifies model comparison.

## 5.2 Categorical Feature Processing

Categorical features include property type, ownership type, furnishing status, facing direction, age category, and locality group.

### Missing-Value Imputation

Missing categorical values are replaced according to the imputation strategy configured in the pipeline.

The workflow's preprocessing configuration should be treated as the source of truth for the exact imputation strategy used in each saved model.

### One-Hot Encoding

One-hot encoding converts categorical values into numerical indicator columns.

For example, a property-type feature might contain:

* Residential Apartment
* Independent/Builder Floor
* Independent House/Villa

One-hot encoding represents these categories through separate binary columns rather than assigning them arbitrary numerical rankings.

This prevents the model from interpreting category labels as inherently ordered numerical quantities.

The encoder is configured to handle previously unseen categories without failing during prediction.

## 5.3 Combining the Transformers

A `ColumnTransformer` applies the numerical and categorical transformations to their respective columns.

The overall structure is:

```text
Input Data
    |
    +-------------------------------+
    |                               |
    v                               v
Numerical Features             Categorical Features
    |                               |
Median Imputation              Missing-Value Imputation
    |                               |
Standard Scaling               One-Hot Encoding
    |                               |
    +---------------+---------------+
                    |
                    v
          Transformed Feature Matrix
                    |
                    v
             Regression Model
```

The preprocessing pipeline is fitted on training data and then reused to transform testing data.

This prevents the test set from influencing the learned imputation values, scaling parameters, or categorical encoding.

---

## Part C: Define the Regression Models

## 6. Purpose

Part C establishes multiple regression algorithms for comparison.

Using different model families is important because each algorithm captures relationships between property characteristics and prices differently.

Four regression models are investigated, together with a simple baseline model.

## 6.1 Dummy Regressor

The `DummyRegressor` provides a basic reference point for predictive performance.

A median-strategy dummy regressor predicts a constant value based on the median target in the training data.

It does not learn relationships between property features and prices.

Its purpose is to answer a fundamental question:

**Does the machine learning model perform better than a simple prediction based on the training-set median?**

A sophisticated model should provide meaningful improvement over this baseline.

## 6.2 Linear Regression

Linear Regression models the relationship between the input variables and the target using a linear function.

Its general form is:

$$
\hat{y} =
\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
$$

where:

* \(\hat{y}\) is the predicted target.
* \(x_1,\ldots,x_p\) are the input features.
* \(\beta_0\) is the intercept.
* \(\beta_1,\ldots,\beta_p\) are the learned coefficients.

### Advantages

* Simple and computationally efficient.
* Provides a useful baseline for regression.
* Relatively easy to interpret.
* Helps identify whether a linear relationship captures the data adequately.

### Limitations

* Cannot naturally represent complex nonlinear relationships without additional feature engineering.
* May be affected by multicollinearity.
* Can underfit complex real-estate pricing patterns.

## 6.3 Decision Tree Regressor

A Decision Tree Regressor divides the feature space into smaller regions using a sequence of decision rules.

For example, a tree might learn that properties with larger areas and particular locality characteristics tend to have different predicted prices.

### Advantages

* Captures nonlinear relationships.
* Can model interactions between features.
* Does not require numerical feature scaling.
* Can be visualized and interpreted when the tree is sufficiently small.

### Limitations

* Can overfit the training data.
* Small changes in the training data can produce a different tree.
* Deep trees may generalize poorly to unseen properties.

Hyperparameters such as tree depth and minimum samples per split can help control model complexity.

## 6.4 Random Forest Regressor

Random Forest is an ensemble learning algorithm that trains multiple decision trees and combines their predictions.

Each tree is trained using randomized sampling and feature selection. The final regression prediction is obtained by aggregating the predictions of the trees.

For \(T\) trees:

$$
\hat{y} =
\frac{1}{T}\sum_{t=1}^{T}\hat{y}_t
$$

where \(\hat{y}_t\) represents the prediction from the \(t\)-th tree.

### Advantages

* Captures nonlinear relationships.
* Models interactions between property features.
* Usually provides more stable predictions than a single decision tree.
* Handles a mixture of numerical and encoded categorical features.
* Can provide feature-importance estimates.

### Limitations

* More computationally expensive than simple linear models.
* Less directly interpretable than a small decision tree.
* Can still overfit, particularly when the trees are excessively complex.

Random Forest becomes an important candidate in this project because property prices can depend on nonlinear combinations of area, locality, property type, and accessibility.

## 6.5 Gradient Boosting Regressor

Gradient Boosting constructs an ensemble of decision trees sequentially.

Each successive tree attempts to improve the predictions produced by the existing ensemble by learning from its remaining errors.

### Advantages

* Captures nonlinear relationships.
* Can provide high predictive accuracy.
* Learns complex feature interactions.
* Often performs well on structured tabular datasets.

### Limitations

* Requires careful hyperparameter selection.
* Can overfit if the number or complexity of trees is excessive.
* Sequential training can be slower than training independent trees.

Important parameters include the learning rate, number of boosting stages, tree depth, and minimum samples required for a split.

## 6.6 Unified Pipeline Structure

Each regression algorithm is integrated with the preprocessing pipeline.

The resulting structure is:

```text
Raw Model Inputs
       |
       v
Preprocessing Pipeline
       |
       v
Regression Algorithm
       |
       v
Predicted Log Price
```

This ensures that every model receives appropriately processed inputs.

---

## Part D: Baseline Model Evaluation Using Cross-Validation

## 7. Purpose

Part D establishes a reliable baseline for comparing the regression algorithms.

Evaluating a model on a single train/test split can produce results that depend on the particular observations assigned to each set.

Cross-validation reduces dependence on one validation partition by evaluating the model across multiple folds.

## 7.1 Five-Fold Cross-Validation

The workflow uses five-fold cross-validation with shuffled folds and a fixed random seed.

The configuration is:

```python
from sklearn.model_selection import KFold

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

The training data is divided into five folds.

For each iteration:

1. Four folds are used to train the model.
2. The remaining fold is used for validation.
3. The validation score is recorded.
4. The process is repeated until every fold has served as the validation fold.

The five scores are summarized to estimate how consistently the model performs across different subsets of the training data.

The test set remains separate from this cross-validation process.

## 7.2 Evaluation Metrics

### Coefficient of Determination — R²

R² measures how well a model explains variation in the target.

$$
R^2 =
1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

where:

* \(y_i\) is the actual target.
* \(\hat{y}_i\) is the predicted target.
* \(\bar{y}\) is the mean actual target.
* \(n\) is the number of observations.

A higher R² generally indicates better predictive performance.

An R² of 1 represents perfect predictions. An R² of 0 corresponds to the performance of predicting the mean target on the same evaluation observations. R² can also be negative when predictions are worse than this reference.

### Root Mean Squared Error — RMSE

RMSE measures the square root of the average squared prediction error.

$$
RMSE =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Squaring errors gives greater weight to large prediction mistakes.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between predicted and actual values.

$$
MAE =
\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

MAE is easier to interpret because it represents the average absolute prediction error in the same units as the target.

### Important Note About the Target Scale

The baseline cross-validation experiments use `log_price`.

Consequently, RMSE and MAE calculated directly from the log-transformed target are expressed in log-price units, not in lakhs.

They must not be interpreted as direct currency errors.

## 7.3 Why Cross-Validation Is Necessary

Cross-validation helps to:

* Compare models under consistent validation conditions.
* Identify models that perform well across multiple folds.
* Detect unstable performance.
* Reduce reliance on a single validation subset.
* Support model selection before final testing.

The mean cross-validation score is considered alongside its variability rather than relying only on the best individual fold.

---

## Part E: Hyperparameter Tuning

## 8. Purpose

The initial models use baseline or default parameter settings. However, model performance depends on how each algorithm is configured.

Hyperparameter tuning searches for parameter combinations that improve predictive performance.

The objective is to identify a configuration that generalizes well rather than simply maximizing training accuracy.

## 8.1 What Are Hyperparameters?

Hyperparameters are settings chosen before a model is fitted.

Unlike learned model coefficients, they are not directly estimated from the training examples by the ordinary fitting procedure.

Examples include:

| Model             | Important Hyperparameters                                                              |
| ----------------- | -------------------------------------------------------------------------------------- |
| Linear Regression | Model regularization settings when using Ridge or other regularized variants           |
| Decision Tree     | Maximum depth, minimum samples per split, minimum samples per leaf                     |
| Random Forest     | Number of trees, maximum features, minimum samples per split, minimum samples per leaf |
| Gradient Boosting | Number of boosting stages, learning rate, maximum depth, minimum samples per split     |

Different hyperparameter values can affect model complexity, training time, and generalization performance.

## 8.2 Search Strategy

The tuning workflow uses cross-validation to evaluate candidate parameter configurations.

The process is:

1. Define a set or distribution of candidate hyperparameters.
2. Select an appropriate cross-validation strategy.
3. Fit each candidate configuration on the training folds.
4. Evaluate its predictions on the corresponding validation folds.
5. Aggregate the scores.
6. Identify the best configuration according to the selected scoring metric.
7. Retain the selected parameters for subsequent evaluation.

The best parameter configuration is recorded so that the experiment can be inspected and reproduced.

## 8.3 Hyperparameter Tuning Results

The recorded tuning summary contains the following RMSE values:

| Model             | Cross-Validation RMSE |
| ----------------- | --------------------: |
| Gradient Boosting |              0.172807 |
| Random Forest     |              0.180321 |
| Ridge Regression  |              0.235837 |
| Decision Tree     |              0.237505 |

These values are in log-price units.

**Gradient Boosting achieved the lowest RMSE in this recorded tuning summary**, followed by Random Forest.

The Ridge result belongs to the regularized linear-model experiment. It should not automatically be treated as the final score of the separately evaluated Linear Regression pipeline.

These results show why several algorithms should be investigated instead of assuming that one model family will perform best.

The tuned parameters are stored in a JSON artifact for later reference.

---

## Part F: Compare the Tuned Models

## 9. Purpose

Part F compares the tuned models under a consistent five-fold cross-validation procedure.

The aim is to determine whether hyperparameter tuning improves performance and whether the models generalize consistently.

The comparison considers:

* Cross-validation R².
* Cross-validation RMSE.
* Cross-validation MAE.
* Variation in performance across folds.
* Training performance.
* The gap between training and validation scores.

## 9.1 Cross-Validation Configuration

The same five-fold cross-validation configuration is used:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Using the same validation configuration makes the comparison more meaningful because the models are evaluated using consistent fold assignments.

## 9.2 Recorded Tuned-Model Results

The recorded results for the tuned models are:

| Model             | Mean CV R² | CV R² Standard Deviation | Training R² | CV RMSE (log-price) | CV MAE (log-price) | Overfit Gap |
| ----------------- | ---------: | -----------------------: | ----------: | ------------------: | -----------------: | ----------: |
| Random Forest     |     0.9106 |                   0.0086 |      0.9815 |              0.1828 |             0.1224 |      0.0709 |
| Gradient Boosting |     0.9072 |                   0.0103 |      0.9385 |              0.1862 |             0.1327 |      0.0314 |
| Linear Regression |     0.8531 |                   0.0103 |      0.8615 |              0.2345 |             0.1723 |      0.0084 |
| Decision Tree     |     0.8493 |                   0.0148 |      0.9511 |              0.2374 |             0.1568 |      0.1018 |

These are the recorded results from the tuned-model comparison. Minor differences may occur if the experiment is rerun with different preprocessing, parameter configurations, or library versions.

### Interpretation

**Random Forest**

Random Forest achieved the highest recorded mean cross-validation R² among these four tuned models. Its relatively small cross-validation R² standard deviation indicates consistent performance across the five folds.

However, its training R² is substantially higher than its validation R². This suggests some overfitting.

**Gradient Boosting**

Gradient Boosting achieved slightly lower mean cross-validation R² than Random Forest but a smaller training-to-validation R² gap.

This suggests that Gradient Boosting offers a useful balance between predictive performance and generalization.

**Linear Regression**

Linear Regression showed a relatively small training-to-validation gap, but its validation performance was lower than that of the two ensemble models.

This suggests that a purely linear model does not capture all the patterns present in the property dataset.

**Decision Tree**

The Decision Tree achieved high training R² but substantially lower cross-validation R².

This is consistent with a single tree learning patterns specific to the training data that do not generalize as effectively to unseen observations.

## 9.3 Understanding the Overfit Gap

The overfit gap is calculated as:

$$
\text{Overfit Gap}
=
R^2_{\text{training}}
-
R^2_{\text{cross-validation}}
$$

A larger gap can indicate that the model fits the training data much better than it performs on validation data.

A smaller gap is generally desirable, but it must be considered together with the overall validation score.

For example, a model with a very small gap can still perform poorly if it underfits the data.

## 9.4 Visualizing Model Comparison

The notebook generates a model-comparison figure:

`reports/figures/11_cv_model_comparison_v2.png`

This visualization helps compare the performance of the models and makes the differences easier to communicate.

The comparison outputs are also saved in tabular form for reporting and further analysis.

---

## Part G: Fit the Final Models and Transform Predictions

## 10. Purpose

Part G fits the selected tuned configurations on the complete training partition and prepares the resulting models for final evaluation and deployment.

The final models use the selected hyperparameters rather than continuing to search for parameters during prediction.

## 10.1 Final Model Fitting

The selected configurations of the four regression algorithms are fitted using the training data.

The training set contains 5,965 observations.

The test set contains 1,492 observations and remains reserved for final evaluation.

The models include:

* Linear Regression.
* Decision Tree Regressor.
* Random Forest Regressor.
* Gradient Boosting Regressor.

Each model includes its preprocessing pipeline so that new property inputs can be transformed in the same way as the training data.

## 10.2 Why the Target Is Log-Transformed

Real-estate prices can be highly skewed. A relatively small number of expensive properties can have prices substantially higher than those of most properties.

The target transformation is:

$$
y_{\text{log}} = \log(1+y)
$$

In the project, this transformation is implemented using `numpy.log1p()`.

The logarithmic transformation can reduce the influence of extreme target values and allow models to learn price relationships on a compressed scale.

It does not eliminate every effect of outliers, but it can make the target distribution easier to model.

## 10.3 Converting Predictions Back to Lakhs

Predictions made on the log-transformed target must be converted back to the original price scale.

The inverse transformation is:

$$
\hat{y} = e^{\hat{y}_{\text{log}}}-1
$$

In Python:

```python
predictions_lakhs = np.expm1(predictions_log)
```

This conversion is necessary before calculating evaluation metrics in lakhs.

For example, a predicted log-price of approximately 5.70 corresponds to a price of:

$$
e^{5.70}-1
$$

which is approximately 298.87 lakhs.

### Using TransformedTargetRegressor

The final-model workflow uses `TransformedTargetRegressor` to automate target transformation.

Conceptually, the structure is:

```text
Input Features
      |
      v
Preprocessing Pipeline
      |
      v
Regression Algorithm
      |
      v
Predicted Log Price
      |
      v
Inverse Target Transformation
      |
      v
Predicted Price in Lakhs
```

When a fitted `TransformedTargetRegressor` is used, its `predict()` method automatically applies the inverse transformation.

Therefore:

* A plain pipeline trained directly on `log_price` returns log-price predictions, which require `np.expm1()`.
* A correctly fitted `TransformedTargetRegressor` trained on actual prices with `func=np.log1p` returns predictions in the original price scale automatically.

Applying `np.expm1()` a second time to predictions that are already in lakhs would produce incorrect results.

## 10.4 Saving the Final Model Artifacts

The workflow saves the final model objects for reuse.

The project contains final-model artifacts following the naming convention:

```text
models/
├── linear_regression_final.joblib
├── decision_tree_final.joblib
├── random_forest_final.joblib
└── gradient_boosting_final.joblib
```

The workflow also maintains tuned model artifacts, such as:

```text
models/
├── linear_regression_tuned.joblib
├── decision_tree_tuned.joblib
├── random_forest_tuned.joblib
└── gradient_boosting_tuned.joblib
```

These files should not be assumed to be interchangeable.

A tuned artifact may be a plain preprocessing-and-regression pipeline that predicts log-price values, whereas a final artifact may wrap the pipeline in a `TransformedTargetRegressor`.

The correct prediction scale therefore depends on the exact saved object.

---

## Part H: Save Everything and Verify the Artifacts

## 11. Purpose

Part H organizes the outputs of the model-training workflow so that the experiments can be inspected, reproduced, and used in later project stages.

Saving the trained model alone is not sufficient for a well-documented machine learning workflow. It is also useful to retain the parameters, evaluation results, and metadata that explain how the model was produced.

## 11.1 Model Artifacts

The workflow saves the final trained models.

The saved models contain the fitted preprocessing operations and regression estimators required for prediction.

This avoids rebuilding the preprocessing logic separately when the model is loaded for inference.

## 11.2 Hyperparameter Configuration

The selected hyperparameters are saved in JSON format.

The notebook uses a best-parameter artifact, with names such as:

```text
models/best_model_params.json
```

or a versioned equivalent:

```text
models/best_model_params_v2.json
```

The exact filename depends on the version of the experiment.

This artifact helps document the settings used to obtain the tuned results.

## 11.3 Model Metadata

The workflow records metadata describing the trained models and their configuration.

The metadata artifact may include:

* Scikit-learn version.
* Numerical feature names.
* Categorical feature names.
* Target transformation.
* Number of training records.
* Model configuration.

The notebook uses a metadata file following the naming convention:

`models/model_meta.json`

Recording metadata helps identify differences between model versions and supports reproducibility.

## 11.4 Cross-Validation Results

The notebook saves cross-validation results as CSV files.

The recorded artifact names include:

```text
reports/cv_comparison.csv
reports/cv_baseline.csv
```

Versioned experiments may use filenames such as:

```text
reports/baseline_cv_results_v2.csv
```

These files preserve the results of the baseline and tuned-model comparisons.

They can be loaded into Pandas for further analysis, reporting, or visualization.

## 11.5 Model Reload Verification

After saving a model, it is important to verify that the saved artifact can be loaded successfully.

Example:

```python
import joblib

model = joblib.load(
    "models/random_forest_final.joblib"
)
```

The loaded model should accept the expected input columns and produce predictions without requiring the original notebook's in-memory variables.

A small prediction test is useful for detecting serialization or input-schema problems before deployment.

## 11.6 Why Saving Artifacts Matters

The saved artifacts support:

* Reproducibility.
* Model comparison.
* Experiment tracking.
* Debugging.
* Version management.
* Integration with the Streamlit application.
* Separation of model training from model inference.

The final application should load an appropriate saved model instead of retraining the model whenever a user requests a property-price prediction.

---

## 12. Final Test-Set Evaluation

After the model-selection and training workflow, the candidate models are evaluated using the held-out test set.

The test set contains 1,492 properties that are separate from the training set.

The selected model is evaluated using the original price scale so that the prediction errors can be interpreted in lakhs.

## 12.1 Evaluation Metrics

The final evaluation reports three principal metrics.

| Metric | Purpose                                                           | Preferred Direction |
| ------ | ----------------------------------------------------------------- | ------------------- |
| R²     | Measures how well the model explains variation in property prices | Higher              |
| MAE    | Measures average absolute prediction error                        | Lower               |
| RMSE   | Penalizes large prediction errors more heavily                    | Lower               |

MAE and RMSE are reported in lakhs because the actual target is `price_lakhs`.

## 12.2 Final Model Comparison

The recorded test-set results for the tuned pipelines are:

| Model             |         R² | MAE (lakhs) | RMSE (lakhs) |
| ----------------- | ---------: | ----------: | -----------: |
| Random Forest     | **0.9010** | **43.3703** | **100.8134** |
| Gradient Boosting |     0.8949 |     49.7949 |     103.8771 |
| Decision Tree     |     0.8727 |     56.0719 |     114.3299 |
| Linear Regression |     0.7882 |     67.8057 |     147.4914 |

These values were calculated after converting the plain pipelines' log-price predictions back to the original price scale using `np.expm1()`.

The table describes the recorded test evaluation of the tuned pipeline artifacts. A model retrained using a different wrapper, preprocessing configuration, or hyperparameter set may produce slightly different results.

### Interpretation

**Random Forest achieved the strongest recorded test-set performance.**

It produced:

* R² of 0.9010.
* MAE of approximately ₹43.37 lakhs.
* RMSE of approximately ₹100.81 lakhs.

An R² of 0.9010 indicates that the model explains approximately 90.1% of the observed variation in the test-set target, under the evaluation setup used.

The MAE indicates that the mean absolute difference between predicted and actual property prices is approximately ₹43.37 lakhs.

The RMSE is higher than the MAE because it gives greater weight to large errors. This indicates that some predictions have considerably larger errors than the typical absolute error.

The model performs better than the other three candidates on all three metrics in this recorded comparison.

However, an R² of 0.9010 does not mean that every property prediction is 90.1% accurate. R² measures explained variation rather than the percentage accuracy of individual predictions.

---

## 13. Additional Experiment: Missing Floor Number Indicator

After comparing the four main models, an additional experiment was performed to investigate whether missing floor-number information could be represented more effectively.

## 13.1 Motivation

The `floor_num` feature contains missing values for some properties.

An examination of the dataset showed:

| Property Type             | Missing Floor Numbers | Total Properties |
| ------------------------- | --------------------: | ---------------: |
| Independent House/Villa   |                   433 |              433 |
| Independent/Builder Floor |                   194 |            2,274 |
| Residential Apartment     |                   127 |            4,750 |
| **Total**                 |               **754** |        **7,457** |

All 433 Independent House/Villa records had missing floor numbers in this dataset.

This suggested that missingness might contain information about property type or the way floor information is recorded.

## 13.2 Feature Creation

A binary feature named `floor_num_missing` was introduced.

Its values are:

* `1`: The property's floor number is missing.
* `0`: The property's floor number is available.

The feature was created before the preprocessing pipeline imputed missing floor values.

The training-set and test-set distributions were:

| Dataset  | Indicator = 0 | Indicator = 1 |
| -------- | ------------: | ------------: |
| Training |         5,363 |           602 |
| Testing  |         1,340 |           152 |

The original `floor_num` feature remained in the model, and the indicator was added as an extra numerical feature.

## 13.3 Experimental Setup

A new Random Forest was trained using the additional indicator.

The original and indicator models used:

* The same training and testing partitions.
* The same target transformation.
* The same Random Forest hyperparameters.
* The same evaluation metrics.

The purpose was to isolate the effect of adding the missing-value indicator.

## 13.4 Results

| Model                          |         R² | MAE (lakhs) | RMSE (lakhs) |
| ------------------------------ | ---------: | ----------: | -----------: |
| Random Forest — Original       | **0.9010** | **43.3703** | **100.8134** |
| Random Forest — With Indicator |     0.9009 |     43.4309 |     100.8636 |

The indicator model performed marginally worse across all three metrics.

The differences are small, but the additional feature did not demonstrate a predictive improvement on this test set.

One possible explanation is that the existing `property_type` feature already captures some of the information associated with missing floor numbers. This is a plausible explanation rather than a proven cause.

### Conclusion

The original feature set was retained.

The `floor_num_missing` feature was not included in the selected model because it did not improve the recorded test-set results.

This experiment demonstrates the importance of validating feature-engineering decisions empirically instead of assuming that adding more features always improves performance.

---

## 14. Model Training Pipeline Summary

The complete workflow can be summarized as follows:

```text
Processed Property Dataset
          |
          v
Load Feature Configuration
          |
          v
Verify Data and Train/Test Split
          |
          v
Separate Input Features and Target
          |
          v
Build Preprocessing Pipeline
          |
          +--------------------------+
          |                          |
          v                          v
Numerical Features              Categorical Features
          |                          |
          v                          v
Median Imputation              Missing-Value Imputation
          |                          |
          v                          v
Standard Scaling               One-Hot Encoding
          |                          |
          +------------+-------------+
                       |
                       v
               Model Training
                       |
          +------------+-------------+
          |            |             |
          v            v             v
       Linear       Decision       Random
     Regression       Tree         Forest
          |
          +---------------- Gradient Boosting
                       |
                       v
             Cross-Validation
                       |
                       v
             Hyperparameter Tuning
                       |
                       v
             Tuned Model Comparison
                       |
                       v
              Final Model Selection
                       |
                       v
             Final Model Evaluation
                       |
                       v
             Save Model and Metadata
                       |
                       v
               Deployment Preparation
```

---

## 15. Project Directory Structure

The model-training workflow uses the following project organization:

```text
real-estate-price-prediction/
│
├── data/
│   └── processed/
│       └── model_data.csv
│
├── notebooks/
│   └── 06_model_training.ipynb
│
├── src/
│   └── train.py
│
├── models/
│   ├── feature_config.json
│   ├── best_model_params.json
│   ├── model_meta.json
│   ├── linear_regression_final.joblib
│   ├── decision_tree_final.joblib
│   ├── random_forest_final.joblib
│   └── gradient_boosting_final.joblib
│
└── reports/
    ├── cv_comparison.csv
    ├── cv_baseline.csv
    └── figures/
        └── 11_cv_model_comparison_v2.png
```

This is a representative structure of the module's files. Versioned parameter files, alternate model artifacts, and additional reports may also exist in the project.

The notebook contains the experimental workflow, while the Python script is intended to make the selected training procedure reusable outside the notebook.

---

## 16. Reproducibility and Best Practices

Several practices were followed throughout the model-training process.

### Fixed Random Seeds

A fixed random seed, such as `random_state=42`, is used in relevant operations to improve reproducibility.

### Consistent Data Splitting

The existing `split` column preserves the designated training and testing partitions.

### Pipeline-Based Preprocessing

Imputation, scaling, encoding, and regression are organized into pipelines to keep transformations consistent between training and prediction.

### Cross-Validation

Five-fold cross-validation provides a more robust comparison than relying solely on a single validation score.

### Hyperparameter Tracking

Selected hyperparameters are recorded in JSON artifacts for reference and reproducibility.

### Held-Out Test Evaluation

The test set is reserved for evaluating the selected model rather than being used to fit the preprocessing pipeline or search for hyperparameters.

### Correct Target Transformation

Log-price predictions are converted back to the original price scale before reporting errors in lakhs.

### Model Serialization

Trained models are saved so they can be loaded without rerunning the complete training notebook.

### Feature Validation

The missing-floor-indicator experiment demonstrates that additional features should be retained only when they provide a useful improvement.

---

## 17. Running the Model Training Workflow

The notebook can be opened using Jupyter Notebook or JupyterLab.

From the project root, the training script can be executed using:

```bash
python src/train.py
```

The script is intended to:

1. Load the processed dataset.
2. Load the feature configuration.
3. Build the preprocessing pipeline.
4. Build the selected regression model.
5. Fit the model on the training data.
6. Evaluate predictions on the held-out test set.
7. Save the fitted model.
8. Verify that the saved model can be loaded.

The script and notebook should use compatible preprocessing definitions and target transformations.

If the notebook and script are maintained separately, changes to feature lists, preprocessing settings, or model hyperparameters should be reflected in the reusable training implementation.

---

## 18. Limitations

Although the selected model demonstrates strong performance on the available test set, several limitations remain.

### Data Coverage

The model is trained on the properties represented in the processed Gurgaon dataset. Its performance may differ for properties in other cities, localities, or market conditions.

### Data Quality

Missing values, incorrect property details, inconsistent listing information, and duplicated geographic coordinates can affect the learned relationships.

### Price Variation

Property prices can depend on factors that are not fully captured by the available dataset, including negotiation, building quality, legal status, amenities, and short-term market conditions.

### Large Prediction Errors

The RMSE is considerably higher than the MAE. This indicates that some predictions have substantially larger errors than the average absolute error.

Residual analysis is therefore important for identifying properties where the model performs poorly.

### Temporal Changes

Real-estate prices change over time. A model trained on historical listings may require periodic retraining to remain relevant to current market conditions.

### Interpretability

Random Forest can capture complex relationships but does not provide the same direct interpretability as a simple linear model. Feature importance and error analysis can help explain its behavior, although they do not establish causality.

### Generalization

The reported scores apply to the particular dataset and test partition used. Performance should be re-evaluated if the dataset, preprocessing pipeline, or feature configuration changes.

---

## 19. Conclusion

The Model Training module establishes a complete experimental workflow for predicting residential property prices in Gurgaon.

The module verifies the processed dataset, builds preprocessing pipelines, trains multiple regression algorithms, applies five-fold cross-validation, tunes hyperparameters, compares model performance, and saves trained artifacts for reuse.

The recorded test-set comparison identified Random Forest as the strongest of the four evaluated regression pipelines, with:

* **R²:** 0.9010
* **MAE:** ₹43.3703 lakhs
* **RMSE:** ₹100.8134 lakhs

Gradient Boosting also performed strongly, while Decision Tree and Linear Regression produced larger test-set errors in the recorded comparison.

An additional experiment investigated whether a binary indicator for missing floor numbers could improve Random Forest performance. The indicator did not improve the recorded metrics, so the original feature set was retained.

The resulting workflow provides a foundation for deploying the selected model in a Streamlit application. By saving the preprocessing pipeline together with the trained estimator, the project can apply the same transformations to new property inputs without repeating the training process.

Overall, the module demonstrates a systematic approach to machine learning model development, evaluation, selection, and preparation for deployment.

---

## 20. Key Files and Artifacts

| File                                            | Purpose                                                     |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `data/processed/model_data.csv`                 | Processed dataset used for training                         |
| `models/feature_config.json`                    | Numerical and categorical feature definitions               |
| `notebooks/06_model_training.ipynb`             | Experimental model-training workflow                        |
| `models/best_model_params.json`                 | Recorded tuned hyperparameters, subject to the version used |
| `models/model_meta.json`                        | Model configuration and metadata                            |
| `models/*_tuned.joblib`                         | Tuned model pipeline artifacts                              |
| `models/*_final.joblib`                         | Final model artifacts                                       |
| `reports/cv_baseline.csv`                       | Baseline cross-validation results                           |
| `reports/cv_comparison.csv`                     | Model comparison results                                    |
| `reports/figures/11_cv_model_comparison_v2.png` | Cross-validation comparison visualization                   |
| `src/train.py`                                  | Reusable training script                                    |

**Note:** The exact contents and filenames should be checked against the files currently present in the repository, because the project contains multiple experiment versions and both tuned and final model artifacts.


# Module 7:  Model Evaluation 

## 1. Introduction

Model evaluation is an important stage of the Gurgaon Real Estate Price Prediction project. After preprocessing the dataset and performing feature engineering, multiple machine learning algorithms were trained and evaluated to identify a model capable of predicting residential property prices accurately.

The primary objective of this phase was to:

* Evaluate the performance of different machine learning algorithms.
* Compare the prediction accuracy of the trained models.
* Identify overfitting and underfitting.
* Analyse actual versus predicted property prices.
* Study prediction errors using residual analysis.
* Select the final machine learning model.
* Evaluate the reliability of the model's prediction interval.

The evaluation was performed using both cross-validation and an independent test dataset.

---

## 2. Dataset Used for Evaluation

The evaluation was performed using the processed Gurgaon residential property dataset.

### Dataset Information

| Description         |         Value |
| ------------------- | ------------: |
| Total properties    |         7,457 |
| Training records    |         5,965 |
| Testing records     |         1,492 |
| Total records       |         7,457 |
| Target variable     |   `log_price` |
| Original target     | `price_lakhs` |
| Prediction location |       Gurgaon |

The dataset contains property-related information such as:

* Number of bedrooms.
* Number of bathrooms.
* Number of balconies.
* Property floor.
* Total floors in the building.
* Property area.
* Property type.
* Ownership type.
* Furnishing status.
* Property facing.
* Property age.
* Locality.
* Latitude and longitude.
* Nearby landmark information.

### Dataset Splitting

The processed dataset was divided into two subsets:

**Training Dataset – 80%**

* 5,965 property records.
* Used for training machine learning algorithms.
* Used for fitting preprocessing operations and model parameters.

**Testing Dataset – 20%**

* 1,492 property records.
* Used to evaluate the final trained models.
* Kept separate from model training to provide an independent performance estimate.

The split was performed before model evaluation to prevent information from the test dataset from influencing model training.

The dataset also contains a `split` column to identify training and testing records.

---

## 3. Target Variable Transformation

The original property price was represented in lakhs.

The target variable was transformed using the natural logarithm:

```python
df["log_price"] = np.log1p(df["price_lakhs"])
```

The transformation can be represented as:

$$
y = \log(1 + P)
$$

Where:

* \(P\) = Original property price in lakhs.
* \(y\) = Log-transformed property price.

### Why was logarithmic transformation used?

Real-estate prices have a wide range of values. Some properties have relatively low prices, while premium properties have considerably higher prices.

Logarithmic transformation helps:

1. Reduce the effect of highly expensive properties.
2. Reduce target-value skewness.
3. Make the target distribution easier for regression models to learn.
4. Improve model stability when dealing with a wide range of property prices.

The model predicts the transformed target variable.

The predicted value is converted back into the original price scale using:

```python
predicted_price = np.expm1(predicted_log)
```

This produces the predicted property price in lakhs.

---

## 4. Machine Learning Algorithms Evaluated

Four machine learning algorithms were trained and compared.

### 4.1 Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

It attempts to establish a linear relationship between input features and the target variable.

The general equation is:

$$
Y = b_0 + b_1X_1 + b_2X_2 + \cdots + b_nX_n
$$

Where:

* \(Y\) = Predicted property price.
* \(b_0\) = Intercept.
* \(b_1, b_2, \ldots, b_n\) = Model coefficients.
* \(X_1, X_2, \ldots, X_n\) = Input features.

**Purpose of evaluation:**

Linear Regression was used as a baseline model to understand how well a relatively simple regression algorithm could predict property prices.

### 4.2 Decision Tree Regressor

Decision Tree Regression predicts numerical values by dividing the dataset into smaller groups based on feature conditions.

It creates a tree-like structure containing decision rules.

For example:

* If property area is greater than a particular value, follow one branch.
* If the number of bedrooms is greater than a particular value, follow another branch.

**Purpose of evaluation:**

The Decision Tree model was evaluated to understand how well a nonlinear tree-based algorithm could capture relationships between property features and prices.

### 4.3 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Each tree generates a prediction, and the final prediction is calculated by combining the predictions of individual trees.

**Purpose of evaluation:**

Random Forest was evaluated because it can capture complex nonlinear relationships between property characteristics and real-estate prices.

It also reduces the instability associated with using a single decision tree.

### 4.4 Gradient Boosting Regressor

Gradient Boosting is an ensemble learning algorithm that builds decision trees sequentially.

Each new tree attempts to correct errors made by the previously constructed trees.

**Purpose of evaluation:**

Gradient Boosting was included to compare another ensemble-based approach with Random Forest.

It helps determine whether sequential error correction provides better predictive performance for the dataset.

---

## 5. Cross-Validation

Cross-validation was used to estimate how consistently each model performs across different subsets of the training dataset.

Instead of relying on a single training and validation split, cross-validation divides the training data into multiple folds.

In each iteration:

1. A portion of the data is used for training.
2. The remaining portion is used for validation.
3. The process is repeated with different validation folds.
4. The performance values are averaged.

The evaluation used cross-validation to compare the models based on:

* Cross-validation R² score.
* Cross-validation RMSE on the logarithmic target scale.

### Cross-Validation Results

| Model             |  CV R² | CV RMSE (Log) |
| ----------------- | -----: | ------------: |
| Random Forest     | 0.9106 |        0.1828 |
| Gradient Boosting | 0.9072 |        0.1862 |
| Linear Regression | 0.8531 |        0.2345 |
| Decision Tree     | 0.8493 |        0.2374 |

### Interpretation

**Random Forest**

* Achieved the highest cross-validation R² score.
* Recorded the lowest cross-validation RMSE.
* Demonstrated strong predictive performance across validation folds.

**Gradient Boosting**

* Achieved performance close to Random Forest.
* Recorded a slightly lower cross-validation R² score.
* Demonstrated competitive performance.

**Linear Regression**

* Provided a baseline for comparison.
* Had lower predictive performance than the ensemble models.

**Decision Tree**

* Recorded the lowest cross-validation R² score among the evaluated models.
* Had the highest cross-validation RMSE.

Based on these results, Random Forest demonstrated the strongest average cross-validation performance.

---

## 6. Overfitting Analysis

Overfitting occurs when a machine learning model learns the training data too closely and performs less effectively on unseen data.

To investigate overfitting, the difference between training performance and cross-validation performance was examined.

The R² difference was calculated as:

$$
\text{Overfitting Gap}
=
R^2_{\text{Training}}
-
R^2_{\text{Cross Validation}}
$$

A larger positive gap may indicate that a model performs substantially better on training data than on unseen validation data.

### Overfitting Comparison

| Model             | Overfitting Gap |
| ----------------- | --------------: |
| Random Forest     |          0.0709 |
| Gradient Boosting |          0.0314 |

### Interpretation

Random Forest achieved strong cross-validation performance but showed a larger training-to-validation performance gap than Gradient Boosting.

Gradient Boosting demonstrated a smaller overfitting gap.

This comparison highlights an important distinction:

* Random Forest achieved better average validation performance.
* Gradient Boosting showed a smaller difference between training and validation performance.

The final selection also considered performance on the independent test dataset.

---

## 7. Actual vs Predicted Price Analysis

Actual versus predicted visualisation was performed to understand how closely the model's predictions matched the actual property prices.

The comparison was performed for all four evaluated models.

### Visualisation

A scatter plot was created for each model.

* X-axis: Actual property prices.
* Y-axis: Predicted property prices.

A reference diagonal line represents perfect prediction.

$$
\text{Actual Price} = \text{Predicted Price}
$$

### Interpretation

* Points closer to the diagonal line represent more accurate predictions.
* Points above the diagonal represent overprediction.
* Points below the diagonal represent underprediction.
* Points farther from the diagonal indicate larger prediction errors.

This visualisation helps identify whether the models consistently predict property prices close to their actual values.

The analysis is especially useful for understanding model behaviour across properties with different price ranges.

---

## 8. Residual Analysis

Residual analysis was performed to understand the errors produced by the prediction models.

A residual represents the difference between the actual value and the predicted value.

$$
\text{Residual} =
\text{Actual Price} -
\text{Predicted Price}
$$

### Residual Interpretation

| Residual | Meaning                               |
| -------- | ------------------------------------- |
| Positive | Model underestimated the actual price |
| Negative | Model overestimated the actual price  |
| Zero     | Perfect prediction                    |

### Residual Visualisation

Residual plots were generated for the evaluated models.

The analysis helps identify:

1. Whether prediction errors are distributed around zero.
2. Whether errors increase with property price.
3. Whether the model consistently overestimates or underestimates certain properties.
4. Whether unusually large prediction errors exist.

### Expected Characteristics of a Good Residual Distribution

A reasonably performing regression model should ideally have:

* Residuals distributed around zero.
* No strong systematic pattern.
* Relatively stable error variation across prediction values.
* A limited number of unusually large residuals.

Residual analysis provides additional information that cannot be obtained from R² alone.

---

## 9. Model Performance Evaluation Metrics

Multiple evaluation metrics were used to compare the performance of the regression algorithms.

### 9.1 R² Score – Coefficient of Determination

R² measures how much of the variation in the target variable is explained by the model.

$$
R^2 =
1 -
\frac{\sum (y_i-\hat{y}_i)^2}
{\sum (y_i-\bar{y})^2}
$$

Where:

* \(y_i\) = Actual value.
* \(\hat{y}_i\) = Predicted value.
* \(\bar{y}\) = Mean actual value.

**Interpretation:**

* R² = 1: Perfect predictions.
* R² = 0: No improvement over predicting the mean.
* R² < 0: Worse than predicting the mean.

A higher R² generally indicates better predictive performance.

### 9.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

**Interpretation:**

A lower MAE indicates that the model's predictions are closer to the actual values on average.

MAE is particularly useful because it is expressed in the same unit as the target variable.

### 9.3 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction errors.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

**Interpretation:**

* A lower RMSE indicates better predictive performance.
* Larger errors receive greater penalties because the errors are squared.

RMSE is useful for identifying models that produce unusually large prediction errors.

---

## 10. Independent Test Dataset Evaluation

After comparing the models using cross-validation, the final model was evaluated using the independent test dataset.

The test dataset contained 1,492 property records that were not used for fitting the models.

The selected Random Forest model achieved the following performance.

### Final Random Forest Test Results

| Evaluation Metric      |        Result |
| ---------------------- | ------------: |
| R² Score (Log Scale)   |        0.9175 |
| MAE (Log Scale)        |        0.1169 |
| RMSE (Log Scale)       |        0.1772 |
| R² Score (Price Scale) |        0.9010 |
| MAE (Price Scale)      |  ₹43.37 Lakhs |
| RMSE (Price Scale)     | ₹100.81 Lakhs |

### Interpretation

**R² Score – 0.9010**

The model explains approximately 90.10% of the variation in property prices in the independent test dataset, measured on the original price scale.

**MAE – ₹43.37 Lakhs**

On average, the absolute difference between predicted and actual property prices was approximately ₹43.37 lakhs.

**RMSE – ₹100.81 Lakhs**

The RMSE was approximately ₹100.81 lakhs, indicating that some predictions had substantially larger errors than the average absolute error.

The difference between MAE and RMSE highlights the influence of larger prediction errors.

---

## 11. Final Model Selection

Based on cross-validation and independent test evaluation, Random Forest Regressor was selected as the final model.

### Reasons for Selection

1. It achieved the highest cross-validation R² score among the evaluated models.
2. It recorded the lowest cross-validation RMSE.
3. It achieved strong performance on the independent test dataset.
4. It captured nonlinear relationships between property features and prices.
5. Its test performance was suitable for the project's property price estimation objective.

Although Gradient Boosting demonstrated a smaller overfitting gap, Random Forest achieved better cross-validation performance.

Therefore, Random Forest was selected based on its overall predictive performance rather than relying on a single evaluation metric.

---

## 12. Prediction Interval Evaluation

In addition to generating a single predicted price, an empirical prediction interval was evaluated to communicate uncertainty around individual predictions.

The interval was constructed using prediction errors from the calibration/test evaluation process.

The interval adjustments were calculated on the logarithmic target scale.

### Prediction Interval Configuration

| Parameter            |     Value |
| -------------------- | --------: |
| Target Coverage      |       90% |
| Lower Log Adjustment | -0.284054 |
| Upper Log Adjustment |  0.284612 |
| Actual Test Coverage |    89.88% |

### Interval Calculation

Let the predicted logarithmic price be:

$$
\hat{y} = \log(1+\hat{P})
$$

The predicted price is converted into the original scale using:

$$
\hat{P} = \exp(\hat{y})-1
$$

The lower estimate is:

$$
P_{\text{lower}} =
\exp(\hat{y}-0.284054)-1
$$

The upper estimate is:

$$
P_{\text{upper}} =
\exp(\hat{y}+0.284612)-1
$$

All three values are expressed in lakhs.

### Interpretation

The target coverage was 90%, and the observed test coverage was approximately 89.88%.

This indicates that approximately 89.88% of the evaluated actual test prices fell within the calculated prediction intervals.

The result is close to the intended 90% coverage.

**Important:** This is an empirical prediction interval based on the project's evaluation procedure. It is not a guarantee that the actual market price will fall within the displayed range.

---

## 13. Model Saving and Deployment Preparation

After selecting the final model, the model pipeline was saved for future use.

The project maintains separate model files for evaluation and deployment.

| File                           | Purpose                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `best_model.pkl`               | Selected model pipeline used for independent test evaluation                 |
| `deployment_model.pkl`         | Final deployment pipeline trained using all 7,457 available property records |
| `prediction_interval.json`     | Stores the empirical prediction interval adjustments                         |
| `eval_meta.json`               | Stores final evaluation metrics and model information                        |
| `feature_config.json`          | Stores the model's feature configuration                                     |
| `model_config.json`            | Stores model configuration information                                       |
| `selected_model_metadata.json` | Stores selected model metadata                                               |
| `best_model_params.json`       | Stores selected model hyperparameters                                        |

The deployment model was trained using the full available dataset after model selection and evaluation.

The independent test performance reported above belongs to the separately evaluated model. It should not be interpreted as a fresh independent test score for the deployment model trained on all records.

The deployment pipeline includes preprocessing operations, so the same preprocessing steps do not need to be manually repeated when loading the pipeline for inference.

---

## 14. Limitations of the Evaluation

Although the selected Random Forest model achieved strong evaluation results, certain limitations must be considered.

### 14.1 Geographic Limitation

The dataset contains residential property listings from Gurgaon.

Therefore, the model is intended for Gurgaon and should not be assumed to provide accurate predictions for other cities.

### 14.2 Dataset Limitation

The model learns from the available property listings.

Changes in market conditions, property demand, location development and listing quality may affect prediction accuracy.

### 14.3 High-Value Property Errors

The model showed substantial underestimation for some properties priced above ₹10 crore.

This indicates that prediction accuracy may vary across different price ranges.

### 14.4 Prediction Uncertainty

The prediction interval is empirical and does not guarantee the actual selling price.

### 14.5 Feature Limitations

The model uses the available property and location features. Factors not represented in the dataset may also influence property prices.

Examples include:

* Exact construction quality.
* Interior condition.
* Legal status.
* Floor-plan design.
* Negotiation and transaction conditions.
* Recent market developments.

---

## 15. Conclusion

The model evaluation phase compared four regression algorithms for predicting residential property prices in Gurgaon.

Linear Regression, Decision Tree, Random Forest and Gradient Boosting were evaluated using cross-validation and an independent test dataset.

Random Forest demonstrated the strongest overall cross-validation performance and achieved an R² score of approximately 0.9010 on the independent test dataset when evaluated on the original price scale.

Its MAE was approximately ₹43.37 lakhs, while its RMSE was approximately ₹100.81 lakhs.

Residual analysis and actual-versus-predicted visualisations were also used to understand the behaviour of the models and their prediction errors.

An empirical prediction interval was evaluated, achieving approximately 89.88% test coverage against a target coverage of 90%.

Based on the evaluation results, Random Forest was selected as the final model for the Gurgaon Real Estate Price Prediction project.

The selected model pipeline was subsequently saved for use in the separate price prediction module.

---

## 16. Scope of This README

This document covers the **Model Evaluation phase only**.

The following implementation is documented separately as Step 8 – Price Prediction Module:

* Loading the deployment model.
* Accepting property details from users.
* Preparing model input.
* Generating property price predictions.
* Converting logarithmic predictions into lakhs.
* Displaying lower and upper prediction estimates.
* Calculating estimated price per square foot.
* Downloading prediction results.

These operations belong to the application and inference stage, not the model evaluation stage.

````

### Where to save it

Inside your project, I recommend:

```text
real-estate-price-prediction/
│
├── README.md
│
├── reports/
│   ├── MODEL_EVALUATION.md
│   └── ...
│
├── models/
├── notebooks/
├── src/
├── data/
└── app/
````

Save the document as:

`reports/MODEL_EVALUATION.md`

Keep the root `README.md` for the overall project introduction and navigation.

**One important note:** The evaluation metrics and dataset figures above reflect the results recorded during our project work. Before publishing, make sure the notebook's actual plots and evaluation outputs are saved alongside the report.

