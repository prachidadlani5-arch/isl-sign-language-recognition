import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "dataset/landmarks.csv"

EXPECTED_WORDS = 2667
EXPECTED_FEATURES = 126

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 60)
print("LANDMARK CSV VERIFICATION")
print("=" * 60)

print("\nLoading landmarks.csv...")

df = pd.read_csv(INPUT_FILE)

print("✓ CSV loaded successfully")

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("Total rows/samples :", len(df))
print("Total columns      :", len(df.columns))

# ============================================================
# CHECK LABEL COLUMN
# ============================================================

print("\n" + "=" * 60)
print("LABEL CHECK")
print("=" * 60)

if "label" not in df.columns:

    print("✗ ERROR: 'label' column is missing")

else:

    print("✓ Label column exists")

    unique_words = df["label"].nunique()

    print("Unique words       :", unique_words)
    print("Expected words     :", EXPECTED_WORDS)

    if unique_words == EXPECTED_WORDS:
        print("✓ All 2667 words are present")

    else:
        print(
            f"⚠ Word count mismatch: "
            f"{unique_words}/{EXPECTED_WORDS}"
        )

# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE CHECK")
print("=" * 60)

feature_columns = [
    column
    for column in df.columns
    if column.startswith("f")
]

print("Feature columns    :", len(feature_columns))
print("Expected features  :", EXPECTED_FEATURES)

if len(feature_columns) == EXPECTED_FEATURES:

    print("✓ Exactly 126 landmark features found")

else:

    print("✗ Feature count is incorrect")

# ============================================================
# CHECK FEATURE NAMES
# ============================================================

expected_names = [
    f"f{i}"
    for i in range(EXPECTED_FEATURES)
]

missing_features = [
    column
    for column in expected_names
    if column not in df.columns
]

if missing_features:

    print("\n✗ Missing feature columns:")
    print(missing_features)

else:

    print("✓ f0 to f125 are present")

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

missing_total = df.isnull().sum().sum()

print("Total missing values:", missing_total)

if missing_total == 0:

    print("✓ No missing values")

else:

    print("⚠ Missing values found")

# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:

    print("✓ No duplicate rows")

else:

    print("⚠ Duplicate rows found")

# ============================================================
# CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPE CHECK")
print("=" * 60)

non_numeric_features = []

for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(df[column]):

        non_numeric_features.append(column)

if len(non_numeric_features) == 0:

    print("✓ All landmark features are numeric")

else:

    print("✗ Non-numeric features found:")
    print(non_numeric_features)

# ============================================================
# CHECK LANDMARK RANGES
# ============================================================

print("\n" + "=" * 60)
print("LANDMARK RANGE CHECK")
print("=" * 60)

feature_data = df[feature_columns]

min_value = feature_data.min().min()
max_value = feature_data.max().max()

print("Minimum value:", min_value)
print("Maximum value:", max_value)

print(
    "\nNote: MediaPipe coordinates can contain "
    "negative z-values, so negative values are normal."
)

# ============================================================
# SAMPLES PER WORD
# ============================================================

print("\n" + "=" * 60)
print("SAMPLES PER WORD")
print("=" * 60)

samples_per_word = df["label"].value_counts()

print(
    "Minimum samples for one word:",
    samples_per_word.min()
)

print(
    "Maximum samples for one word:",
    samples_per_word.max()
)

print(
    "Average samples per word:",
    round(samples_per_word.mean(), 2)
)

# ============================================================
# SHOW LOWEST SAMPLE COUNTS
# ============================================================

print("\nWords with the fewest samples:")

print(
    samples_per_word
    .sort_values()
    .head(10)
)

# ============================================================
# SHOW HIGHEST SAMPLE COUNTS
# ============================================================

print("\nWords with the most samples:")

print(
    samples_per_word
    .sort_values(ascending=False)
    .head(10)
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION SUMMARY")
print("=" * 60)

checks = {
    "2667 words": unique_words == EXPECTED_WORDS,
    "126 features": len(feature_columns) == EXPECTED_FEATURES,
    "No missing values": missing_total == 0,
    "Numeric features": len(non_numeric_features) == 0,
}

for name, result in checks.items():

    if result:
        print(f"✓ {name}")

    else:
        print(f"✗ {name}")

# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)