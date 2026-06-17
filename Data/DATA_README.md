# Data

This folder contains the datasets used by the experimental scripts.

## Files

`ARA_SMS_Dataset_Final.csv`  
Primary Arabic SMS dataset used for SMS text-based spam classification. The dataset contains real-world Arabic SMS messages collected from mobile devices in Egypt and labeled under a binary ham/spam taxonomy.

Required columns:

| Column | Description |
|---|---|
| `Message` | SMS message text |
| `Label` | Message label: `Ham` or `Spam` |

`Smishing_Dataset_Final.csv`  
Smishing-oriented URL dataset used for URL-based phishing/smishing analysis. The dataset contains legitimate and malicious URLs used to evaluate the heuristic engine and learned URL classifiers.

Required columns:

| Column | Description |
|---|---|
| `url` | URL string |
| `label` | URL label: `0` for legitimate/benign and `1` for malicious/phishing |

## Privacy and anonymization

The Arabic SMS dataset was anonymized before release. Personally identifiable information appearing in message content was removed, masked, or replaced, including names, phone numbers, email addresses, physical addresses, bank identifiers, transaction-related references, and other sensitive details.

No subscriber metadata, device identifiers, sender or receiver numbers, SIM information, geographic location data, or other directly identifying metadata are included.

SMS-derived URLs were handled using the same privacy-preserving procedure. Where URLs contained personal or transaction-specific information, such information was removed or masked before inclusion.

## Usage note

The scripts in the `scripts/` folder expect these datasets to be placed under the `data/` directory using the filenames shown above.

Expected structure:

```text
project_root/
  data/
    ARA_SMS_Dataset_Final.csv
    Smishing_Dataset_Final.csv
  scripts/
  splits/
  outputs/