Hey Brain Wave Research Members this repository acts as a storage point for our dementia project. Here you will find inmportant information and functions:

**Final Results**: This holds the cortical measurements across all the patients in the OASIS dataset

**Functions**: This holds prewritten code that can be used to anaylze the dataset to give you an idea of what kind of figures you can produce

**PT**: Holds all the information related to the patients in this study 

**What is this dataset**: The OASIS (Open Access Series of Imaging Studies) is a publicly available neuroimaging dataset 
designed to study brain aging and Alzheimer’s disease using structural and functional MRI.

| **CDR**  | Dementia severity             | 0 = none, 0.5 = very mild, 1+ = dementia |
 
| **MMSE** | Cognitive score (0–30)        | Higher = better cognition                |

| **nWBV** | Normalized whole brain volume | Strong structural biomarker              |

| **eTIV** | Intracranial volume           | Used for normalization                   |

| **ASF**  | Atlas scaling factor          | Related to head size                     |


**Notes for Contributors**
Use baseline data (Visit = 1) unless explicitly analyzing longitudinal changes
Avoid data leakage (do not mix visits from the same subject across train/test splits)
Clearly define outcome variables (e.g., CDR-based classification) before analysis
All figures should be publication-quality (labeled axes, units, legends)
