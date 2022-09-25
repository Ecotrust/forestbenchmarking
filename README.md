# STAC implementation for Forest Benchmarking and Modelling

About 

STAC implementation for building and benchmarking machine learning models for predicting forest attributes. 

The datasets envisioned for this work include plot and stand-level measurements collected from state and federal agencies in Oregon and Washington, as well as forest disturbance records (e.g., of harvest, pests, fire, etc.). The “features” for predictive modeling will include satellite, lidar, hydrologic, infrastructure, terrain, and climatic data layers.

Structure of the STAC catalog

```text
Catalog (Root): forest_benchmarking_stands
|
|_ Collection: dataset_id
|  - collection.json
|     |
|     |_ Item: QQ_ID_DATASETCODE
|        - qq_id_datasetcode.json
|        |
|        |_ Asset: single or multiband COG
|        |_ Asset: Thumbnail
|        |_ Asset: Metadata
|
|_ Collection: labels_collection_id
   - collection.json 
      |
      |_ Item (Label extension): QQ_ID_LABELCODE
         - qq_id_labelcode.json
         - links: Links to all dataset items within the same QQ.
         |
         |_ Asset: FeatureCollection with attributes for each label.

```
STAC version 1.0.0

## Explore a small example 

[Open sample Forest Benchmarking catalog with the STAC Browser](https://radiantearth.github.io/stac-browser/#/external/raw.githubusercontent.com/Ecotrust/forestbenchmarking/main/data/forest_benchmarking_catalog/catalog.json)


## Installation

Clone conda environment

```bash
conda env create -f environment.yml
```
