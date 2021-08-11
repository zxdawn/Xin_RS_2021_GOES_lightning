# Xin_RS_2021_GOES_lightning

## /notebooks

Python jupyter notebooks of plotting my paper's figures.

- Fig. 1 and Fig. 3

  region_distribution.ipynb

- Fig. 2b

  workflow.ipynb

- Fig. 4

  low_cth.ipynb

- Fig. 5, Fig. 6, Fig. 9, and Fig. 10

  relationships.ipynb

- Fig. 7 and Fig. 8

  case_studies.ipynb

## /scripts

The general program flow is outlined below by [GitMind](https://gitmind.com/). The first step is pre-processing the original GOES-16 ABI, GLM and ERA5 data. Then, the results are passed to track convections. Finally, the lightning data are paired with the tracks and saved into csv files.

<img src="https://github.com/zxdawn/Xin_RS_2021_GOES_lightning/raw/main/figures/workflow.png" width="700">

## /data

The merged data used in relationships.ipynb are available at [Zenodo](https://doi.org/10.5281/zenodo.5179871).

The GOES-16 ABI L1 C13, L2 ACHAC, and GLM L2 data can be accessed from the [Amazon Web Services](https://registry.opendata.aws/noaa-goes/), [Google Cloud](https://console.cloud.google.com/marketplace/partners/noaa-public), and [NOAA CLASS service](https://www.class.noaa.gov/).