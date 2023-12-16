# Unsupervised Multi-modal Feature Alignment for Time Series Representation Learning 


## Datasets
We use the 30 datasets from UEA archive and four anomaly detection datasets in this study.

* [UEA Archive](http://www.timeseriesclassification.com/)

* [Soil Moisture Active Passive satellite (SMAP)
and Mars Science Laboratory rover (SML) datasets](https://dl.acm.org/doi/10.1145/3219819.3219845)

* [Server Machine
Dataset, SMD](https://dl.acm.org/doi/10.1145/3292500.3330672)

* [Application Server Dataset, ASD](https://dl.acm.org/doi/10.1145/3447548.3467075)

The UEA datasets should be in the "Multivariate_ts/" folder with the structure `Multivariate_ts/[dataset_name]/[dataset_name]_TRAIN.ts` and `Multivariate_ts/[dataset_name]/[dataset_name]_TEST.ts`.

For SMAP and MSL datasets, create a folder named `SMAP&MSL` under 'AD_data/', and put the `.npy` data files into `AD_data/SMAP&MSL/`.

Similarly, to test SMD and ASD datasets, create a folder named `SMD&ASD` under 'AD_data/' then put the data files of `.pkl` into the folder `AD_data/SMD&ASD/`.

### Usage
```
CUDA_VISIBLE_DEVICES=0 python main.py --experiment_description exp1 --run_description run1 --seed 123 --training_mode specreg --data_path Eplipsy --view_list fft-db1-rp-gadf
```