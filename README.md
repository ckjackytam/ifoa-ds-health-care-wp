# ifoa-ds-health-care-wp

## 1. Project Overview 

This repo contains the code base for the IFoA Techniques in Data Science in Health and Care Working Party. 

The working party created a framework to help actuaries determine which data science techniques are appropriate for health and care projects. 

The repo includes case studies on mortality modelling demonstrating the framework in practice. Code is split between R (traditional actuarial modelling) and Python (machine learning).

## 2. The Framework 

The link to the framework is on the [IFoA website](https://actuaries.org.uk/media/ppqhbjfp/a-framework-for-ds-technicques-to-hc-actuarial-20251003.pdf). This will also be published as a sessional paper in the British Actuarial Journal later in 2026.  

The framework is designed for projects relating to tabular data and comprises four categories: 

- Study design and technology requirements 
- Pre-model 
- Model 
- Post-model 

## 3. Case Studies 

These mortality modelling case studies demonstrate the framework in practice.

### CMI Mortality Modelling 

CMI has kindly given the working party permission to use their mortality dataset (CMI Working Paper 162, 2022) for our case study. Note: this data is only available to CMI contributing members and is not open-sourced. 

**GAM** (`Python/build-better-base-gam`)

This study uses GAM (via PyGAM) to build a mortality model from scratch while using GBM as a tool to detect important interaction effects. We published an article in The Actuary explaining our approach in detail: [A new pathway](https://www.theactuary.com/2025/07/02/new-pathway-framework-incorporating-data-science-health-and-care). 


**Neural Network** (`Python/simple-nn`)

Can we use NN to build an explainable mortality model with a bespoke loss function (zero-inflated Poisson to account for excessive zeros in the target variable)? We used PyTorch to find out. 

### ILEC Mortality Modelling  

The Society of Actuaries has a comprehensive open-source [ILEC mortality dataset](https://www.soa.org/resources/research-reports/2024/ilec-mort-2012-19/) on US mortality experience. This is an ideal dataset to demonstrate how to use explainable ML under our framework.  The Python implementation includes two validation approaches: random train/test split and calendar year split. The Python code for this study is in `Python/big-pitch/`, while the R code is in `R/big pitch/`. We published a paper on this study and presented at IFoA Health & Care Big Pitch 2026: [Hybrid modelling](https://actuaries.org.uk/media/xikgrpeg/bridging-transparency-and-predictive-power-integrating-explainable-ml-into-actuarial-modelling-fiona-fan-michiel-luteijn-jacky-tam.pdf
).

## 4. Repository Structure 

**Python folders:**
- `src/` - Shared Python utilities and modules
- `build-better-base-gam/` - CMI GAM with GBM interaction detection
- `simple-nn/` - CMI neural network with custom loss function
- `big-pitch/` - ILEC explainable ML study

**R folders:**
- `big pitch/` - ILEC study (R implementation)


## 5. Getting Started 

**Prerequisites:**
- Python 3.10+ and R 4.x+
- Access to CMI data (for CMI case studies)

**Setup:**

1. Clone the repository
2. Navigate to the case study folder of interest
3. Install dependencies: `pip install -r requirements.txt` (where available)
4. Run notebooks in numerical order (01, 02, 03...)

**Note:** CMI case studies require CMI member data access. ILEC case studies use publicly available SOA data.

## 6. Data Sources

- [SOA ILEC Mortality Dataset](https://www.soa.org/resources/research-reports/2024/ilec-mort-2012-19/) (open-source)
- CMI Working Paper 162 (2022) - Term assurance experience 2016-2020 (CMI members only)

## 7. Citation

If you use this code or framework in your work, please cite the relevant paper(s):

**Framework:**
Luteijn, J.M. et al. (2026). *A Framework for Data Science Techniques in Health and Care*. Institute and Faculty of Actuaries. https://actuaries.org.uk/media/ppqhbjfp/a-framework-for-ds-technicques-to-hc-actuarial-20251003.pdf

**GAM Study:**
Tam, J., & Luteijn, J.M. (2025). "A new pathway: A framework for incorporating data science into health and care." *The Actuary*. https://www.theactuary.com/2025/07/02/new-pathway-framework-incorporating-data-science-health-and-care

**Hybrid Modelling:**
Fan, F., Luteijn, M., & Tam, J. (2026). "Bridging transparency and predictive power: Integrating explainable ML into actuarial modelling." Presented at IFoA Health & Care Big Pitch 2026. https://actuaries.org.uk/media/xikgrpeg/bridging-transparency-and-predictive-power-integrating-explainable-ml-into-actuarial-modelling-fiona-fan-michiel-luteijn-jacky-tam.pdf


