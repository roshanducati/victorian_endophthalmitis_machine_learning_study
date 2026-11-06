# Reproduce every reported result, in order. Requires the registry export at
# data/raw/ (see DATA_SCHEMA.md). Run from the repository root.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Invoke-Step([string]$Label, [string[]]$Arguments) {
    Write-Host $Label
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Label" }
}

Invoke-Step "[1/9] Preprocess the registry export (strict outcome definition)" @("01_preprocess.py", "--strict")
Invoke-Step "[2/9] Develop the models with microbiology" @("02_train.py", "--strict")
Invoke-Step "[3/9] Develop the pre-culture models" @("02_train.py", "--strict", "--nomicro")
Invoke-Step "[4/9] Comparators: logistic regression and presenting acuity alone" @("03_benchmarks.py")
Invoke-Step "[5/9] Sensitivity analysis: predictors with high missingness" @("04_sensitivity_missing_predictors.py")
Invoke-Step "[6/9] Held-out test-set statistics" @("05_heldout_statistics.py")
Invoke-Step "[7/9] Patient-clustered bootstrap (10 000 resamples)" @("06_patient_cluster_bootstrap.py", "--n-boot", "10000")
Invoke-Step "[8/9] Tables, Figure 3 and Figure S1" @("07_tables_and_figures.py")
Invoke-Step "[9/9] Figures 1 and 2" @("08_figures_test_set.py")
Invoke-Step "Checking the reproduced results against the manuscript" @("verify_reproduction.py")
