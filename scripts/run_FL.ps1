param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Model,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$Dataset
)

$ErrorActionPreference = "Stop"

if (-not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = "0"
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

switch ($Model) {
    { $_ -in "LinearPH", "NNPH", "NNnPH" } { $Module = "LinearPH_NNPH_NNnPH.experiments" }
    "AFT_Weibull" { $Module = "AFT_Weibull.experiments" }
    "DeepHit" { $Module = "DeepHit.experiments" }
    "DeepPseudo" { $Module = "DeepPseudo.experiments" }
    "Fed_CoxPH" { $Module = "Fed_CoxPH.experiments" }
    { $_ -in "FedPDNN", "FedPLSTM", "FedPAttn" } { $Module = "FedPseudo_Flare.experiments" }
    default { throw "Unsupported model: $Model" }
}

python -m $Module `
    --partition FL `
    --model $Model `
    --dataset $Dataset `
    --device cuda:0