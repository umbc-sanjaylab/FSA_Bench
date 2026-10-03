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

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

python experiments.py `
    --setting CL `
    --model $Model `
    --dataset $Dataset `
    --device cuda:0