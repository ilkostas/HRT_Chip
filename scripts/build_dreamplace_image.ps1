# Build the default mixed-size Docker image (analytical proxy contract).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Tag = if ($env:HRT_DREAMPLACE_IMAGE_TAG) { $env:HRT_DREAMPLACE_IMAGE_TAG } else { "hrt-chip-dreamplace:local" }
docker build -f "$Root/docker/Dockerfile.dreamplace" -t $Tag "$Root/docker"
Write-Host "Built $Tag"
