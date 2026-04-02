param(
    [Parameter(Mandatory = $true)]
    [string] $WorkDir,
    [string] $Image = $(if ($env:HRT_DREAMPLACE_IMAGE) { $env:HRT_DREAMPLACE_IMAGE } else { "hrt-chip-dreamplace:local" })
)
$ErrorActionPreference = "Stop"
$abs = (Resolve-Path $WorkDir).Path
$extra = @()
if ($env:HRT_TESTCASE_ROOT) {
    $tr = (Resolve-Path $env:HRT_TESTCASE_ROOT).Path
    $extra += "-v", "${tr}:/testcase:ro"
}
docker run --rm -v "${abs}:/work" @extra $Image
