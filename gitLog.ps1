$modified   = git diff --name-only HEAD
$staged     = git diff --cached --name-only
$untracked  = git ls-files --others --exclude-standard

$allFiles = $modified + $staged + $untracked | Sort-Object -Unique

foreach ($file in $allFiles) {
    Write-Output "* $file"
}

