# Get modified files (tracked files that have been modified)
$modified = git diff --name-only HEAD

# Get staged files (tracked files that have been staged)
$staged = git diff --cached --name-only

# Get deleted files (tracked files that have been deleted)
$deletedFiles = git ls-files --deleted | ForEach-Object { $_.Trim() }

$allFiles = @{
    Added = @()
    Deleted = @()
    Changed = @()
}

# Process modified files
foreach ($file in $modified) {
    $allFiles["Changed"] += $file
}

# Process staged files
foreach ($file in $staged) {
    if ($allFiles["Changed"] -contains $file) {
        # No need to add again if already in Changed
    } else {
        $allFiles["Added"] += $file
    }
}

# Process deleted files
foreach ($file in $deletedFiles) {
    $allFiles["Deleted"] += $file
}

# Output the results in the desired order and format
foreach ($file in $allFiles["Added"] | Sort-Object -Unique) {
    Write-Output(" *added $file")
}

foreach ($file in $allFiles["Deleted"] | Sort-Object -Unique) {
    Write-Output(" *deleted $file")
}

foreach ($file in $allFiles["Changed"] | Sort-Object -Unique) {
    Write-Output(" *changed $file")
}
