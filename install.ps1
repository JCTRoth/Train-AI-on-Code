# Update Help for PowerShell
Update-Help -Force

# Install Python3 and Python3 pip if not already installed
if (-not (Get-Command python --Version 3)) {
    Write-Host "Installing Python3..."
    iex "& { $(irm https://aka.ms/install-python3.ps1) } -UseMsiInstaller"
}

if (-not (Get-Command pip --Version 3)) {
    Write-Host "Installing pip for Python3..."
    python3 -m ensurepip --upgrade
}

# Update pip
Write-Host "Updating pip..."
python3 -m pip install --upgrade pip

# Install required Python libraries
Write-Host "Installing required Python libraries..."
python3 -m pip install torch transformers torchvision

