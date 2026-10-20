$ErrorActionPreference = 'Stop'
$runtime = 'C:\msys64\opt\dreamerv3-python'
if (!(Test-Path "$runtime\python.exe")) {
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $archive = Join-Path $runtime 'python.zip'
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
    Remove-Item -LiteralPath $archive
    @('python312.zip', '.', 'Lib\site-packages', 'import site') | Set-Content "$runtime\python312._pth" -Encoding ascii
    Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile "$runtime\get-pip.py"
    & "$runtime\python.exe" "$runtime\get-pip.py" --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
}
& "$runtime\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'PyTorch setup failed' }
& "$runtime\python.exe" -m pip install numpy==2.2.6 gymnasium==1.2.1 --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed' }
& "$runtime\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
