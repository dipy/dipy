# Powershell Install script

if($env:INSTALL_TYPE -match "conda")
{
  # Get Anaconda path
  Write-Output "Conda path: $env:CONDA\Scripts"
  #gci env:*

  Invoke-Expression "conda config --set always_yes yes --set changeps1 no"
  Invoke-Expression "conda update -yq conda"
  Invoke-Expression "conda install conda-build anaconda-client"
  Invoke-Expression "conda config --add channels conda-forge"
  Invoke-Expression "conda create -n testenv --yes python=$env:PYTHON_VERSION pip"
  Invoke-Expression "conda install -yq --name testenv $env:DEPENDS $env:EXTRA_DEPENDS pytest"
}
else
{
  $env:PIPI = "pip install --timeout=60 --find-links=$env:EXTRA_WHEELS"
  # Print and check this environment variable
  Write-Output "Pip command: $env:PIPI"
  Invoke-Expression "python -m pip install -U pip"
  Invoke-Expression "pip --version"
  Invoke-Expression "$env:PIPI $env:DEPENDS $env:EXTRA_DEPENDS pytest"
}
