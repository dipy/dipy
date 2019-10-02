# Powershell Install script

# Useful function from https://stackoverflow.com/questions/50093582/powershell-not-recognizing-conda-as-cmdlet-function-or-operable-program
function Invoke-CmdScript {
  param(
    [String] $scriptName
  )
  $cmdLine = """$scriptName"" $args & set"
  & $Env:SystemRoot\system32\cmd.exe /c $cmdLine |
  Select-String '^([^=]*)=(.*)$' | ForEach-Object {
    $varName = $_.Matches[0].Groups[1].Value
    $varValue = $_.Matches[0].Groups[2].Value
    Set-Item Env:$varName $varValue
  }
}

if($env:INSTALL_TYPE -match "conda")
{
    Write-Output "Activate testenv"
    Invoke-CmdScript $env:CONDA\Scripts\activate.bat testenv
}

$env:PIPI = "pip install --timeout=60 --find-links=$env:EXTRA_WHEELS"
# Print and check this environment variable
Write-Output "Pip command: $env:PIPI"


# Print and Install FURY
Write-Output "======================== Install DIPY ========================"
Invoke-Expression "$env:PIPI --user -e ."

# Run tests
Write-Output "======================== Run DIPY tests ========================"
Invoke-Expression "pytest -svv dipy"
