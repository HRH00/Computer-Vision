# redirect stderr into stdout
$p = &{python -V} 2>&1
# check if an ErrorRecord was returned
if($p -is [System.Management.Automation.ErrorRecord])
{
    Write-Host "Install python 3.11" 
    # grab the version string from the error message
    $p.Exception.Message
	Write-Host "Press any key to continue..."
	$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

else 
{
    # otherwise return as is
    Write-Host "Python Version is:" 

    $p
    Write-Host "Downloading pip" 

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    Write-Host "running get pip" 
    python get-pip.py
    Write-Host "installing pip" 

    python -m pip install --upgrade pip
    Write-Host "installing scikit-learn"
    pip install scikit-learn
    Write-Host "installing matplotlib"
    pip install matplotlib
    Write-Host "installing opencv-contrib-python"
    pip install opencv-contrib-python
    Write-Host "installing numpy"
    pip install numpy
    write-host "Script Complete"


}