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
    $p
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    python -m pip install --upgrade pip
    pip install -U scikit-learn
    pip install 
    pip install opencv-contrib-python
    pip install numpy

}