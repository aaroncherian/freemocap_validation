$DATA = "D:\freemocap_validation_dataset\data"
$OUT  = "D:\freemocap_validation_dataset\archives"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null

Get-ChildItem $DATA -Directory -Filter "sub-*" |
    Sort-Object Name |
    ForEach-Object {
        $zip = Join-Path $OUT "freemocap_validation_dataset_v1.0_$($_.Name).zip"

        if (Test-Path $zip) {
            Remove-Item $zip
        }

        Write-Host "Creating $zip ..."
        tar.exe -a -c -f $zip -C $DATA $_.Name
    }