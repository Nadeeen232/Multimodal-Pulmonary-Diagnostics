# Capture the container-list region; omit the workstation path in the header.
param([switch]$FullWindow)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class DockerWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr window);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr window, int command);
}
'@
$dockerWindow = Get-Process -Name 'Docker Desktop' | Where-Object {
    $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like '*sic_graduation_project*'
} | Select-Object -First 1
if (-not $dockerWindow) { throw 'Open the project container view in Docker Desktop first.' }
$wasMinimized = [DockerWindowCapture]::IsIconic($dockerWindow.MainWindowHandle)
if ($wasMinimized) {
    [DockerWindowCapture]::ShowWindow($dockerWindow.MainWindowHandle, 9) | Out-Null
    Start-Sleep -Seconds 2
}
$bounds = New-Object DockerWindowCapture+Rect
if (-not [DockerWindowCapture]::GetWindowRect($dockerWindow.MainWindowHandle, [ref]$bounds)) {
    throw 'Unable to read Docker Desktop window dimensions.'
}
$bitmap = New-Object System.Drawing.Bitmap(($bounds.Right - $bounds.Left), ($bounds.Bottom - $bounds.Top))
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$dc = $graphics.GetHdc()
try {
    if (-not [DockerWindowCapture]::PrintWindow($dockerWindow.MainWindowHandle, $dc, 2)) {
        throw 'Docker Desktop window capture failed.'
    }
} finally { $graphics.ReleaseHdc($dc) }
try {
    $outputPath = Join-Path $PSScriptRoot '../reports/docker_desktop_screenshot.png'
    if ($FullWindow) {
        # Full-window captures stay private because the header contains a host path.
        $outputPath = Join-Path $PSScriptRoot '../.tmp/docker_desktop_full.png'
        $bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    } else {
        $region = New-Object System.Drawing.Rectangle(270,155,($bitmap.Width-300),230)
        if ($region.Right -gt $bitmap.Width -or $region.Bottom -gt $bitmap.Height) {
            throw 'Window too small for the public container-list capture.'
        }
        $capture = $bitmap.Clone($region,$bitmap.PixelFormat)
        try { $capture.Save($outputPath,[System.Drawing.Imaging.ImageFormat]::Png) }
        finally { $capture.Dispose() }
    }
    Write-Output $outputPath
} finally {
    $graphics.Dispose(); $bitmap.Dispose()
    if ($wasMinimized) {
        [DockerWindowCapture]::ShowWindow($dockerWindow.MainWindowHandle, 6) | Out-Null
    }
}
