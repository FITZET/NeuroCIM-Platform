#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef BuildFlavor
  #define BuildFlavor "CPU"
#endif
#ifndef SourceRoot
  #error SourceRoot must point to the prepared NeuroCIM-Windows-x64 directory
#endif
#ifndef OutputRoot
  #define OutputRoot "."
#endif

[Setup]
AppId={{C9BCE9D5-0A60-4A91-BA65-14E7B59458B9}
AppName=NeuroCIM
AppVersion={#AppVersion}
AppVerName=NeuroCIM {#AppVersion} ({#BuildFlavor})
AppPublisher=FITZET
AppPublisherURL=https://github.com/FITZET
DefaultDirName={autopf}\NeuroCIM
DefaultGroupName=NeuroCIM
UninstallDisplayIcon={app}\NeuroCIM\NeuroCIM.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
OutputDir={#OutputRoot}
OutputBaseFilename=NeuroCIM-Setup-{#BuildFlavor}-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=1700000000
SlicesPerDisk=1
WizardStyle=modern
CloseApplications=yes
RestartApplications=yes
SetupMutex=NeuroCIMSetupMutex
ChangesEnvironment=no
DisableProgramGroupPage=yes
VersionInfoVersion={#AppVersion}
VersionInfoCompany=FITZET
VersionInfoDescription=NeuroCIM DRAM CIM Simulation Platform
VersionInfoProductName=NeuroCIM
VersionInfoProductVersion={#AppVersion}

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: unchecked

[Files]
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\NeuroCIM"; Filename: "{app}\NeuroCIM\NeuroCIM.exe"
Name: "{autodesktop}\NeuroCIM"; Filename: "{app}\NeuroCIM\NeuroCIM.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NeuroCIM\NeuroCIM.exe"; Description: "启动 NeuroCIM"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKLM; Subkey: "Software\FITZET\NeuroCIM"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\FITZET\NeuroCIM"; ValueType: string; ValueName: "Flavor"; ValueData: "{#BuildFlavor}"

