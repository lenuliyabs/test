#define MyAppName "HistoAnalyzer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "HistoAnalyzer"
#define MyAppExeName "HistoAnalyzer.exe"

[Setup]
AppId={{A8F24A48-6608-4F7D-9376-1E5D6FAF2A61}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=release
OutputBaseFilename=Setup_HistoAnalyzer
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "downloadmodels"; Description: "Скачать модели ИИ (один раз, затем оффлайн)"; GroupDescription: "Дополнительно:"; Flags: unchecked

[Files]
Source: "dist\HistoAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCR; Subkey: ".histo"; ValueType: string; ValueData: "HistoAnalyzer.Project"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "HistoAnalyzer.Project"; ValueType: string; ValueData: "HistoAnalyzer Project"; Flags: uninsdeletekey
Root: HKCR; Subkey: "HistoAnalyzer.Project\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"; Flags: uninsdeletekey
Root: HKCR; Subkey: "HistoAnalyzer.Project\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "--download-models"; Flags: postinstall skipifsilent nowait; Tasks: downloadmodels
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
