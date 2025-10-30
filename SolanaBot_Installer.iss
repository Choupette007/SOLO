; ============================
; SOLOTradingBot_Installer.iss
; Binary installer for SOLOTradingBot beta testers
; ============================

[Setup]
AppId={{7C2B3E1C-6E3E-4B5E-BF1E-5A2C7F0C9E12}
AppName=SOLOTradingBot
AppVersion=1.0.0
AppPublisher=SOLOTradingBot
DefaultDirName={autopf}\SOLOTradingBot
DefaultGroupName=SOLOTradingBot
OutputDir=C:\Users\Admin\Desktop\SOLOTradingBot\installer
OutputBaseFilename=SOLOTradingBotSetup
Compression=lzma
SolidCompression=yes
SetupIconFile=C:\Users\Admin\Desktop\SOLOTradingBot\bot_icon.ico
UninstallDisplayIcon={app}\SOLOTradingBot.exe
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0
DisableDirPage=no
DisableProgramGroupPage=yes
SignedUninstaller=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
; ADDED: Install the default.env alongside app binary (optional reference)
Source: "default.env"; DestDir: "{app}"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{src}\default.env'))

; ADDED: Install .env directly into appdata so users don't need to copy anything
Source: "default.env"; DestDir: "{localappdata}\SOLOTradingBot"; DestName: ".env"; Flags: ignoreversion

Source: "C:\Users\Admin\Desktop\SOLOTradingBot\dist\SOLOTradingBot.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Admin\Desktop\SOLOTradingBot\config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Admin\Desktop\SOLOTradingBot\solana_bot_gui_corrected_working.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Admin\Desktop\SOLOTradingBot\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Admin\Desktop\SOLOTradingBot\utils\rugcheck_auth.py"; DestDir: "{app}\utils"; Flags: ignoreversion
Source: "C:\Users\Admin\Desktop\SOLOTradingBot\venv\Lib\site-packages\*"; DestDir: "{app}\Lib\site-packages"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\SOLOTradingBot"; Filename: "{app}\SOLOTradingBot.exe"
Name: "{userdesktop}\SOLOTradingBot"; Filename: "{app}\SOLOTradingBot.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SOLOTradingBot.exe"; Description: "Run SOLOTradingBot now"; Flags: nowait postinstall skipifsilent

[Dirs]
; ADDED: Ensure appdata directory exists on install
Name: "{localappdata}\SOLOTradingBot"
Name: "{localappdata}\SOLOTradingBot"; Permissions: users-modify

[Code]
function InitializeSetup(): Boolean;
var
  Is64: Boolean;
  Major, Minor, Build: Integer;
begin
  Is64 := IsWin64;
  GetWindowsVersionEx(Major, Minor, Build);
  if (not Is64) or (Major < 10) then
  begin
    MsgBox('This beta requires 64-bit Windows 10 or later. Installation cannot continue.',
           mbCriticalError, MB_OK);
    Result := False;
    exit;
  end;

  MsgBox('Welcome to the SOLOTradingBot Beta Installer.'#13#10#13#10 +
         'This installer includes the compiled application and required files.'#13#10 +
         'After installation, edit %LOCALAPPDATA%\SOLOTradingBot\.env with your keys.',
         mbInformation, MB_OK);
  Result := True;
end;