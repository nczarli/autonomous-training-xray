{
**********************************************************************************
 * Autonomous Training in X-Ray Imaging Systems
 * 
 * Training a deep learning model based on noisy labels from a rule based algorithm.
 * 
 * Copyright 2023 Nikodem Czarlinski
 * 
 * Licensed under the Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)
 * (the "License"); you may not use this file except in compliance with 
 * the License. You may obtain a copy of the License at
 * 
 *     https://creativecommons.org/licenses/by-nc/3.0/
 * 
**********************************************************************************
}
library pythonDll;

{ Important note about DLL memory management: ShareMem must be the
  first unit in your library's USES clause AND your project's (select
  Project-View Source) USES clause if your DLL exports any procedures or
  functions that pass strings as parameters or function results. This
  applies to all strings passed to and from your DLL--even those that
  are nested in records and classes. ShareMem is the interface unit to
  the BORLNDMM.DLL shared memory manager, which must be deployed along
  with your DLL. To avoid using BORLNDMM.DLL, pass string information
  using PChar or ShortString parameters. }

uses
  Sharemem,
  System.SysUtils,
  System.Classes,
  Windows,
  ShellApi,
  PythonEngine;

/// <summary>
/// Represents a custom thread class that inherits from TThread.
/// </summary>
/// <remarks>
/// This class provides a base for creating custom thread classes
/// with specific execution logic.
/// </remarks>
type
  TMyThread = class(TThread)
  protected
    procedure Execute; override;
  end;

var
  TrainingThread: TMyThread;
  gEngine : TPythonEngine;
const
  useCpu = False;
  PythonTrainingFile = 'C:\Users\Nikodem\Documents\GitHub\autonomous-training-xray\ssr\ssr.py';
  PythonImportFile = 'C:\Users\Nikodem\Documents\GitHub\autonomous-training-xray\RBA-delphi\import_file.py';
  PythonSegmentFile = 'C:\Users\Nikodem\Documents\GitHub\autonomous-training-xray\RBA-delphi\segment_cherries.py';
  



// A resource file contains various types of data, such as icons, bitmaps, 
// strings, and other resources, that can be used by your Delphi application. 
// By including a resource file in your project, you can access and use these 
// resources within your code.
{$R *.res}



/// <summary> Represents a custom thread class that inherits from TThread.
/// </summary>
/// <remarks>
/// This executes the machine learning training training script.
/// </remarks>
procedure TMyThread.Execute;
var
  CommandLine: String;
  PythonFile: String;
  ThingDone: Boolean;
  Arguments: String;
  StartupInfo: TStartupInfo;
  ProcessInfo: TProcessInformation;
begin
  ThingDone := False;
  OutputDebugStringA('Inside training thread.');

  // do some work...
  while not Terminated do
  begin
    if not ThingDone then
      begin
        // Target Python File
        PythonFile :=  PythonTrainingFile;

        // Arguments passed to Python file
        if useCpu then
          Arguments := '--dataset "training_inside_delphi" --use_cpu_only True'
        else
          Arguments := '--dataset "training_inside_delphi"';


        OutputDebugStringA('Creating training process.');

        // Create Process
        CommandLine := 'python.exe "' + PythonFile + '" ' + Arguments;
        FillChar(StartupInfo, SizeOf(TStartupInfo), 0);
        StartupInfo.cb := SizeOf(TStartupInfo);
        CreateProcess(nil, PChar(CommandLine), nil, nil, False, 0, nil, nil, StartupInfo, ProcessInfo);
        OutputDebugStringA('Created training process.');

        // Set flag to indicate process started
        ThingDone := True;
        OutputDebugStringA('Training process done.');
      end;

     // Sleep for a short while to avoid consuming too much CPU time
    Sleep(1);
  end;
  // clean up...

  OutputDebugStringA('Training thread finished, terminating process.');

  // terminate process
  TerminateProcess(ProcessInfo.hProcess, 0);
  CloseHandle(ProcessInfo.hProcess);
  CloseHandle(ProcessInfo.hThread);

end;

/// <summary> Starts the machine learning training thread. </summary>
procedure StartTrainingThread stdcall;
begin
  TrainingThread := TMyThread.Create(True); // Create the thread object
  OutputDebugStringA('Creating training thread.');
  TrainingThread.FreeOnTerminate := True; // Free the thread object when it terminates
  TrainingThread.Start; // Start the thread
  OutputDebugStringA('Started training thread.')
end;

/// <summary> Terminates the machine learning training thread. </summary>
procedure TerminateTrainingThread stdcall;
begin
  TrainingThread.Terminate;
end;

/// <summary> Initialises the python engine. </summary>
procedure InitPythonEngine stdcall;
begin
  gEngine := TPythonEngine.Create(nil);
  gEngine.AutoFinalize := False;
  gEngine.UseLastKnownVersion := False;
  gEngine.RegVersion := '3.9';  //<-- Use the same version as the python 3.x your main program uses
  gEngine.APIVersion := 1013;
  gEngine.DllName := 'python39.dll';
  gEngine.LoadDll;
  gEngine.ExecFile(PythonImportFile);
  OutputDebugStringA('Finished Initialisation');
end;

/// <summary> Tests print using the python engine. </summary>
procedure PrintUsingEngine stdcall;
begin
  try
    gEngine := GetPythonEngine;
    gEngine.ExecString('import sys');
    gEngine.ExecString('print(sys.modules.keys())');
    gEngine.ExecString('import torch');
    gEngine.ExecString('a=1233');
    gEngine.ExecString('print("Print Using Engine 1234")');
  except
  end;
end;

/// <summary> Checks what if modules need to be imported every time engine is run.
/// </summary>
procedure PrintA stdcall;
begin
  try
    gEngine := GetPythonEngine;
    gEngine.ExecString('import sys');
    gEngine.ExecString('print(sys.modules.keys())');
    gEngine.ExecString('print(a)');
  except
  end;
end;


/// <summary> Performs rule based algorithm and machine learning inference on image.
/// </summary>
procedure ChangeImagePython stdcall;
var
  PathToFile : String;
begin
  try
    PathToFile := PythonSegmentFile;
    gEngine := GetPythonEngine;
    gEngine.ExecFile(PathToFile);
  except
  end;
end;


/// <summary> Sends data to the python engine for debugging. </summary>
procedure PythonInputOutput1SendData(Sender: TObject;
  const Data: AnsiString);
begin
{$IFDEF MSWINDOWS}
  OutputDebugStringA( PAnsiChar(Data) );
{$ENDIF}
{$IFDEF LINUX}
  WriteLn( ErrOutput, Data );
{$ENDIF}

end;

exports
  StartTrainingThread,
  TerminateTrainingThread,
  InitPythonEngine,
  PrintUsingEngine,
  PrintA,
  ChangeImagePython;
begin
end.
