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
unit main_app_real;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, Vcl.Graphics, Vcl.Controls, Vcl.Forms, Vcl.Dialogs,
  System.IOUtils, Vcl.StdCtrls, Jpeg, ExtCtrls, ComCtrls, PythonVersions,
  PythonEngine, Vcl.PythonGUIInputOutput,ShellApi, Vcl.ExtDlgs,
  System.Diagnostics;

type
  TForm1 = class(TForm)
    DrawNextFile: TButton;
    CurrentFileName: TLabel;
    CurrentFileNumber: TLabel;
    Image1: TImage;
    Image2: TImage;
    PythonTime: TLabel;
    StaticText1: TStaticText;
    StaticText2: TStaticText;
    Image3: TImage;
    StaticText3: TStaticText;
    Label1: TLabel;
    procedure DrawNextFileClick(Sender: TObject);

  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;
  FileNames: TArray<string>;
  VCurrentFileName: string;
  VCurrentFileNumber: Integer;

const
  PythonDLLPath = 'C:\Users\Nikodem\Documents\GitHub\autonomous-training-xray\pythonDll\Win64\Debug\pythonDll.dll'; // Modify this path to your DLL location
  LoaderPath =  'C:\Users\Nikodem\Desktop\Images\PhotoLoaderReal';
  OriginalPath = 'C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\original_image.bmp';
  RBAPath =    'C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\segmented_image.bmp';
  MLPath =   'C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\ml_image.bmp';


implementation



procedure InitPythonEngine; stdcall; external PythonDLLPath;
procedure ChangeImagePython; stdcall; external PythonDLLPath;
procedure StartTrainingThread; stdcall; external PythonDLLPath;
procedure TerminateTrainingThread; stdcall; external PythonDLLPath;


/// <summary>
/// Shuffles the elements in the specified array.
/// </summary>
/// <param name="Arr">
/// The array to shuffle.
/// </param>
/// <remarks>
/// This procedure uses the Fisher-Yates algorithm to shuffle the elements
/// </remarks>
/// <returns> 
/// None.
/// </returns>
procedure ShuffleArray(var Arr: TArray<string>);
var
  i, j: Integer;
  Temp: string;
begin
  Randomize;
  for i := Length(Arr) - 1 downto 1 do
  begin
    j := Random(i + 1);
    Temp := Arr[i];
    Arr[i] := Arr[j];
    Arr[j] := Temp;
  end;
end;

/// <summary>
/// Loads a JPEG image from the specified file.
/// </summary>
/// <param name="FileName">
/// The name of the file to load.
/// </param>
/// <returns>
/// A TBitmap object containing the loaded image.
/// </returns>
function LoadJPEGasBitmap(const FileName: string): TBitmap;
var
  Jpg: TJpegImage;
begin
  Result := TBitmap.Create;
  Jpg := TJpegImage.Create;
  try
    Jpg.LoadFromFile(FileName);
    Result.Assign(Jpg);
  finally
    Jpg.Free;
  end;
end;

/// <summary>
/// Loads a bitmap image from the specified file.
/// </summary>
/// <param name="FileName">
/// The name of the file to load.
/// </param>
/// <returns>
/// A TBitmap object containing the loaded image.
/// </returns>
function LoadBitmap(const FileName: string): TBitmap;
var
  Picture: TPicture;
begin
  Result := TBitmap.Create;
  Picture := TPicture.Create;
  try
    Picture.LoadFromFile(FileName);
    Result.Assign(Picture.Bitmap);
  finally
    Picture.Free;
  end;
end;


/// <summary>
/// Loads the file names from the specified directory.
/// </summary>
/// <returns>
/// None.
/// </returns>
procedure loadFileNames();
var
  i: Integer;
begin
  // Read the file names
  FileNames:= TDirectory.GetFiles(LoaderPath, '*.bmp');
  OutputDebugString(PChar('[INFO]: File names loaded, first file name: ' + FileNames[0]));

  // Shuffle the array
  ShuffleArray(FileNames);
  OutputDebugString(PChar('[INFO]: File names shuffled, first file name: ' + FileNames[0]));

  // Current file
  VCurrentFileName := FileNames[0];
  VCurrentFileNumber := -1;
end;

/// <summary>
/// Loads and displays the next file.
/// </summary>
/// <param name="FileName">
/// The name of the file to display.
/// </param>
/// <returns>
/// None.
/// </returns>
procedure drawFile(const FileName: string);
var
  MyBitmap: TBitmap;
begin
  MyBitmap := LoadBitmap(FileName);
  try
    // Use the bitmap here...
    with Form1.Image1 do
    begin
      Canvas.Brush.Bitmap := MyBitmap;
      Canvas.FillRect(Rect(0,0,448,500));
      OutputDebugString('[INFO]: Drawing successful' );
    end;

  finally
    with Form1.Image1 do
      Canvas.Brush.Bitmap := nil;
      MyBitmap.Free;
  end;

end;

/// <summary>
/// Performs the rule based algorithm and machine learning inference.
/// </summary>
/// <returns>
/// None.
/// </returns>
procedure extractCherries;
var
  MyBitmap: TBitmap;
  TransformedBitmap: TBitmap;
  MLBitmap: TBitmap;
  Stopwatch: TStopwatch;
begin
  MyBitmap := LoadBitmap(VCurrentFileName);
  MyBitmap.SaveToFile(OriginalPath);

  // Start the stopwatch
  Stopwatch := TStopwatch.StartNew;

  ChangeImagePython;

  // Stop the stopwatch
  Stopwatch.Stop;
  Form1.PythonTime.Caption := 'Elapsed time: ' + IntToStr(Stopwatch.ElapsedMilliseconds) + ' ms';

  TransformedBitmap := TBitmap.Create;
  TransformedBitmap.LoadFromFile(RBAPath);
  try
    with Form1.Image2 do
    begin
      Canvas.Brush.Bitmap := TransformedBitmap;
      Canvas.FillRect(Rect(0,0,448,500));
      OutputDebugString('[INFO]: Processing Done' );
    end;

  finally
    with Form1.Image2 do
      begin
        Canvas.Brush.Bitmap := nil;
        MyBitmap.Free;
        TransformedBitmap.Free
      end;
  end;

  MLBitmap := TBitmap.Create;
  MLBitmap.LoadFromFile(MLPath);
  try
    with Form1.Image3 do
    begin
      Canvas.Brush.Bitmap := MLBitmap;
      Canvas.FillRect(Rect(0,0,448,500));
      OutputDebugString('[INFO]: Processing Done' );
    end;

  finally
    with Form1.Image3 do
      begin
        Canvas.Brush.Bitmap := nil;
        TransformedBitmap.Free
      end;
  end;
end;

// The .dfm file (Delphi Form Module) contains the visual representation 
// and properties of a form, including its components, their positions, 
// and their properties.
{$R *.dfm}


/// <summary>
/// Iterates to the next file name and displays it.
/// </summary>
/// <param name="Sender">
/// Button click event.
/// </param>
/// <returns>
/// None.
/// </returns>
procedure TForm1.DrawNextFileClick(Sender: TObject);
var
  FileName: string;
begin
  // Initialisation
  if VCurrentFileNumber = -1 then
  begin
       VCurrentFileNumber := 0;
       VCurrentFileName:=  FileNames[0];
  end
  // Found end of array
  else if VCurrentFileNumber = Length(FileNames) then
  begin
       VCurrentFileNumber := 0;
       VCurrentFileName:=  FileNames[0];
  end
  // Iterate to next file name
  else
  begin
    VCurrentFileNumber := VCurrentFileNumber + 1;
    VCurrentFileName:=  FileNames[VCurrentFileNumber];
    Form1.CurrentFileName.Caption := VCurrentFileName;
    Form1.CurrentFileNumber.Caption := IntToStr(VCurrentFileNumber);
  end;


  drawFile(VCurrentFileName);
  extractCherries;
end;





initialization
  MaskFPUExceptions(True);   // Needed for Python type compatibility
  loadFileNames;
  InitPythonEngine;
  StartTrainingThread;


finalization
  TerminateTrainingThread;


end.
