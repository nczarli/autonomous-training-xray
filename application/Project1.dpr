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
program Project1;

uses
  Vcl.Forms,
  main_app_real in 'main_app_real.pas' {Form1};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
