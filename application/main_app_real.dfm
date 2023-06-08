object Form1: TForm1
  Left = 0
  Top = 0
  BorderStyle = bsToolWindow
  Caption = 'Detector'
  ClientHeight = 712
  ClientWidth = 1495
  Color = clWhite
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object CurrentFileName: TLabel
    Left = 213
    Top = 658
    Width = 143
    Height = 25
    Caption = 'CurrentFileName'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
  end
  object CurrentFileNumber: TLabel
    Left = 1247
    Top = 658
    Width = 162
    Height = 25
    Caption = 'CurrentFileNumber'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
  end
  object Image1: TImage
    Left = 24
    Top = 24
    Width = 448
    Height = 500
  end
  object Image2: TImage
    Left = 520
    Top = 24
    Width = 448
    Height = 500
  end
  object PythonTime: TLabel
    Left = 662
    Top = 583
    Width = 100
    Height = 25
    Caption = 'PythonTime'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
  end
  object Image3: TImage
    Left = 1008
    Top = 24
    Width = 448
    Height = 500
  end
  object Label1: TLabel
    Left = 1135
    Top = 658
    Width = 63
    Height = 25
    Caption = 'File No.'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
  end
  object DrawNextFile: TButton
    Left = 24
    Top = 651
    Width = 175
    Height = 41
    Caption = 'Draw Next File'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
    TabOrder = 0
    OnClick = DrawNextFileClick
  end
  object StaticText1: TStaticText
    Left = 192
    Top = 536
    Width = 105
    Height = 29
    Caption = 'Input Image'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
    TabOrder = 1
  end
  object StaticText2: TStaticText
    Left = 704
    Top = 536
    Width = 185
    Height = 29
    Caption = 'Rule Based Algorithm'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
    TabOrder = 2
  end
  object StaticText3: TStaticText
    Left = 1200
    Top = 530
    Width = 156
    Height = 29
    Caption = 'Machine Learning'
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'Segoe UI'
    Font.Style = []
    ParentFont = False
    TabOrder = 3
  end
end
