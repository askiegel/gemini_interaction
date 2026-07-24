param(
    [switch]$Execute,
    [switch]$Continuous,
    [int]$ListenSeconds = 10,
    [double]$MinimumConfidence = 0.35
)

$ErrorActionPreference = "Stop"

$WslProjectDirectory = "~/robot_services/cognitive"

function Get-EnglishRecognizer {
    Add-Type -AssemblyName System.Speech

    $recognizers =
        [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()

    if ($recognizers.Count -eq 0) {
        throw "No Windows speech recognizer is installed."
    }

    $recognizerInfo =
        $recognizers |
        Where-Object { $_.Culture.Name -eq "en-US" } |
        Select-Object -First 1

    if ($null -eq $recognizerInfo) {
        $recognizerInfo = $recognizers | Select-Object -First 1
    }

    return $recognizerInfo
}

function New-CommandRecognizer {
    $recognizerInfo = Get-EnglishRecognizer

    Write-Host "Recognizer: $($recognizerInfo.Name)"
    Write-Host "Culture:    $($recognizerInfo.Culture)"

    $recognizer =
        New-Object System.Speech.Recognition.SpeechRecognitionEngine(
            $recognizerInfo
        )

    $recognizer.SetInputToDefaultAudioDevice()

    $commandChoices =
        New-Object System.Speech.Recognition.Choices

    $commandChoices.Add(@(
        "move forward",
        "go forward",
        "forward",
        "turn left",
        "go left",
        "left",
        "turn right",
        "go right",
        "right",
        "stop",
        "stop moving",
        "find my backpack",
        "find the backpack",
        "find backpack"
    ))

    $grammarBuilder =
        [System.Speech.Recognition.GrammarBuilder]::new()

    $grammarBuilder.Culture = $recognizerInfo.Culture
    $grammarBuilder.Append($commandChoices)

    $grammar =
        New-Object `
            -TypeName System.Speech.Recognition.Grammar `
            -ArgumentList (, $grammarBuilder)

    $grammar.Name = "MiniPupperCommands"

    $recognizer.LoadGrammar($grammar)

    $recognizer.InitialSilenceTimeout =
        [TimeSpan]::FromSeconds($ListenSeconds)

    $recognizer.BabbleTimeout =
        [TimeSpan]::FromSeconds(2)

    $recognizer.EndSilenceTimeout =
        [TimeSpan]::FromMilliseconds(600)

    $recognizer.EndSilenceTimeoutAmbiguous =
        [TimeSpan]::FromMilliseconds(900)

    return $recognizer
}

function Convert-ToCanonicalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RecognizedText
    )

    switch ($RecognizedText.Trim().ToLowerInvariant()) {
        "move forward"      { return "Move forward" }
        "go forward"        { return "Move forward" }
        "forward"           { return "Move forward" }

        "turn left"         { return "Turn left" }
        "go left"           { return "Turn left" }
        "left"              { return "Turn left" }

        "turn right"        { return "Turn right" }
        "go right"          { return "Turn right" }
        "right"             { return "Turn right" }

        "stop"              { return "Stop" }
        "stop moving"       { return "Stop" }

        "find my backpack"  { return "Find my backpack" }
        "find the backpack" { return "Find my backpack" }
        "find backpack"     { return "Find my backpack" }

        default             { return $RecognizedText.Trim() }
    }
}

function Invoke-CognitiveCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RecognizedText
    )

    $canonicalCommand =
        Convert-ToCanonicalCommand -RecognizedText $RecognizedText

    $env:MINI_PUPPER_VOICE_COMMAND = $canonicalCommand

    if ([string]::IsNullOrWhiteSpace($env:WSLENV)) {
        $env:WSLENV = "MINI_PUPPER_VOICE_COMMAND"
    }
    elseif (
        $env:WSLENV -notmatch
        "(^|:)MINI_PUPPER_VOICE_COMMAND($|:)"
    ) {
        $env:WSLENV =
            "$($env:WSLENV):MINI_PUPPER_VOICE_COMMAND"
    }

    $executeArgument = ""

    if ($Execute) {
        $executeArgument = "--execute"
    }

    $wslCommand = @'
cd ~/robot_services/cognitive || exit 1

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python3 voice_command.py \
    --text "$MINI_PUPPER_VOICE_COMMAND" \
    __EXECUTE_ARGUMENT__
'@

    $wslCommand =
        $wslCommand.Replace(
            "__EXECUTE_ARGUMENT__",
            $executeArgument
        )

    Write-Host ""
    Write-Host "===== WINDOWS VOICE RELAY =====" -ForegroundColor Cyan
    Write-Host "Heard:      $RecognizedText"
    Write-Host "Command:    $canonicalCommand"

    if ($Execute) {
        Write-Host "Mode:       LIVE EXECUTION" -ForegroundColor Yellow
    }
    else {
        Write-Host "Mode:       DRY RUN" -ForegroundColor Green
    }

    Write-Host ""

    & wsl.exe bash -lc $wslCommand

    if ($LASTEXITCODE -ne 0) {
        throw "WSL cognitive pipeline failed with exit code $LASTEXITCODE."
    }
}

function Read-SpokenCommand {
    param(
        [Parameter(Mandatory = $true)]
        [System.Speech.Recognition.SpeechRecognitionEngine]$Recognizer
    )

    Write-Host ""
    Write-Host "Listening for $ListenSeconds seconds..." -ForegroundColor Cyan
    Write-Host "Wait one second, then say one command clearly:"
    Write-Host "  Move forward"
    Write-Host "  Turn left"
    Write-Host "  Turn right"
    Write-Host "  Stop"
    Write-Host "  Find my backpack"
    Write-Host ""

    $state = [hashtable]::Synchronized(@{
        Completed  = $false
        Recognized = $false
        Rejected   = $false
        Text       = $null
        Confidence = 0.0
        MaxAudio   = 0
        Error      = $null
    })

    $speechRecognizedHandler = {
        param($sender, $eventArgs)

        $state.Text = $eventArgs.Result.Text
        $state.Confidence = $eventArgs.Result.Confidence
        $state.Recognized = $true
    }

    $speechRejectedHandler = {
        param($sender, $eventArgs)

        $state.Rejected = $true

        if ($null -ne $eventArgs.Result) {
            $state.Text = $eventArgs.Result.Text
            $state.Confidence = $eventArgs.Result.Confidence
        }
    }

    $audioLevelHandler = {
        param($sender, $eventArgs)

        if ($eventArgs.AudioLevel -gt $state.MaxAudio) {
            $state.MaxAudio = $eventArgs.AudioLevel
        }
    }

    $recognizeCompletedHandler = {
        param($sender, $eventArgs)

        if ($null -ne $eventArgs.Error) {
            $state.Error = $eventArgs.Error.Message
        }

        $state.Completed = $true
    }

    $Recognizer.add_SpeechRecognized($speechRecognizedHandler)
    $Recognizer.add_SpeechRecognitionRejected($speechRejectedHandler)
    $Recognizer.add_AudioLevelUpdated($audioLevelHandler)
    $Recognizer.add_RecognizeCompleted($recognizeCompletedHandler)

    try {
        $Recognizer.RecognizeAsync(
            [System.Speech.Recognition.RecognizeMode]::Single
        )

        $deadline = (Get-Date).AddSeconds($ListenSeconds)

        while (
            -not $state.Completed -and
            (Get-Date) -lt $deadline
        ) {
            Start-Sleep -Milliseconds 100
        }

        if (-not $state.Completed) {
            $Recognizer.RecognizeAsyncCancel()

            $cancelDeadline = (Get-Date).AddSeconds(2)

            while (
                -not $state.Completed -and
                (Get-Date) -lt $cancelDeadline
            ) {
                Start-Sleep -Milliseconds 100
            }
        }
    }
    finally {
        $Recognizer.remove_SpeechRecognized($speechRecognizedHandler)
        $Recognizer.remove_SpeechRecognitionRejected($speechRejectedHandler)
        $Recognizer.remove_AudioLevelUpdated($audioLevelHandler)
        $Recognizer.remove_RecognizeCompleted($recognizeCompletedHandler)
    }

    Write-Host "Maximum microphone level: $($state.MaxAudio)"

    if (-not [string]::IsNullOrWhiteSpace($state.Error)) {
        Write-Host "Recognition error: $($state.Error)" -ForegroundColor Red
        return $null
    }

    if (-not $state.Recognized) {
        if ($state.Rejected) {
            Write-Host "Speech was heard but did not match a command." `
                -ForegroundColor Yellow

            if (-not [string]::IsNullOrWhiteSpace($state.Text)) {
                Write-Host "Closest text: $($state.Text)"
                Write-Host (
                    "Confidence: {0:P0}" -f $state.Confidence
                )
            }
        }
        else {
            Write-Host "No command was recognized." -ForegroundColor Yellow
        }

        return $null
    }

    Write-Host "Text:       $($state.Text)"
    Write-Host (
        "Confidence: {0:P0}" -f $state.Confidence
    )

    if ($state.Confidence -lt $MinimumConfidence) {
        Write-Host (
            "Rejected: confidence was below {0:P0}." -f
            $MinimumConfidence
        ) -ForegroundColor Yellow

        return $null
    }

    return $state.Text
}

Write-Host "============================================"
Write-Host " Mini Pupper 2 Windows Voice Relay"
Write-Host "============================================"

if ($Execute) {
    Write-Host ""
    Write-Host "WARNING: LIVE ROBOT EXECUTION IS ENABLED." `
        -ForegroundColor Yellow
}
else {
    Write-Host ""
    Write-Host "Dry-run mode is enabled." `
        -ForegroundColor Green
    Write-Host "The robot will not move."
}

$recognizer = $null

try {
    $recognizer = New-CommandRecognizer

    do {
        $spokenCommand =
            Read-SpokenCommand -Recognizer $recognizer

        if (
            -not [string]::IsNullOrWhiteSpace(
                $spokenCommand
            )
        ) {
            Invoke-CognitiveCommand `
                -RecognizedText $spokenCommand
        }

        if ($Continuous) {
            Write-Host ""
            Write-Host "Listening again. Press Ctrl+C to exit."
        }
    }
    while ($Continuous)
}
finally {
    if ($null -ne $recognizer) {
        $recognizer.Dispose()
    }

    Remove-Item `
        Env:MINI_PUPPER_VOICE_COMMAND `
        -ErrorAction SilentlyContinue
}
