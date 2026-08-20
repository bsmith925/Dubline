$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Upstream = Join-Path $ProjectRoot "vendor\index-tts"
Set-Location -LiteralPath $ProjectRoot
python -m uv --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    python -m pip install --user uv
}
Push-Location -LiteralPath $Upstream
try {
    python -m uv sync
} finally {
    Pop-Location
}
$Runtime = Join-Path $Upstream ".venv\Scripts\python.exe"
python -m uv pip install --python $Runtime -r (Join-Path $ProjectRoot "requirements.txt")
python -m uv pip install --python $Runtime llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m uv pip install --python $Runtime hf_xet melband-roformer-infer==0.1.5 "numpy==2.2.6" "opencv-python==4.12.0.88"
& $Runtime -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', local_dir=r'$Upstream\checkpoints'); print('IndexTTS-2.5 checkpoints ready')"
& $Runtime -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available(r'$Upstream\checkpoints')"
$Bandit = Join-Path $ProjectRoot "vendor\bandit-v2"
if (-not (Test-Path -LiteralPath (Join-Path $Bandit "src\models\bandit\bandit.py"))) {
    git clone https://github.com/kwatcharasupat/bandit-v2 $Bandit
}
$BanditCheckpoint = Join-Path $Bandit "checkpoints\checkpoint-multi.ckpt"
if (-not (Test-Path -LiteralPath $BanditCheckpoint)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $BanditCheckpoint) | Out-Null
    curl.exe -L --fail --output $BanditCheckpoint "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1"
}
$BanditHash = (Get-FileHash -LiteralPath $BanditCheckpoint -Algorithm MD5).Hash
if ($BanditHash -ne "FEA2868787551B0CFF36CFCF7C3622A3") {
    throw "Bandit multilingual checkpoint failed its integrity check"
}
$WhisperCache = Join-Path $ProjectRoot "vendor\whisper"
& $Runtime -c "import whisper; whisper.load_model('turbo', device='cpu', download_root=r'$WhisperCache'); print('Whisper turbo ready')"
& $Runtime -c "from demucs.pretrained import get_model; get_model('htdemucs'); print('HTDemucs recovery model ready')"
$TranslationQcDir = Join-Path $ProjectRoot "vendor\qwen3-8b"
& $Runtime -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-8B-GGUF','Qwen3-8B-Q4_K_M.gguf',local_dir=r'$TranslationQcDir'); print('Independent Qwen3 bilingual translation QC ready')"
$TranslationDir = Join-Path $ProjectRoot "vendor\hy-mt2-7b"
& $Runtime -c "from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hy-MT2-7B-GGUF','Hy-MT2-7B-Q4_K_M.gguf',local_dir=r'$TranslationDir'); print('Hy-MT2 translation model ready')"
$RoformerBase = Join-Path $ProjectRoot "vendor\melband-roformer"
$RoformerDir = Join-Path $RoformerBase "melband-roformer-kim-vocals"
New-Item -ItemType Directory -Force -Path $RoformerDir | Out-Null
& $Runtime -c "from huggingface_hub import hf_hub_download; hf_hub_download('KimberleyJSN/melbandroformer','MelBandRoformer.ckpt',local_dir=r'$RoformerDir'); from mel_band_roformer.download import ensure_model_assets; ensure_model_assets(models_dir=r'$RoformerBase',download_missing=False); print('MelBand-RoFormer ready')"
$RoformerHash = (Get-FileHash -LiteralPath (Join-Path $RoformerDir "MelBandRoformer.ckpt") -Algorithm SHA256).Hash
if ($RoformerHash -ne "87201F4D31AFB5BC79993230FC49446918425574DB48C01C405E44F365C7559E") {
    throw "MelBand-RoFormer checkpoint failed its integrity check"
}
$PyannoteRuntime = Join-Path $ProjectRoot "vendor\pyannote-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PyannoteRuntime)) {
    python -m uv venv (Join-Path $ProjectRoot "vendor\pyannote-env") --python 3.11
}
python -m uv pip install --python $PyannoteRuntime "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python $PyannoteRuntime "qwen-asr==0.0.6"
$AsrDir = Join-Path $ProjectRoot "vendor\qwen3-asr-0.6b-qwen"
$AsrEscalationDir = Join-Path $ProjectRoot "vendor\qwen3-asr-1.7b-qwen"
$AlignerDir = Join-Path $ProjectRoot "vendor\qwen3-forced-aligner-0.6b-qwen"
& $Runtime -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-0.6B',local_dir=r'$AsrDir'); snapshot_download('Qwen/Qwen3-ASR-1.7B',local_dir=r'$AsrEscalationDir'); snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B',local_dir=r'$AlignerDir'); print('Qwen3 ASR, escalation model, and forced aligner ready')"
$FaceModels = Join-Path $ProjectRoot "vendor\opencv-face"
New-Item -ItemType Directory -Force -Path $FaceModels | Out-Null
$YuNet = Join-Path $FaceModels "face_detection_yunet_2023mar.onnx"
$SFace = Join-Path $FaceModels "face_recognition_sface_2021dec.onnx"
if (-not (Test-Path -LiteralPath $YuNet)) {
    curl.exe -L --fail --output $YuNet "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
}
if (-not (Test-Path -LiteralPath $SFace)) {
    curl.exe -L --fail --output $SFace "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}
$QwenTtsRuntime = Join-Path $ProjectRoot "vendor\qwen-tts-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $QwenTtsRuntime)) {
    python -m uv venv (Join-Path $ProjectRoot "vendor\qwen-tts-env") --python 3.11
}
python -m uv pip install --python $QwenTtsRuntime "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python $QwenTtsRuntime qwen-tts soundfile
$QwenTtsDir = Join-Path $ProjectRoot "vendor\qwen3-tts-1.7b-base"
& $Runtime -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base',local_dir=r'$QwenTtsDir'); print('Qwen3-TTS fallback ready')"
$PyannoteDir = Join-Path $ProjectRoot "vendor\pyannote-community-1"
if ($env:HF_TOKEN -or (Test-Path -LiteralPath (Join-Path $PyannoteDir "config.yaml"))) {
    python -m uv pip install --python $PyannoteRuntime "pyannote.audio==4.0.7" soundfile
    if ($env:HF_TOKEN) {
        & $PyannoteRuntime -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1',token='$env:HF_TOKEN',local_dir=r'$PyannoteDir'); print('pyannote Community-1 ready')"
    }
} else {
    Write-Warning "pyannote Community-1 remains optional because its publisher requires accepting the model agreement and setting HF_TOKEN. CAMPPlus fallback is active."
}
$MuseTalk = Join-Path $ProjectRoot "vendor\MuseTalk"
if (-not (Test-Path -LiteralPath (Join-Path $MuseTalk "scripts\inference.py"))) {
    git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git $MuseTalk
}
$MuseTalkRuntime = Join-Path $ProjectRoot "vendor\musetalk-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $MuseTalkRuntime)) {
    python -m uv python install 3.10
    python -m uv venv (Join-Path $ProjectRoot "vendor\musetalk-env") --python 3.10
}
python -m uv pip install --python $MuseTalkRuntime torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m uv pip install --python $MuseTalkRuntime -r (Join-Path $MuseTalk "requirements.txt")
python -m uv pip install --python $MuseTalkRuntime openmim
& $MuseTalkRuntime -m mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0"
& $MuseTalkRuntime -m pip install "chumpy==0.70" --no-build-isolation
& $MuseTalkRuntime -m mim install "mmpose==1.1.0"
$MuseTalkModels = Join-Path $MuseTalk "models"
& $Runtime -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk',local_dir=r'$MuseTalkModels'); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir=r'$MuseTalkModels\sd-vae',allow_patterns=['config.json','diffusion_pytorch_model.bin']); snapshot_download('openai/whisper-tiny',local_dir=r'$MuseTalkModels\whisper',allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json']); snapshot_download('yzd-v/DWPose',local_dir=r'$MuseTalkModels\dwpose',allow_patterns=['dw-ll_ucoco_384.pth']); snapshot_download('ByteDance/LatentSync',local_dir=r'$MuseTalkModels\syncnet',allow_patterns=['latentsync_syncnet.pt']); snapshot_download('ManyOtherFunctions/face-parse-bisent',local_dir=r'$MuseTalkModels\face-parse-bisent',allow_patterns=['79999_iter.pth','resnet18-5c106cde.pth']); print('MuseTalk 1.5 finishing models ready')"
Push-Location -LiteralPath $MuseTalk
try {
    # This initializes DWPose and downloads MuseTalk's small S3FD detector to the
    # normal Torch checkpoint cache now, rather than on the first unattended film.
    & $MuseTalkRuntime -c "from musetalk.utils.preprocessing import get_landmark_and_bbox; print('MuseTalk face models ready')"
} finally {
    Pop-Location
}
& $Runtime -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is not active'; print('CUDA ready:', torch.cuda.get_device_name(0))"
Write-Host "Setup complete. Start Dubline with .\run.ps1"
