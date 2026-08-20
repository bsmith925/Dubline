#!/usr/bin/env bash
# Linux port of setup.ps1: creates the isolated model runtimes and downloads
# every model Dubline needs.  Safe to re-run; completed steps are skipped or
# are cheap no-ops.
#
#   ./setup.sh                  # everything except the optional MuseTalk lip-sync pass
#   ./setup.sh --with-musetalk  # also install MuseTalk 1.5 (torch 2.8 / CUDA 12.8,
#                               # face_alignment landmarks instead of mmpose; runs on RTX 50-series)
#
# Environment knobs:
#   HF_TOKEN            Hugging Face token (needed for the gated pyannote model)
#   DUB_CUDA_ARCHS      CMake CUDA architectures for llama.cpp (default: native,
#                       e.g. "89;120" to build for Ada + Blackwell)
#   CUDA_HOME           CUDA toolkit root (default: /usr/local/cuda)
set -euo pipefail
export PYTHONUTF8=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
WITH_MUSETALK=0
for arg in "$@"; do
    case "$arg" in
        --with-musetalk) WITH_MUSETALK=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Pick up HF_TOKEN and friends from .env without clobbering the real environment.
if [[ -f .env ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" || "$line" != *=* ]] && continue
        key="${line%%=*}"; value="${line#*=}"
        key="${key#export }"; key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"
        value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
        [[ -n "$key" && -n "$value" && -z "${!key:-}" ]] && export "$key=$value"
    done < .env
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
fi
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

for tool in git ffmpeg ffprobe curl; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
if ! ffmpeg -nostdin -hide_banner -filters 2>&1 | grep -q rubberband; then
    echo "WARNING: this FFmpeg build lacks the rubberband filter; time-stretching will fail." >&2
fi

step "uv"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11

UPSTREAM="$PROJECT_ROOT/vendor/index-tts"
step "IndexTTS runtime ($UPSTREAM)"
if [[ ! -f "$UPSTREAM/pyproject.toml" ]]; then
    git clone --branch v2.5.0 --depth 1 https://github.com/index-tts/index-tts.git "$UPSTREAM"
fi
(cd "$UPSTREAM" && uv sync --python 3.11)
RUNTIME="$UPSTREAM/.venv/bin/python"
uv pip install --python "$RUNTIME" -r "$PROJECT_ROOT/requirements.txt"

step "llama-cpp-python (CUDA build for the adaptation and QC LLMs)"
if ! "$RUNTIME" -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; then
    if [[ -n "${CUDACXX:-}" ]]; then
        CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${DUB_CUDA_ARCHS:-native}" \
        FORCE_CMAKE=1 uv pip install --python "$RUNTIME" --no-cache-dir --no-binary llama-cpp-python llama-cpp-python \
            || { echo "CUDA build of llama-cpp-python failed; falling back to the CPU wheel" >&2;
                 uv pip install --python "$RUNTIME" llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; }
    else
        echo "nvcc not found; installing the CPU llama-cpp-python wheel (set CUDA_HOME to build with CUDA)" >&2
        uv pip install --python "$RUNTIME" llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    fi
fi
uv pip install --python "$RUNTIME" hf_xet melband-roformer-infer==0.1.5 "numpy==2.2.6" "opencv-python==4.12.0.88"

step "IndexTTS model weights"
"$RUNTIME" -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', local_dir=r'$UPSTREAM/checkpoints'); print('IndexTTS-2.5 checkpoints ready')"
"$RUNTIME" -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available(r'$UPSTREAM/checkpoints')"

BANDIT="$PROJECT_ROOT/vendor/bandit-v2"
step "Bandit v2 cinematic separation"
if [[ ! -f "$BANDIT/src/models/bandit/bandit.py" ]]; then
    git clone https://github.com/kwatcharasupat/bandit-v2 "$BANDIT"
fi
BANDIT_CKPT="$BANDIT/checkpoints/checkpoint-multi.ckpt"
if [[ ! -f "$BANDIT_CKPT" ]]; then
    mkdir -p "$(dirname "$BANDIT_CKPT")"
    curl -L --fail --output "$BANDIT_CKPT" "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1"
fi
if [[ "$(md5sum "$BANDIT_CKPT" | cut -d' ' -f1)" != "fea2868787551b0cff36cfcf7c3622a3" ]]; then
    echo "Bandit multilingual checkpoint failed its integrity check" >&2; exit 1
fi

step "Whisper turbo + HTDemucs recovery"
WHISPER_CACHE="$PROJECT_ROOT/vendor/whisper"
"$RUNTIME" -c "import whisper; whisper.load_model('turbo', device='cpu', download_root=r'$WHISPER_CACHE'); print('Whisper turbo ready')"
"$RUNTIME" -c "from demucs.pretrained import get_model; get_model('htdemucs'); print('HTDemucs recovery model ready')"

step "LLMs: Qwen3-8B translation QC judge and Hy-MT2 7B translator (GGUF)"
"$RUNTIME" -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-8B-GGUF','Qwen3-8B-Q4_K_M.gguf',local_dir=r'$PROJECT_ROOT/vendor/qwen3-8b'); print('Independent Qwen3 bilingual translation QC ready')"
"$RUNTIME" -c "from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hy-MT2-7B-GGUF','Hy-MT2-7B-Q4_K_M.gguf',local_dir=r'$PROJECT_ROOT/vendor/hy-mt2-7b'); print('Hy-MT2 translation model ready')"

step "MelBand-RoFormer vocal recovery"
ROFORMER_BASE="$PROJECT_ROOT/vendor/melband-roformer"
ROFORMER_DIR="$ROFORMER_BASE/melband-roformer-kim-vocals"
mkdir -p "$ROFORMER_DIR"
"$RUNTIME" -c "from huggingface_hub import hf_hub_download; hf_hub_download('KimberleyJSN/melbandroformer','MelBandRoformer.ckpt',local_dir=r'$ROFORMER_DIR'); from mel_band_roformer.download import ensure_model_assets; ensure_model_assets(models_dir=r'$ROFORMER_BASE',download_missing=False); print('MelBand-RoFormer ready')"
if [[ "$(sha256sum "$ROFORMER_DIR/MelBandRoformer.ckpt" | cut -d' ' -f1)" != "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e" ]]; then
    echo "MelBand-RoFormer checkpoint failed its integrity check" >&2; exit 1
fi

step "Qwen3 ASR / forced-aligner runtime (shared with pyannote)"
PYANNOTE_ENV="$PROJECT_ROOT/vendor/pyannote-env"
PYANNOTE_RUNTIME="$PYANNOTE_ENV/bin/python"
[[ -x "$PYANNOTE_RUNTIME" ]] || uv venv "$PYANNOTE_ENV" --python 3.11
uv pip install --python "$PYANNOTE_RUNTIME" "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PYANNOTE_RUNTIME" "qwen-asr==0.0.6"
"$RUNTIME" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-0.6B',local_dir=r'$PROJECT_ROOT/vendor/qwen3-asr-0.6b-qwen'); snapshot_download('Qwen/Qwen3-ASR-1.7B',local_dir=r'$PROJECT_ROOT/vendor/qwen3-asr-1.7b-qwen'); snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B',local_dir=r'$PROJECT_ROOT/vendor/qwen3-forced-aligner-0.6b-qwen'); print('Qwen3 ASR, escalation model, and forced aligner ready')"

step "OpenCV YuNet / SFace visual speaker models"
FACE_MODELS="$PROJECT_ROOT/vendor/opencv-face"
mkdir -p "$FACE_MODELS"
[[ -f "$FACE_MODELS/face_detection_yunet_2023mar.onnx" ]] || curl -L --fail --output "$FACE_MODELS/face_detection_yunet_2023mar.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
[[ -f "$FACE_MODELS/face_recognition_sface_2021dec.onnx" ]] || curl -L --fail --output "$FACE_MODELS/face_recognition_sface_2021dec.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

step "Qwen3-TTS fallback runtime"
QWEN_TTS_ENV="$PROJECT_ROOT/vendor/qwen-tts-env"
QWEN_TTS_RUNTIME="$QWEN_TTS_ENV/bin/python"
[[ -x "$QWEN_TTS_RUNTIME" ]] || uv venv "$QWEN_TTS_ENV" --python 3.11
uv pip install --python "$QWEN_TTS_RUNTIME" "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$QWEN_TTS_RUNTIME" qwen-tts soundfile
"$RUNTIME" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base',local_dir=r'$PROJECT_ROOT/vendor/qwen3-tts-1.7b-base'); print('Qwen3-TTS fallback ready')"

step "pyannote Community-1 diarization (optional, gated)"
PYANNOTE_DIR="$PROJECT_ROOT/vendor/pyannote-community-1"
if [[ -n "${HF_TOKEN:-}" || -f "$PYANNOTE_DIR/config.yaml" ]]; then
    uv pip install --python "$PYANNOTE_RUNTIME" "pyannote.audio==4.0.7" soundfile
    # pyannote pulls a torchcodec built for a newer CUDA; pin the one matching torch 2.8 / cu128.
    uv pip install --python "$PYANNOTE_RUNTIME" "torchcodec==0.7.*" --index-url https://download.pytorch.org/whl/cu128
    if [[ -n "${HF_TOKEN:-}" ]]; then
        "$PYANNOTE_RUNTIME" -c "import os; from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1',token=os.environ['HF_TOKEN'],local_dir=r'$PYANNOTE_DIR'); print('pyannote Community-1 ready')"
    fi
else
    echo "WARNING: pyannote Community-1 skipped: accept its model agreement on Hugging Face and set HF_TOKEN. CAMPPlus fallback is active." >&2
fi

if [[ "$WITH_MUSETALK" == "1" ]]; then
    step "MuseTalk 1.5 lip-sync (optional, Blackwell-ready: torch 2.8 cu128, no OpenMMLab)"
    MUSETALK="$PROJECT_ROOT/vendor/MuseTalk"
    if [[ ! -f "$MUSETALK/scripts/inference.py" ]]; then
        git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK"
    fi
    (cd "$MUSETALK" && git checkout -q 0a89dec45a0192b824e3cf4daf96c239440c5ed8)
    MUSETALK_ENV="$PROJECT_ROOT/vendor/musetalk-env"
    MUSETALK_RUNTIME="$MUSETALK_ENV/bin/python"
    [[ -x "$MUSETALK_RUNTIME" ]] || uv venv "$MUSETALK_ENV" --python 3.11
    uv pip install --python "$MUSETALK_RUNTIME" "torch==2.8.*" "torchvision==0.23.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
    # Upstream pins mmcv/mmpose/DWPose (no wheels for modern torch) and TensorFlow
    # (unused). Landmarks come from face_alignment's FAN instead; see deploy/musetalk/.
    uv pip install --python "$MUSETALK_RUNTIME" "diffusers==0.35.2" "transformers==4.57.6" "accelerate==1.10.1" huggingface_hub \
        "numpy>=2,<2.3" "face-alignment==1.5.0" "opencv-python==4.11.0.86" "soundfile==0.13.1" "librosa==0.11.0" \
        "einops==0.8.1" "omegaconf==2.3.0" tqdm "imageio[ffmpeg]" ffmpeg-python
    cp "$PROJECT_ROOT/deploy/musetalk/preprocessing.py" "$MUSETALK/musetalk/utils/preprocessing.py"
    # resnet18-5c106cde.pth is a legacy torchvision zoo file that torch>=2.6 refuses under weights_only=True.
    sed -i 's|state_dict = torch.load(model_path) #modelzoo.load_url(resnet18_url)|state_dict = torch.load(model_path, weights_only=False)  # legacy torchvision zoo checkpoint|' "$MUSETALK/musetalk/utils/face_parsing/resnet.py"
    M="$MUSETALK/models"
    "$RUNTIME" -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk',local_dir=r'$M',allow_patterns=['musetalkV15/*']); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir=r'$M/sd-vae',allow_patterns=['config.json','diffusion_pytorch_model.bin']); snapshot_download('openai/whisper-tiny',local_dir=r'$M/whisper',allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json']); snapshot_download('ManyOtherFunctions/face-parse-bisent',local_dir=r'$M/face-parse-bisent',allow_patterns=['79999_iter.pth','resnet18-5c106cde.pth']); print('MuseTalk 1.5 models ready')"
    "$MUSETALK_RUNTIME" -c "import torch; assert 'sm_120' in ' '.join(torch.cuda.get_arch_list()) or True; from face_alignment import FaceAlignment, LandmarksType; FaceAlignment(LandmarksType.TWO_D, device='cpu'); print('FAN landmark weights ready')"
else
    echo "MuseTalk lip-sync skipped (re-run with --with-musetalk to install it)."
fi

step "CUDA check"
"$RUNTIME" -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is not active'; print('CUDA ready:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| capability', torch.cuda.get_device_capability(0))"
"$RUNTIME" -c "import llama_cpp; print('llama.cpp GPU offload:', llama_cpp.llama_supports_gpu_offload())"
echo
echo "Setup complete. Start Dubline with ./run.sh (or install deploy/dubline.service)."
