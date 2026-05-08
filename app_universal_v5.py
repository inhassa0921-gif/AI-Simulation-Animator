# -*- coding: utf-8 -*-
"""
AI Simulation Animator - Universal GPT-generated simulation runner.

GPT-generated code must provide:
1. PARAM_CONFIG
2. init_state(params)
3. update_state(state, params, frame)
4. draw_frame(ax, state, params, frame)
Optional:
5. FIGURE_CONFIG
6. helper functions

This app is intentionally topic-agnostic.
"""

import io
import json
import math
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import streamlit as st


# =========================================================
# Page
# =========================================================
st.set_page_config(
    page_title="AI Simulation Animator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(251, 207, 232, 0.30), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(191, 219, 254, 0.30), transparent 30%),
        linear-gradient(180deg, #fbfbff 0%, #f8fafc 100%);
}

.hero {
    padding: 1.35rem 1.45rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #fff7ed 0%, #eef2ff 52%, #ecfeff 100%);
    border: 1px solid rgba(148, 163, 184, 0.30);
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 2.05rem;
    line-height: 1.15;
    font-weight: 850;
    color: #111827;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #475569;
    margin-bottom: 0.35rem;
}

.pill {
    display: inline-block;
    padding: 0.32rem 0.68rem;
    margin: 0.16rem 0.2rem 0.16rem 0;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.38);
    color: #334155;
    font-size: 0.86rem;
    font-weight: 650;
}

.card {
    padding: 1rem 1.05rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
    margin-bottom: 0.75rem;
}

.good { color: #047857; font-weight: 800; }
.warn { color: #b45309; font-weight: 800; }
.bad { color: #be123c; font-weight: 800; }

div[data-testid="stTabs"] button { font-weight: 750; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
.stButton button, .stDownloadButton button {
    border-radius: 13px !important;
    font-weight: 800 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

APP_VERSION = "v5.0 universal-safe-runner"

DEFAULT_CODE = r"""
PARAM_CONFIG = {
    "amplitude": {
        "type": "slider",
        "min": 0.1,
        "max": 2.0,
        "default": 1.0,
        "step": 0.05,
        "description": "Wave amplitude."
    },
    "frequency": {
        "type": "slider",
        "min": 0.5,
        "max": 8.0,
        "default": 2.0,
        "step": 0.1,
        "description": "Wave frequency."
    },
    "phase_speed": {
        "type": "slider",
        "min": 0.0,
        "max": 0.4,
        "default": 0.12,
        "step": 0.01,
        "description": "Animation phase speed."
    },
    "line_color": {
        "type": "color",
        "default": "#7c3aed",
        "description": "Main curve color."
    }
}

FIGURE_CONFIG = {
    "projection": "2d",
    "figsize": (8.5, 5.0),
    "dpi": 120
}

def init_state(params):
    x = np.linspace(0, 1, 400)
    y = np.zeros_like(x)
    return {"x": x, "y": y}

def update_state(state, params, frame):
    x = state["x"]
    amp = params["amplitude"]
    freq = params["frequency"]
    phase_speed = params["phase_speed"]
    state["y"] = amp * np.sin(2 * np.pi * freq * x - frame * phase_speed)
    return state

def draw_frame(ax, state, params, frame):
    ax.clear()
    ax.plot(state["x"], state["y"], color=params["line_color"], linewidth=2.8, label="Animated wave")
    ax.set_title("Universal GPT-generated Simulation Example", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalized position")
    ax.set_ylabel("Response")
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
"""

REQUIRED_NAMES = ["PARAM_CONFIG", "init_state", "update_state", "draw_frame"]

for key, default in {
    "raw_code_input": DEFAULT_CODE,
    "generated_gif_bytes": None,
    "generated_gif_name": None,
    "generated_png_bytes": None,
    "generated_png_name": None,
    "generated_prompt_text": "",
    "last_exec_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# Utility
# =========================================================
def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "simulation"


def bytes_name(prefix: str, ext: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_filename(prefix)}_{stamp}.{ext}"


def extract_code_from_text(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        ranked = []
        for block in blocks:
            score = sum(1 for name in REQUIRED_NAMES if name in block)
            if "FIGURE_CONFIG" in block:
                score += 0.5
            ranked.append((score, block.strip()))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked and ranked[0][0] > 0:
            return ranked[0][1]
        return blocks[-1].strip()
    return text.strip()


def strip_imports(code: str) -> str:
    kept = []
    removed = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            removed.append(line)
        else:
            kept.append(line)
    if removed:
        st.warning("보안/호환성을 위해 import 문을 제거했습니다. np와 math는 이미 제공됩니다.")
    return "\n".join(kept)


def safe_builtins() -> Dict[str, Any]:
    names = [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "pow", "range",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
        "print", "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    ]
    builtins_obj = __builtins__
    if isinstance(builtins_obj, dict):
        return {name: builtins_obj[name] for name in names if name in builtins_obj}
    return {name: getattr(builtins_obj, name) for name in names if hasattr(builtins_obj, name)}


def execute_user_code(raw_code: str, allow_imports: bool = False) -> Tuple[Dict[str, Any], str]:
    code = extract_code_from_text(raw_code)
    if not allow_imports:
        code = strip_imports(code)

    namespace: Dict[str, Any] = {
        "np": np,
        "math": math,
        "__builtins__": safe_builtins(),
    }

    # Important:
    # globals and locals are the SAME dictionary.
    # This lets helper functions call each other.
    exec(code, namespace, namespace)
    return namespace, code


def validate_namespace(namespace: Dict[str, Any]) -> Tuple[bool, list]:
    problems = []
    if "PARAM_CONFIG" not in namespace:
        problems.append("PARAM_CONFIG가 없습니다.")
    elif not isinstance(namespace["PARAM_CONFIG"], dict):
        problems.append("PARAM_CONFIG는 dictionary여야 합니다.")

    for fn in ["init_state", "update_state", "draw_frame"]:
        if fn not in namespace:
            problems.append(f"{fn} 함수가 없습니다.")
        elif not callable(namespace[fn]):
            problems.append(f"{fn}가 함수가 아닙니다.")
    return len(problems) == 0, problems


def normalize_param_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, cfg in config.items():
        if isinstance(cfg, dict):
            out[str(name)] = dict(cfg)
        else:
            out[str(name)] = {"type": "number", "default": cfg, "description": "Auto-wrapped parameter."}
    return out


def infer_widget_type(name: str, cfg: Dict[str, Any]) -> str:
    t = str(cfg.get("type", "")).lower().strip()
    aliases = {
        "float": "number",
        "double": "number",
        "bool": "checkbox",
        "boolean": "checkbox",
        "str": "text",
        "string": "text",
        "dropdown": "select",
        "choice": "select",
    }
    if t:
        return aliases.get(t, t)

    lower = name.lower()
    if "color" in lower or "colour" in lower:
        return "color"
    if any(k in lower for k in ["show", "use", "enable", "boundary", "clip", "normalize", "loop"]):
        return "checkbox"
    if any(k in lower for k in ["name", "label", "title", "mode", "material", "type"]):
        return "text"
    if any(k in lower for k in ["num", "count", "number", "seed", "frames", "fps", "steps"]):
        return "int"
    return "slider"


def clamp(value: float, lo: float, hi: float) -> float:
    lo, hi = float(lo), float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return float(min(max(float(value), lo), hi))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_parameter_ui(param_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not param_config:
        st.sidebar.info("PARAM_CONFIG가 인식되면 여기에 조절창이 자동 생성됩니다.")
        return params

    for name, cfg in param_config.items():
        wtype = infer_widget_type(name, cfg)
        desc = str(cfg.get("description", ""))
        unit = str(cfg.get("unit", "")).strip()
        label = f"{name}" + (f" ({unit})" if unit else "")
        default = cfg.get("default", cfg.get("value", 0.0))

        try:
            if wtype == "checkbox":
                params[name] = st.sidebar.checkbox(label, value=bool(default), help=desc)

            elif wtype == "color":
                color_default = str(default) if str(default).startswith("#") else "#7c3aed"
                params[name] = st.sidebar.color_picker(label, value=color_default, help=desc)

            elif wtype == "text":
                params[name] = st.sidebar.text_input(label, value=str(default), help=desc)

            elif wtype == "select":
                options = cfg.get("options", [])
                if not isinstance(options, (list, tuple)) or len(options) == 0:
                    options = [str(default)]
                options = [str(o) for o in options]
                default_str = str(default) if str(default) in options else options[0]
                params[name] = st.sidebar.selectbox(label, options, index=options.index(default_str), help=desc)

            elif wtype == "int":
                lo = int(cfg.get("min", 0))
                hi = int(cfg.get("max", 100000))
                step = int(cfg.get("step", 1))
                val = int(clamp(int(cfg.get("default", lo)), lo, hi))
                params[name] = st.sidebar.number_input(label, min_value=lo, max_value=hi, value=val, step=step, help=desc)

            elif wtype == "log_slider":
                lo = float(cfg.get("min", 1e-6))
                hi = float(cfg.get("max", 1e3))
                val = float(cfg.get("default", math.sqrt(lo * hi) if lo > 0 and hi > 0 else 1.0))
                if lo <= 0 or hi <= 0:
                    st.sidebar.warning(f"{name}: log_slider는 양수 min/max가 필요해서 number로 표시합니다.")
                    params[name] = st.sidebar.number_input(label, value=val, help=desc)
                else:
                    with st.sidebar.expander(label, expanded=True):
                        log_value = st.slider(
                            "log10(value)",
                            min_value=math.log10(lo),
                            max_value=math.log10(hi),
                            value=math.log10(clamp(val, lo, hi)),
                            step=float(cfg.get("log_step", 0.01)),
                            key=f"{name}_log_slider",
                            help=desc,
                        )
                        actual = 10 ** log_value
                        st.caption(f"value = {actual:.6g}")
                        params[name] = float(actual)

            elif wtype in ["slider", "number"]:
                lo = float(cfg.get("min", -10.0))
                hi = float(cfg.get("max", 10.0))
                step = float(cfg.get("step", 0.01))
                val = clamp(float(cfg.get("default", 0.0)), lo, hi)

                if wtype == "slider":
                    params[name] = st.sidebar.slider(label, min_value=lo, max_value=hi, value=val, step=step, help=desc)
                else:
                    params[name] = st.sidebar.number_input(label, min_value=lo, max_value=hi, value=val, step=step, format="%.8f", help=desc)

            else:
                st.sidebar.warning(f"{name}: 알 수 없는 type '{wtype}' → number로 표시합니다.")
                params[name] = st.sidebar.number_input(label, value=float(default), help=desc)

        except Exception as e:
            st.sidebar.error(f"{name} UI 생성 오류: {e}")
            params[name] = default

    return params


def get_figure_config(namespace: Dict[str, Any], default_size: Tuple[float, float], default_dpi: int) -> Dict[str, Any]:
    cfg = namespace.get("FIGURE_CONFIG", {})
    if not isinstance(cfg, dict):
        cfg = {}

    projection = str(cfg.get("projection", "2d")).lower()
    if projection not in ["2d", "3d"]:
        projection = "2d"

    figsize = cfg.get("figsize", default_size)
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        figsize = default_size

    dpi = cfg.get("dpi", default_dpi)
    try:
        dpi = int(dpi)
    except Exception:
        dpi = int(default_dpi)

    return {
        "projection": projection,
        "figsize": (float(figsize[0]), float(figsize[1])),
        "dpi": int(dpi),
        "facecolor": cfg.get("facecolor", "white"),
    }


def create_figure(namespace: Dict[str, Any], default_size: Tuple[float, float], default_dpi: int):
    cfg = get_figure_config(namespace, default_size, default_dpi)
    fig = plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"], facecolor=cfg["facecolor"])
    if cfg["projection"] == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    return fig, ax


def call_init_state(fn, params):
    try:
        return fn(params)
    except TypeError:
        return fn()


def call_update_state(fn, state, params, frame):
    try:
        result = fn(state, params, frame)
    except TypeError:
        try:
            result = fn(state, params)
        except TypeError:
            result = fn(state)
    return state if result is None else result


def call_draw_frame(fn, ax, state, params, frame):
    try:
        return fn(ax, state, params, frame)
    except TypeError:
        try:
            return fn(ax, state, params)
        except TypeError:
            return fn(ax, state)


def run_state_to_frame(namespace: Dict[str, Any], params: Dict[str, Any], frame: int):
    state = call_init_state(namespace["init_state"], params)
    for f in range(frame + 1):
        state = call_update_state(namespace["update_state"], state, params, f)
    return state


def render_single_frame(namespace: Dict[str, Any], params: Dict[str, Any], frame: int, figsize, dpi):
    fig, ax = create_figure(namespace, figsize, dpi)
    state = run_state_to_frame(namespace, params, int(frame))
    call_draw_frame(namespace["draw_frame"], ax, state, params, int(frame))
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def create_animation(namespace: Dict[str, Any], params: Dict[str, Any], nframes: int, fps: int, figsize, dpi):
    fig, ax = create_figure(namespace, figsize, dpi)
    state = call_init_state(namespace["init_state"], params)

    def update(frame):
        nonlocal fig, ax, state
        fig.clear()
        cfg = get_figure_config(namespace, figsize, dpi)
        if cfg["projection"] == "3d":
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        state = call_update_state(namespace["update_state"], state, params, frame)
        call_draw_frame(namespace["draw_frame"], ax, state, params, frame)
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=int(nframes),
        interval=1000 / max(int(fps), 1),
        blit=False,
        repeat=False,
    )
    return ani, fig


def smoke_test(namespace: Dict[str, Any], params: Dict[str, Any], frames: int, figsize, dpi) -> Tuple[bool, str]:
    try:
        state = call_init_state(namespace["init_state"], params)
        for f in range(int(frames)):
            state = call_update_state(namespace["update_state"], state, params, f)
        fig, ax = create_figure(namespace, figsize, dpi)
        call_draw_frame(namespace["draw_frame"], ax, state, params, int(frames))
        plt.close(fig)
        return True, "OK"
    except Exception:
        return False, traceback.format_exc()


def make_prompt(topic: str, visual_goal: str, extra_conditions: str) -> str:
    return (
        "내가 만들고 싶은 시뮬레이션 주제는 [" + topic + "]이야.\n\n"
        "보여주고 싶은 모습:\n" + visual_goal + "\n\n"
        "추가 조건:\n" + extra_conditions + "\n\n"
        "Python으로 파라미터 조절 가능한 애니메이션을 만들고 싶어.\n"
        "나는 네가 작성한 코드를 웹사이트에 붙여넣어서 실행할 거야.\n"
        "어떤 주제가 올지 모르기 때문에, 파라미터와 시각화 방식은 네가 주제에 맞게 직접 정해야 해.\n\n"
        "아래 4개는 반드시 작성해줘.\n\n"
        "1. PARAM_CONFIG\n"
        "2. init_state(params)\n"
        "3. update_state(state, params, frame)\n"
        "4. draw_frame(ax, state, params, frame)\n\n"
        "필요하면 선택적으로 아래도 작성해줘.\n\n"
        "5. FIGURE_CONFIG\n"
        "6. 보조 함수(helper function)\n\n"
        "[PARAM_CONFIG 작성 규칙]\n"
        "- 사용자가 조절하면 좋은 파라미터를 주제에 맞게 직접 선정해줘.\n"
        "- 각 파라미터는 dictionary로 작성해줘.\n"
        "- 가능한 type은 \"slider\", \"number\", \"int\", \"checkbox\", \"color\", \"text\", \"select\", \"log_slider\" 중에서 골라줘.\n"
        "- slider/number/int/log_slider에는 가능한 한 min, max, default, step을 넣어줘.\n"
        "- select에는 options와 default를 넣어줘.\n"
        "- 모든 파라미터에는 description을 넣어줘.\n"
        "- 단위가 있으면 unit도 넣어줘.\n\n"
        "[함수 작성 규칙]\n"
        "- init_state(params)는 초기 상태를 만들고 state dictionary를 return해야 해.\n"
        "- update_state(state, params, frame)는 프레임마다 상태를 업데이트하고 state를 return해야 해.\n"
        "- draw_frame(ax, state, params, frame)는 matplotlib의 ax 객체에 직접 그림을 그려야 해.\n"
        "- draw_frame 안에서 축 이름, 제목, 범례, 색상, grid 등을 주제에 맞게 직접 정해줘.\n"
        "- draw_frame 첫 줄에는 가능하면 ax.clear()를 넣어줘.\n"
        "- 3D 그래프가 아니면 ax.text2D, ax.set_zlabel, ax.view_init 같은 3D 전용 기능을 쓰지 마.\n"
        "- Streamlit 코드, 저장 코드, FuncAnimation 코드는 쓰지 마.\n"
        "- numpy는 np로 이미 import되어 있다고 가정해.\n"
        "- math는 math로 이미 import되어 있다고 가정해.\n"
        "- matplotlib.pyplot은 쓰지 말고 ax 중심으로 그려줘.\n"
        "- 파일 입출력은 하지 마.\n"
        "- 외부 라이브러리는 사용하지 마.\n"
        "- PARAM_CONFIG에 정의하지 않은 params[\"변수명\"]은 사용하지 마.\n"
        "- state에서 사용할 변수는 반드시 init_state에서 먼저 만들어줘.\n"
        "- 필요하면 보조 함수를 만들어도 되지만, 코드블록 하나 안에 모두 포함해줘.\n\n"
        "[FIGURE_CONFIG 작성 규칙]\n"
        "- 2D 그래프면 FIGURE_CONFIG = {\"projection\": \"2d\"}로 설정해줘.\n"
        "- 3D 그래프가 필요하면 FIGURE_CONFIG = {\"projection\": \"3d\"}로 설정해줘.\n"
        "- 예: FIGURE_CONFIG = {\"projection\": \"2d\", \"figsize\": (8, 5.2), \"dpi\": 110}\n\n"
        "[최종 출력 형식]\n"
        "먼저 파라미터를 왜 그렇게 골랐는지 간단히 설명해줘.\n"
        "그 다음 마지막에는 복사하기 쉽게 Python 코드블록 하나로 PARAM_CONFIG, FIGURE_CONFIG, init_state, update_state, draw_frame, 보조 함수를 모두 정리해줘.\n"
    )


# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">🧪 AI Simulation Animator <span style="font-size:1rem;color:#64748b;">{APP_VERSION}</span></div>
  <div class="hero-subtitle">
    주제를 미리 정하지 않는 범용 앱입니다. GPT가 만든 <b>PARAM_CONFIG · init_state · update_state · draw_frame</b>을 붙여넣으면,
    웹에서 파라미터 조절, Preview, GIF 렌더, 다운로드까지 할 수 있습니다.
  </div>
  <span class="pill">🎛️ Auto UI</span>
  <span class="pill">🧩 Helper Functions OK</span>
  <span class="pill">📐 2D / 3D</span>
  <span class="pill">🎬 GIF</span>
  <span class="pill">📥 Download</span>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar Settings
# =========================================================
with st.sidebar:
    st.header("⚙️ Render Settings")

    file_name = sanitize_filename(st.text_input("file name", value="simulation"))

    nframes = st.number_input("GIF frames", min_value=5, max_value=600, value=120, step=5)
    fps = st.number_input("GIF fps", min_value=1, max_value=60, value=20, step=1)
    preview_frame = st.slider(
        "preview frame",
        min_value=0,
        max_value=max(int(nframes) - 1, 0),
        value=min(40, max(int(nframes) - 1, 0)),
        step=1,
    )
    smoke_frames = st.number_input("smoke test frames", min_value=1, max_value=100, value=6, step=1)

    st.divider()
    st.header("🖼️ Default Figure")
    fig_w = st.number_input("default width", min_value=3.0, max_value=16.0, value=9.0, step=0.2)
    fig_h = st.number_input("default height", min_value=2.5, max_value=12.0, value=5.8, step=0.2)
    dpi = st.number_input("default dpi", min_value=60, max_value=240, value=120, step=10)

    st.divider()
    st.header("🛡️ Execution")
    trust_code = st.checkbox("I trust the pasted code", value=True)
    allow_imports = st.checkbox("Allow import lines", value=False)

default_figsize = (float(fig_w), float(fig_h))
default_dpi = int(dpi)


# =========================================================
# Tabs
# =========================================================
tab_prompt, tab_code, tab_params, tab_preview, tab_render, tab_help = st.tabs(
    ["① Prompt", "② Code Lab", "③ Parameters", "④ Preview", "⑤ Render / Download", "⑥ Help"]
)


# =========================================================
# Prompt Tab
# =========================================================
with tab_prompt:
    st.subheader("① GPT에게 주제 기반 코드 요청하기")
    st.markdown(
        """
<div class="card">
원하는 주제와 보여주고 싶은 모습을 적으면, GPT에게 보낼 프롬프트가 생성됩니다.
이 프롬프트의 핵심은 <b>파라미터와 그래프 방식을 GPT가 주제에 맞게 직접 정하게 하는 것</b>입니다.
</div>
""",
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "만들고 싶은 시뮬레이션 주제",
        value="VO2에서 c_R strain에 따른 Apical/Equatorial V-O length와 orbital occupancy 변화",
    )
    visual_goal = st.text_area(
        "보여주고 싶은 모습",
        value="c_R 축 strain이 변할 때 Apical V-O length, Equatorial V-O length, d_parallel / pi_star orbital occupancy가 함께 변하는 그래프를 보고 싶어.",
        height=95,
    )
    extra_conditions = st.text_area(
        "추가 조건",
        value="파라미터를 조절하면 그래프 모양이 확실히 달라져야 해. 발표용이므로 축 이름, 범례, 제목을 깔끔하게 넣어줘.",
        height=90,
    )

    prompt = make_prompt(topic, visual_goal, extra_conditions)
    st.session_state.generated_prompt_text = prompt
    st.code(prompt, language="text")
    st.download_button(
        "📥 Download Prompt TXT",
        data=prompt.encode("utf-8"),
        file_name=bytes_name(file_name + "_prompt", "txt"),
        mime="text/plain",
        use_container_width=True,
    )


# =========================================================
# Code Lab Tab
# =========================================================
with tab_code:
    st.subheader("② GPT 답변 또는 Python 코드 붙여넣기")
    st.markdown(
        """
<div class="card">
GPT 답변 전체를 붙여넣어도 됩니다. 앱이 코드블록을 자동으로 찾습니다.
보조 함수가 있어도 같은 namespace에서 실행되도록 설계했습니다.
</div>
""",
        unsafe_allow_html=True,
    )

    raw_code_input = st.text_area(
        "GPT answer / Python code",
        value=st.session_state.raw_code_input,
        height=470,
        key="code_area",
    )
    st.session_state.raw_code_input = raw_code_input

    extracted_code = extract_code_from_text(raw_code_input)
    displayed_code = extracted_code if allow_imports else strip_imports(extracted_code)

    with st.expander("실제로 실행될 코드 확인", expanded=False):
        st.code(displayed_code, language="python")

    col1, col2, col3 = st.columns(3)
    with col1:
        validate_clicked = st.button("🔎 Validate Code", use_container_width=True)
    with col2:
        st.download_button(
            "📥 Download Current Code",
            data=displayed_code.encode("utf-8"),
            file_name=bytes_name(file_name + "_code", "py"),
            mime="text/x-python",
            use_container_width=True,
        )
    with col3:
        if st.button("🧹 Reset Example", use_container_width=True):
            st.session_state.raw_code_input = DEFAULT_CODE
            st.rerun()


# =========================================================
# Execute user code
# =========================================================
namespace = None
clean_code = ""
ok = False
problems = []
exec_error = None

if trust_code and st.session_state.raw_code_input.strip():
    try:
        namespace, clean_code = execute_user_code(st.session_state.raw_code_input, allow_imports=allow_imports)
        ok, problems = validate_namespace(namespace)
        if not ok:
            st.session_state.last_exec_error = "\n".join(problems)
    except Exception:
        exec_error = traceback.format_exc()
        st.session_state.last_exec_error = exec_error
else:
    problems = ["코드를 실행하려면 sidebar에서 'I trust the pasted code'를 체크하세요."]

if validate_clicked:
    if exec_error:
        st.error("코드 실행 오류")
        st.code(exec_error, language="text")
    elif not ok:
        st.error("필수 구성요소가 부족합니다.")
        st.write(problems)
    else:
        st.success("코드 구조 인식 완료")


# =========================================================
# Build parameter UI
# =========================================================
param_config = {}
params = {}
if ok and namespace:
    param_config = normalize_param_config(namespace.get("PARAM_CONFIG", {}))

with st.sidebar:
    st.divider()
    st.header("🎛️ GPT Parameters")
    params = build_parameter_ui(param_config)


# =========================================================
# Parameters Tab
# =========================================================
with tab_params:
    st.subheader("③ 자동 인식된 파라미터")

    if exec_error:
        st.error("코드 실행 오류")
        st.code(exec_error, language="text")
    elif not ok:
        st.warning("아직 유효한 코드가 아닙니다.")
        if problems:
            st.write(problems)
    else:
        st.markdown('<div class="card"><b class="good">PARAM_CONFIG 인식 완료</b> — 왼쪽 사이드바에서 파라미터를 조절하세요.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Current parameter values")
            st.json(json_safe(params))
        with col2:
            st.markdown("#### PARAM_CONFIG")
            st.json(json_safe(param_config))

        st.download_button(
            "📥 Download Parameters JSON",
            data=json.dumps(json_safe(params), ensure_ascii=False, indent=4).encode("utf-8"),
            file_name=bytes_name(file_name + "_params", "json"),
            mime="application/json",
            use_container_width=True,
        )


# =========================================================
# Preview Tab
# =========================================================
with tab_preview:
    st.subheader("④ Preview")

    if exec_error:
        st.error("코드 실행 오류")
        st.code(exec_error, language="text")
    elif not ok or not namespace:
        st.warning("먼저 Code Lab에 유효한 코드를 붙여넣어 주세요.")
        if problems:
            st.write(problems)
    else:
        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.markdown("#### Static preview")
            if st.button("🔄 Render Preview Frame", use_container_width=True):
                try:
                    fig = render_single_frame(namespace, params, int(preview_frame), default_figsize, default_dpi)
                    st.pyplot(fig, clear_figure=True)
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format="png", dpi=default_dpi, bbox_inches="tight")
                    plt.close(fig)
                    png_buffer.seek(0)
                    st.session_state.generated_png_bytes = png_buffer.getvalue()
                    st.session_state.generated_png_name = bytes_name(file_name + "_preview", "png")
                    st.success("Preview 렌더링 완료")
                except Exception:
                    st.error("Preview 렌더링 오류")
                    st.code(traceback.format_exc(), language="text")

            if st.session_state.generated_png_bytes:
                st.download_button(
                    "📥 Download Last Preview PNG",
                    data=st.session_state.generated_png_bytes,
                    file_name=st.session_state.generated_png_name or "preview.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with col2:
            st.markdown("#### Smoke test")
            if st.button("🧪 Run Smoke Test", use_container_width=True):
                passed, msg = smoke_test(namespace, params, int(smoke_frames), default_figsize, default_dpi)
                if passed:
                    st.success("Smoke test 통과")
                else:
                    st.error("Smoke test 실패")
                    st.code(msg, language="text")

            st.info(
                """
오류가 나면 대부분 아래 중 하나입니다.

1. PARAM_CONFIG에 없는 params["변수명"] 사용  
2. init_state에서 만들지 않은 state["변수명"] 사용  
3. 2D/3D projection 불일치  
4. draw_frame에서 ax 대신 plt/fig 직접 사용  
5. 코드블록 바깥으로 보조 함수가 빠짐  
"""
            )


# =========================================================
# Render / Download Tab
# =========================================================
with tab_render:
    st.subheader("⑤ Render & Download")

    st.markdown(
        """
<div class="card">
웹사이트 배포 환경에서는 서버 내부 경로에 저장해도 사용자가 찾기 어렵습니다.
그래서 이 탭은 <b>Render → Download</b> 방식으로 동작합니다.
</div>
""",
        unsafe_allow_html=True,
    )

    if exec_error:
        st.error("코드 실행 오류")
        st.code(exec_error, language="text")
    elif not ok or not namespace:
        st.warning("먼저 유효한 코드를 붙여넣어 주세요.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎬 Render GIF", use_container_width=True):
                try:
                    with st.spinner("GIF 생성 중입니다. 프레임 수가 많으면 시간이 걸릴 수 있습니다."):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            gif_name = bytes_name(file_name, "gif")
                            gif_path = Path(tmpdir) / gif_name
                            ani, fig = create_animation(namespace, params, int(nframes), int(fps), default_figsize, default_dpi)
                            ani.save(gif_path, writer=PillowWriter(fps=int(fps)))
                            plt.close(fig)

                            st.session_state.generated_gif_bytes = gif_path.read_bytes()
                            st.session_state.generated_gif_name = gif_name

                    st.success("GIF 생성 완료")
                except Exception:
                    st.error("GIF 생성 오류")
                    st.code(traceback.format_exc(), language="text")

            if st.session_state.generated_gif_bytes:
                st.download_button(
                    "📥 Download GIF",
                    data=st.session_state.generated_gif_bytes,
                    file_name=st.session_state.generated_gif_name or "animation.gif",
                    mime="image/gif",
                    use_container_width=True,
                )
                st.image(st.session_state.generated_gif_bytes, caption="Last rendered GIF")

        with col2:
            if st.button("🖼️ Render PNG Preview", use_container_width=True):
                try:
                    fig = render_single_frame(namespace, params, int(preview_frame), default_figsize, default_dpi)
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format="png", dpi=default_dpi, bbox_inches="tight")
                    plt.close(fig)
                    png_buffer.seek(0)
                    st.session_state.generated_png_bytes = png_buffer.getvalue()
                    st.session_state.generated_png_name = bytes_name(file_name + "_preview", "png")
                    st.success("PNG 생성 완료")
                except Exception:
                    st.error("PNG 생성 오류")
                    st.code(traceback.format_exc(), language="text")

            if st.session_state.generated_png_bytes:
                st.download_button(
                    "📥 Download PNG",
                    data=st.session_state.generated_png_bytes,
                    file_name=st.session_state.generated_png_name or "preview.png",
                    mime="image/png",
                    use_container_width=True,
                )

        st.divider()
        col3, col4 = st.columns(2)

        with col3:
            st.download_button(
                "🐍 Download Python Code",
                data=(clean_code or "").encode("utf-8"),
                file_name=bytes_name(file_name + "_code", "py"),
                mime="text/x-python",
                use_container_width=True,
            )

        with col4:
            st.download_button(
                "🧾 Download Parameters JSON",
                data=json.dumps(json_safe(params), ensure_ascii=False, indent=4).encode("utf-8"),
                file_name=bytes_name(file_name + "_params", "json"),
                mime="application/json",
                use_container_width=True,
            )


# =========================================================
# Help Tab
# =========================================================
with tab_help:
    st.subheader("⑥ 사용 설명")

    st.markdown(
        """
<div class="card">
<h4>✅ 핵심 구조</h4>
<p>이 앱은 주제를 미리 알지 않습니다. GPT가 주제에 맞게 아래 4개를 만들어오면, 앱은 그대로 실행합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.code(
        """
PARAM_CONFIG = {...}

FIGURE_CONFIG = {"projection": "2d", "figsize": (8, 5), "dpi": 120}

def init_state(params):
    ...
    return state

def update_state(state, params, frame):
    ...
    return state

def draw_frame(ax, state, params, frame):
    ax.clear()
    ...
""",
        language="python",
    )

    st.markdown(
        """
### 자주 나는 오류와 해결

**1. `NameError: helper_function is not defined`**  
이번 버전은 `exec(code, namespace, namespace)`로 고쳐서, 보조 함수가 같은 코드블록 안에 있으면 정상 인식됩니다.

**2. `KeyError: 'xxx'`**  
`params["xxx"]` 또는 `state["xxx"]`를 쓰는데, 각각 `PARAM_CONFIG` 또는 `init_state`에 없을 때 납니다.

**3. 3D 코드인데 2D 축 오류**  
`FIGURE_CONFIG = {"projection": "3d"}`를 넣어야 합니다.

**4. 웹사이트에서 파일 경로를 못 찾음**  
배포 환경에서는 서버 내부 경로가 보입니다. 이 앱은 `Download GIF/PNG` 방식으로 받도록 되어 있습니다.

**5. 그래프가 겹침**  
이번 버전은 매 프레임 `fig.clear()` 후 새 축을 만들어 twin axis가 쌓이는 문제를 줄였습니다.

---

### requirements.txt
"""
    )

    st.code(
        """
streamlit
numpy
matplotlib
pillow
""",
        language="text",
    )

    st.warning("주의: 이 앱은 붙여넣은 Python 코드를 실행합니다. 공개 배포 시에는 신뢰할 수 있는 사람만 사용하게 하세요.")
