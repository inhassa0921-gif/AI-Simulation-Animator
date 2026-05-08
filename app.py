import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import traceback
import inspect
import shutil
import json
import math
import re

# =========================================================
# 0. Page config must be called before any other Streamlit command
# =========================================================
st.set_page_config(
    page_title="AI Simulation Animator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 1. Pretty UI
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #f8f5ff 45%, #fffaf2 100%);
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(120, 120, 170, 0.18);
        box-shadow: 0 18px 45px rgba(60, 70, 110, 0.10);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 0.35rem;
        color: #1f2a44;
    }
    .hero-subtitle {
        color: #4b587c;
        font-size: 1.03rem;
        line-height: 1.55;
    }
    .mini-card {
        padding: 1rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(120, 120, 170, 0.14);
        box-shadow: 0 10px 25px rgba(60, 70, 110, 0.07);
        min-height: 112px;
    }
    .mini-title {
        font-weight: 800;
        color: #25304f;
        margin-bottom: 0.25rem;
        font-size: 1.04rem;
    }
    .mini-text {
        color: #5b6687;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .ok-pill {
        display: inline-block;
        padding: 0.24rem 0.62rem;
        border-radius: 999px;
        background: #e7f8ef;
        color: #176b3a;
        font-weight: 750;
        font-size: 0.86rem;
        border: 1px solid #b9efd0;
    }
    .warn-pill {
        display: inline-block;
        padding: 0.24rem 0.62rem;
        border-radius: 999px;
        background: #fff1d6;
        color: #8a5a00;
        font-weight: 750;
        font-size: 0.86rem;
        border: 1px solid #ffd98c;
    }
    .bad-pill {
        display: inline-block;
        padding: 0.24rem 0.62rem;
        border-radius: 999px;
        background: #ffe6e6;
        color: #8d1d1d;
        font-weight: 750;
        font-size: 0.86rem;
        border: 1px solid #ffc2c2;
    }
    div[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.78);
        border-right: 1px solid rgba(120, 120, 170, 0.14);
    }
    .small-note {
        color: #6f7792;
        font-size: 0.9rem;
        line-height: 1.45;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-card">
  <div class="hero-title">🧪 AI Simulation Animator</div>
  <div class="hero-subtitle">
    어떤 주제든 GPT가 <b>파라미터 · 초기 상태 · 시간 변화 · 그리는 방식</b>을 코드로 만들고,<br>
    이 웹사이트는 그 코드를 받아서 <b>파라미터 조절형 애니메이션</b>으로 실행하고 저장합니다.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="mini-card"><div class="mini-title">① 주제 입력 📝</div><div class="mini-text">예: PN junction, 리소그래피, 오비탈, 확산, 열전달 등</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="mini-card"><div class="mini-title">② GPT 코드 생성 🤖</div><div class="mini-text">GPT가 PARAM_CONFIG, init, update, draw 함수를 작성</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="mini-card"><div class="mini-title">③ 파라미터 조절 🎛️</div><div class="mini-text">코드가 정한 파라미터를 자동 UI로 생성</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="mini-card"><div class="mini-title">④ 저장 💾</div><div class="mini-text">GIF, PNG, 코드, 파라미터, 프롬프트를 Dropbox에 저장</div></div>', unsafe_allow_html=True)

# =========================================================
# 2. Default directories
# =========================================================
def default_save_root() -> Path:
    candidates = [
        Path.home() / "Dropbox" / "AI_Simulations",
        Path.home() / "OneDrive" / "AI_Simulations",
        Path.home() / "Desktop" / "AI_Simulations",
    ]
    for p in candidates:
        if p.parent.exists():
            return p
    return Path.cwd() / "AI_Simulations"

# =========================================================
# 3. Utility functions
# =========================================================
def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "simulation"


def make_dirs(root: Path) -> dict:
    paths = {
        "root": root,
        "gif": root / "GIF",
        "png": root / "PNG",
        "code": root / "CODE",
        "params": root / "PARAMS",
        "prompts": root / "PROMPTS",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def extract_first_python_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        for block in blocks:
            if "PARAM_CONFIG" in block and "def init_state" in block:
                return block.strip()
        for block in blocks:
            if "def draw_frame" in block:
                return block.strip()
        return blocks[-1].strip()
    return text.strip()


def extract_param_names_from_code(code: str) -> list:
    names = re.findall(r'params\["(.*?)"\]', code)
    names += re.findall(r"params\['(.*?)'\]", code)
    return sorted(set(names))


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed_roots = {"numpy", "math", "matplotlib"}
    root = name.split(".")[0]
    if root in allowed_roots:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{name}' is blocked in this runner. Use np, math, and ax-based matplotlib only.")


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "print": print,
    "Exception": Exception,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "__import__": safe_import,
}


def execute_user_code(code: str) -> tuple:
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "math": math,
        "plt": plt,
    }
    try:
        compiled = compile(code, "<GPT_SIMULATION_CODE>", "exec")
        exec(compiled, namespace)
        return True, namespace, None
    except Exception:
        return False, namespace, traceback.format_exc()


def normalize_param_config(namespace: dict, code: str) -> dict:
    config = namespace.get("PARAM_CONFIG", {})
    if not isinstance(config, dict):
        config = {}

    # Fallback: if GPT forgot PARAM_CONFIG, create rough controls from params["..."] usage.
    detected = extract_param_names_from_code(code)
    for name in detected:
        if name not in config:
            config[name] = guess_config_from_name(name)
    return config


def guess_config_from_name(name: str) -> dict:
    lower = name.lower()
    if "color" in lower or "colour" in lower:
        return {"type": "color", "default": "#4F7CFF", "description": "Auto-detected color parameter"}
    if any(k in lower for k in ["use", "show", "enable", "enabled", "boundary", "clip", "normalize", "fixed", "loop"]):
        return {"type": "checkbox", "default": True, "description": "Auto-detected boolean parameter"}
    if any(k in lower for k in ["name", "label", "title", "material", "mode", "type"]):
        return {"type": "text", "default": name, "description": "Auto-detected text parameter"}
    if any(k in lower for k in ["num", "count", "number", "seed", "order", "index", "steps"]):
        return {"type": "int", "min": 0, "max": 10000, "default": 10, "step": 1, "description": "Auto-detected integer parameter"}

    default, min_value, max_value, step = default_range_for(name)
    return {
        "type": "slider",
        "min": min_value,
        "max": max_value,
        "default": default,
        "step": step,
        "description": "Auto-detected numeric parameter. Better results if GPT provides PARAM_CONFIG.",
    }


def default_range_for(name: str) -> tuple:
    lower = name.lower()
    rules = [
        (["temp", "temperature"], (300.0, 0.0, 1200.0, 10.0)),
        (["time", "duration"], (1.0, 0.0, 100.0, 0.1)),
        (["dose"], (50.0, 0.0, 500.0, 1.0)),
        (["intensity", "power"], (1.0, 0.0, 100.0, 0.1)),
        (["speed", "velocity"], (0.03, 0.0, 2.0, 0.001)),
        (["rate"], (0.01, 0.0, 2.0, 0.001)),
        (["field", "voltage", "bias"], (0.0, -10.0, 10.0, 0.1)),
        (["radius"], (0.25, 0.0, 2.0, 0.01)),
        (["width", "length", "height", "depth", "thickness"], (0.2, 0.0, 2.0, 0.01)),
        (["blur", "noise", "roughness"], (0.01, 0.0, 0.5, 0.001)),
        (["amplitude"], (0.1, 0.0, 2.0, 0.01)),
        (["freq", "frequency"], (1.0, 0.0, 100.0, 0.1)),
        (["angle", "phase"], (0.0, -180.0, 180.0, 1.0)),
        (["angular"], (0.05, -2.0, 2.0, 0.001)),
        (["diffusion"], (0.01, 0.0, 2.0, 0.001)),
        (["concentration", "density", "doping"], (1.0, 0.0, 100.0, 0.1)),
        (["potential", "energy"], (1.0, -10.0, 10.0, 0.1)),
    ]
    for keys, values in rules:
        if any(k in lower for k in keys):
            return values
    return 1.0, -10.0, 10.0, 0.01


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def get_signature_len(func) -> int:
    try:
        return len(inspect.signature(func).parameters)
    except Exception:
        return 0


def call_init_state(init_state_func, params):
    argc = get_signature_len(init_state_func)
    if argc >= 2:
        return init_state_func(params, np.random.default_rng(int(params.get("runner_seed", 0))))
    return init_state_func(params)


def call_update_state(update_state_func, state, params, frame):
    argc = get_signature_len(update_state_func)
    if argc >= 3:
        result = update_state_func(state, params, frame)
    else:
        result = update_state_func(state, params)
    return state if result is None else result


def call_draw_frame(draw_frame_func, fig, ax, state, params, frame):
    argc = get_signature_len(draw_frame_func)
    if argc >= 5:
        return draw_frame_func(fig, ax, state, params, frame)
    return draw_frame_func(ax, state, params, frame)


def create_figure(namespace: dict, params: dict, runner: dict):
    fig_config = namespace.get("FIGURE_CONFIG", {})
    if not isinstance(fig_config, dict):
        fig_config = {}

    projection_choice = runner.get("projection", "auto")
    projection = fig_config.get("projection", "2d") if projection_choice == "auto" else projection_choice
    projection = str(projection).lower()

    figsize = fig_config.get("figsize", (runner.get("fig_width", 8.0), runner.get("fig_height", 5.2)))
    dpi = int(runner.get("dpi", fig_config.get("dpi", 110)))

    fig = plt.figure(figsize=figsize, dpi=dpi)

    if projection in ["3d", "three_d", "3-d"]:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    return fig, ax


def render_frame(namespace: dict, params: dict, runner: dict, state: dict, frame: int):
    draw_frame_func = namespace["draw_frame"]
    fig, ax = create_figure(namespace, params, runner)
    ax.cla()
    call_draw_frame(draw_frame_func, fig, ax, state, params, frame)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def create_animation(namespace: dict, params: dict, runner: dict):
    init_state_func = namespace["init_state"]
    update_state_func = namespace["update_state"]
    draw_frame_func = namespace["draw_frame"]

    state = call_init_state(init_state_func, params)
    fig, ax = create_figure(namespace, params, runner)

    def update(frame):
        nonlocal state
        ax.cla()
        state = call_update_state(update_state_func, state, params, frame)
        call_draw_frame(draw_frame_func, fig, ax, state, params, frame)
        return ax,

    ani = FuncAnimation(
        fig,
        update,
        frames=int(runner["num_frames"]),
        interval=1000 / max(int(runner["fps"]), 1),
        blit=False,
    )
    return ani, fig


def validate_namespace(namespace: dict) -> list:
    errors = []
    if "PARAM_CONFIG" not in namespace:
        errors.append("PARAM_CONFIG가 없습니다. 그래도 params 사용값을 자동 추출해 임시 UI를 만들 수는 있습니다.")
    elif not isinstance(namespace.get("PARAM_CONFIG"), dict):
        errors.append("PARAM_CONFIG는 dict 형식이어야 합니다.")

    for func_name in ["init_state", "update_state", "draw_frame"]:
        if func_name not in namespace or not callable(namespace.get(func_name)):
            errors.append(f"필수 함수 `{func_name}`가 없습니다.")
    return errors


def run_smoke_test(namespace: dict, params: dict, runner: dict):
    init_state_func = namespace["init_state"]
    update_state_func = namespace["update_state"]
    draw_frame_func = namespace["draw_frame"]

    state = call_init_state(init_state_func, params)
    for frame in range(3):
        state = call_update_state(update_state_func, state, params, frame)
    fig, ax = create_figure(namespace, params, runner)
    call_draw_frame(draw_frame_func, fig, ax, state, params, 3)
    plt.close(fig)
    return True

# =========================================================
# 4. Sidebar: runner settings
# =========================================================
st.sidebar.markdown("## 🧭 Runner Settings")

save_root_text = st.sidebar.text_input("Save root folder", value=str(default_save_root()))
save_root = Path(save_root_text).expanduser()
paths = make_dirs(save_root)

file_name = sanitize_filename(st.sidebar.text_input("Base file name", value="simulation"))

num_frames = st.sidebar.number_input("Animation frames", min_value=5, max_value=3000, value=120, step=5)
fps = st.sidebar.number_input("FPS", min_value=1, max_value=60, value=20, step=1)
preview_steps = st.sidebar.number_input("Preview update steps", min_value=0, max_value=1000, value=30, step=5)

projection = st.sidebar.selectbox("Figure projection", ["auto", "2d", "3d"], index=0)
fig_width = st.sidebar.number_input("Figure width", min_value=3.0, max_value=16.0, value=8.0, step=0.5)
fig_height = st.sidebar.number_input("Figure height", min_value=3.0, max_value=12.0, value=5.2, step=0.5)
dpi = st.sidebar.number_input("DPI", min_value=60, max_value=240, value=110, step=10)
runner_seed = st.sidebar.number_input("Runner seed", min_value=0, max_value=999999, value=0, step=1)

runner = {
    "num_frames": int(num_frames),
    "fps": int(fps),
    "preview_steps": int(preview_steps),
    "projection": projection,
    "fig_width": float(fig_width),
    "fig_height": float(fig_height),
    "dpi": int(dpi),
    "runner_seed": int(runner_seed),
}

# =========================================================
# 5. Default demo code
# =========================================================
DEFAULT_CODE = r'''
PARAM_CONFIG = {
    "particle_count": {
        "type": "int",
        "min": 20,
        "max": 800,
        "default": 180,
        "step": 10,
        "description": "Number of moving particles"
    },
    "motion_strength": {
        "type": "slider",
        "min": 0.001,
        "max": 0.08,
        "default": 0.018,
        "step": 0.001,
        "description": "Random motion per frame"
    },
    "drift_strength": {
        "type": "slider",
        "min": -0.03,
        "max": 0.03,
        "default": 0.004,
        "step": 0.001,
        "description": "Directional drift along x-axis"
    },
    "particle_color": {
        "type": "color",
        "default": "#4F7CFF",
        "description": "Particle color"
    },
    "show_trail": {
        "type": "checkbox",
        "default": True,
        "description": "Show faint initial positions as a reference"
    }
}

FIGURE_CONFIG = {
    "projection": "2d",
    "figsize": (8, 5.2),
    "dpi": 110
}

def init_state(params):
    n = int(params["particle_count"])
    rng = np.random.default_rng(0)
    x = rng.random(n)
    y = rng.random(n)
    return {
        "x": x,
        "y": y,
        "x0": x.copy(),
        "y0": y.copy(),
        "frame": 0
    }

def update_state(state, params, frame):
    motion = params["motion_strength"]
    drift = params["drift_strength"]
    state["frame"] += 1
    state["x"] += drift + motion * np.random.randn(len(state["x"]))
    state["y"] += motion * np.random.randn(len(state["y"]))
    state["x"] = np.mod(state["x"], 1.0)
    state["y"] = np.clip(state["y"], 0.0, 1.0)
    return state

def draw_frame(ax, state, params, frame):
    if params["show_trail"]:
        ax.scatter(state["x0"], state["y0"], s=8, alpha=0.13, color="gray", label="initial")
    ax.scatter(state["x"], state["y"], s=22, alpha=0.82, color=params["particle_color"], label="current")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized x-position")
    ax.set_ylabel("Normalized y-position")
    ax.set_title(f"Generic GPT-generated particle demo | Frame {frame}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
'''

# =========================================================
# 6. Main tabs
# =========================================================
tab_prompt, tab_code, tab_params, tab_preview, tab_save, tab_help = st.tabs(
    ["① GPT Prompt", "② Code Lab", "③ Parameters", "④ Preview", "⑤ Save", "⑥ Help"]
)

# =========================================================
# 7. Prompt tab
# =========================================================
with tab_prompt:
    st.subheader("① GPT에게 주제 기반 코드 요청하기")
    st.markdown(
        """
<div class="small-note">
여기서 주제와 보여주고 싶은 모습을 적으면, GPT에게 복사해 보낼 프롬프트가 만들어집니다.  
핵심은 <b>파라미터와 그래프 방식까지 GPT가 주제에 맞게 직접 정하게 하는 것</b>입니다.
</div>
""",
        unsafe_allow_html=True,
    )

    topic = st.text_input("만들고 싶은 시뮬레이션 주제", value="반도체 PN junction에서 페르미 준위와 밴드 밴딩 변화")
    visual_goal = st.text_area(
        "보여주고 싶은 모습",
        value="p형과 n형이 접합되면서 페르미 준위가 열평형에서 평탄해지고, Ec/Ei/Ev 밴드가 접합부 근처에서 휘어지는 모습을 자세히 보여주고 싶어.",
        height=105,
    )
    extra_requirements = st.text_area(
        "추가 조건",
        value="파라미터를 조절하면 그래프 모양이 확실히 달라져야 해. 발표용이므로 축 이름, 범례, 제목을 깔끔하게 넣어줘.",
        height=90,
    )

    generated_prompt = f"""
내가 만들고 싶은 시뮬레이션 주제는 [{topic}]이야.

보여주고 싶은 모습:
{visual_goal}

추가 조건:
{extra_requirements}

Python으로 파라미터 조절 가능한 애니메이션을 만들고 싶어.
나는 네가 작성한 코드를 웹사이트에 붙여넣어서 실행할 거야.
어떤 주제가 올지 모르기 때문에, 파라미터와 시각화 방식은 네가 주제에 맞게 직접 정해야 해.

아래 4개는 반드시 작성해줘.

1. PARAM_CONFIG
2. init_state(params)
3. update_state(state, params, frame)
4. draw_frame(ax, state, params, frame)

필요하면 선택적으로 아래도 작성해줘.

5. FIGURE_CONFIG

[PARAM_CONFIG 작성 규칙]
- 사용자가 조절하면 좋은 파라미터를 주제에 맞게 직접 선정해줘.
- 각 파라미터는 dictionary로 작성해줘.
- 가능한 type은 "slider", "number", "int", "checkbox", "color", "text", "select", "log_slider" 중에서 골라줘.
- slider/number/int/log_slider에는 가능한 한 min, max, default, step을 넣어줘.
- select에는 options와 default를 넣어줘.
- 모든 파라미터에는 description을 넣어줘.
- 단위가 있으면 unit도 넣어줘.

[함수 작성 규칙]
- init_state(params)는 초기 상태를 만들고 state dictionary를 return해야 해.
- update_state(state, params, frame)는 프레임마다 상태를 업데이트하고 state를 return해야 해.
- draw_frame(ax, state, params, frame)는 matplotlib의 ax 객체에 직접 그림을 그려야 해.
- draw_frame 안에서 축 이름, 제목, 범례, 색상, grid 등을 주제에 맞게 직접 정해줘.
- Streamlit 코드, 저장 코드, FuncAnimation 코드는 쓰지 마.
- numpy는 np로 이미 import되어 있다고 가정해.
- matplotlib.pyplot은 쓰지 말고 ax 중심으로 그려줘.
- 파일 입출력은 하지 마.
- 외부 라이브러리는 사용하지 마.

[FIGURE_CONFIG 작성 규칙]
- 2D 그래프면 {{"projection": "2d"}}로 설정해줘.
- 3D 그래프가 필요하면 {{"projection": "3d"}}로 설정해줘.
- 예: FIGURE_CONFIG = {{"projection": "2d", "figsize": (8, 5.2), "dpi": 110}}

[최종 출력 형식]
먼저 파라미터를 왜 그렇게 골랐는지 간단히 설명해줘.
그 다음 마지막에는 복사하기 쉽게 Python 코드블록 하나로 PARAM_CONFIG, FIGURE_CONFIG, init_state, update_state, draw_frame을 모두 정리해줘.
""".strip()

    st.code(generated_prompt, language="text")

    if st.button("💾 Save this prompt", use_container_width=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_path = paths["prompts"] / f"{file_name}_{timestamp}_prompt.txt"
        prompt_path.write_text(generated_prompt, encoding="utf-8")
        st.success("프롬프트 저장 완료")
        st.write(prompt_path)

# =========================================================
# 8. Code tab
# =========================================================
with tab_code:
    st.subheader("② GPT 코드 붙여넣기")
    st.markdown(
        """
GPT 답변 전체를 붙여넣어도 됩니다. 코드블록이 있으면 자동으로 Python 코드 부분만 추출합니다.  
필수 구성: `PARAM_CONFIG`, `init_state`, `update_state`, `draw_frame`
"""
    )

    raw_code_input = st.text_area(
        "Paste GPT answer or Python code here",
        value=DEFAULT_CODE,
        height=560,
    )

    user_code = extract_first_python_code_block(raw_code_input)

    with st.expander("🔎 Extracted executable code", expanded=False):
        st.code(user_code, language="python")

# Ensure variables exist for first run
if "raw_code_input" not in locals():
    raw_code_input = DEFAULT_CODE
user_code = extract_first_python_code_block(raw_code_input)

ok, namespace, exec_error = execute_user_code(user_code)
validation_errors = validate_namespace(namespace) if ok else ["코드 실행 단계에서 오류가 발생했습니다."]
param_config = normalize_param_config(namespace, user_code) if ok else {}

# =========================================================
# 9. Sidebar parameter widgets
# =========================================================
st.sidebar.markdown("## 🎛️ GPT Parameters")

params = {}

def render_widget(name: str, cfg: dict):
    cfg = cfg or {}
    ptype = str(cfg.get("type", "slider")).lower()
    label = cfg.get("label", name)
    unit = cfg.get("unit", "")
    if unit:
        label = f"{label} ({unit})"
    help_text = cfg.get("description", cfg.get("help", ""))
    default = cfg.get("default", None)

    if ptype in ["bool", "boolean", "checkbox"]:
        return st.checkbox(label, value=bool(default if default is not None else True), help=help_text, key=f"param_{name}")

    if ptype in ["color", "colour"]:
        return st.color_picker(label, value=str(default or "#4F7CFF"), help=help_text, key=f"param_{name}")

    if ptype in ["text", "str", "string"]:
        return st.text_input(label, value=str(default if default is not None else ""), help=help_text, key=f"param_{name}")

    if ptype in ["select", "selectbox", "dropdown", "radio"]:
        options = cfg.get("options", [])
        if not options:
            options = [default if default is not None else "option_1"]
        default_value = default if default in options else options[0]
        index = options.index(default_value)
        return st.selectbox(label, options=options, index=index, help=help_text, key=f"param_{name}")

    if ptype in ["int", "integer"]:
        min_value = int(cfg.get("min", 0))
        max_value = int(cfg.get("max", 1000000))
        value = int(default if default is not None else min_value)
        value = max(min_value, min(max_value, value))
        step = int(cfg.get("step", 1)) or 1
        return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help_text, key=f"param_{name}")

    if ptype in ["log", "log_slider", "logslider"]:
        min_value = float(cfg.get("min", 1e-6))
        max_value = float(cfg.get("max", 1e6))
        min_value = max(min_value, 1e-12)
        max_value = max(max_value, min_value * 10)
        value = float(default if default is not None else math.sqrt(min_value * max_value))
        value = max(min_value, min(max_value, value))
        log_min = math.log10(min_value)
        log_max = math.log10(max_value)
        log_default = math.log10(value)
        log_step = float(cfg.get("log_step", 0.05))
        chosen = st.slider(label, min_value=log_min, max_value=log_max, value=log_default, step=log_step, help=help_text, key=f"param_{name}")
        value_now = 10 ** chosen
        st.caption(f"{name} = {value_now:.4g}")
        return value_now

    # number or slider fallback
    if default is None:
        default, min_value, max_value, step = default_range_for(name)
    else:
        min_value = float(cfg.get("min", default_range_for(name)[1]))
        max_value = float(cfg.get("max", default_range_for(name)[2]))
        step = float(cfg.get("step", default_range_for(name)[3]))
        default = float(default)

    if max_value <= min_value:
        max_value = min_value + abs(step or 1.0) * 100
    default = max(min_value, min(max_value, default))
    step = abs(step) if step != 0 else 0.01

    if ptype in ["number", "float", "numeric", "input"]:
        return st.number_input(label, value=float(default), step=float(step), format="%.8f", help=help_text, key=f"param_{name}")

    return st.slider(label, min_value=float(min_value), max_value=float(max_value), value=float(default), step=float(step), help=help_text, key=f"param_{name}")

if not ok:
    st.sidebar.error("코드 실행 오류가 있어 파라미터를 만들 수 없습니다.")
elif not param_config:
    st.sidebar.info("PARAM_CONFIG 또는 params 사용값이 감지되면 여기에 조절창이 생깁니다.")
else:
    grouped = defaultdict(list)
    for pname, pcfg in param_config.items():
        group = "Parameters"
        if isinstance(pcfg, dict):
            group = pcfg.get("group", pcfg.get("section", "Parameters"))
        grouped[group].append((pname, pcfg))

    for group, items in grouped.items():
        with st.sidebar.expander(f"✨ {group}", expanded=True):
            for pname, pcfg in items:
                params[pname] = render_widget(pname, pcfg if isinstance(pcfg, dict) else {"default": pcfg})

# Add runner metadata into params so GPT code can optionally read it, but keep names unlikely to clash.
params["_runner_frames"] = int(num_frames)
params["_runner_fps"] = int(fps)
params["_runner_seed"] = int(runner_seed)

# =========================================================
# 10. Parameters tab
# =========================================================
with tab_params:
    st.subheader("③ 감지된 파라미터와 코드 상태")

    if ok:
        st.markdown('<span class="ok-pill">Python code executed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="bad-pill">Execution error</span>', unsafe_allow_html=True)
        st.code(exec_error, language="text")

    if validation_errors:
        for item in validation_errors:
            if item.startswith("PARAM_CONFIG"):
                st.warning(item)
            elif "없습니다" in item and "PARAM_CONFIG" not in item:
                st.error(item)
            else:
                st.warning(item)
    else:
        st.markdown('<span class="ok-pill">Required structure detected</span>', unsafe_allow_html=True)

    st.markdown("### PARAM_CONFIG")
    if param_config:
        st.json(json_safe(param_config))
    else:
        st.info("감지된 PARAM_CONFIG가 없습니다.")

    st.markdown("### Current parameter values")
    st.json(json_safe(params))

    if st.button("🧪 Run smoke test", use_container_width=True):
        if not ok or any("필수 함수" in e for e in validation_errors):
            st.error("필수 함수가 부족해서 테스트할 수 없습니다.")
        else:
            try:
                run_smoke_test(namespace, params, runner)
                st.success("기본 실행 테스트 통과")
            except Exception:
                st.error("테스트 중 오류 발생")
                st.code(traceback.format_exc(), language="text")

# =========================================================
# 11. Preview tab
# =========================================================
with tab_preview:
    st.subheader("④ Preview")

    ready = ok and all(callable(namespace.get(name)) for name in ["init_state", "update_state", "draw_frame"])
    if not ready:
        st.error("필수 함수가 부족하거나 코드 오류가 있어 Preview를 실행할 수 없습니다.")
        if exec_error:
            st.code(exec_error, language="text")
    else:
        st.markdown('<span class="ok-pill">Ready to preview</span>', unsafe_allow_html=True)

    col_preview_btn, col_tip = st.columns([1, 1])
    with col_preview_btn:
        preview_button = st.button("🎬 Update Preview", use_container_width=True)
    with col_tip:
        st.info("Preview는 초기 상태에서 update_state를 지정 횟수만큼 실행한 뒤 draw_frame으로 한 장면을 그립니다.")

    if preview_button and ready:
        try:
            state = call_init_state(namespace["init_state"], params)
            for frame in range(int(preview_steps)):
                state = call_update_state(namespace["update_state"], state, params, frame)
            fig = render_frame(namespace, params, runner, state, int(preview_steps))
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.error("Preview 실행 오류")
            st.code(traceback.format_exc(), language="text")

# =========================================================
# 12. Save tab
# =========================================================
with tab_save:
    st.subheader("⑤ Save")
    st.markdown("저장 경로는 왼쪽 사이드바의 `Save root folder`에서 바꿀 수 있습니다.")
    st.code(
        f"GIF     : {paths['gif']}\nPNG     : {paths['png']}\nCODE    : {paths['code']}\nPARAMS  : {paths['params']}\nPROMPTS : {paths['prompts']}",
        language="text",
    )

    ready = ok and all(callable(namespace.get(name)) for name in ["init_state", "update_state", "draw_frame"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Save GIF", use_container_width=True):
            if not ready:
                st.error("코드가 준비되지 않아 GIF를 저장할 수 없습니다.")
            else:
                gif_path = paths["gif"] / f"{file_name}_{timestamp}.gif"
                code_path = paths["code"] / f"{file_name}_{timestamp}_code.py"
                param_path = paths["params"] / f"{file_name}_{timestamp}_params.json"
                try:
                    with st.spinner("GIF 저장 중... 프레임 수가 많으면 시간이 걸릴 수 있습니다."):
                        ani, fig = create_animation(namespace, params, runner)
                        ani.save(gif_path, writer=PillowWriter(fps=int(fps)))
                        plt.close(fig)
                    code_path.write_text(user_code, encoding="utf-8")
                    param_path.write_text(json.dumps(json_safe(params), indent=4, ensure_ascii=False), encoding="utf-8")
                    st.success("GIF 저장 완료")
                    st.write(gif_path)
                except Exception:
                    st.error("GIF 저장 오류")
                    st.code(traceback.format_exc(), language="text")

    with col2:
        if st.button("🖼️ Save PNG", use_container_width=True):
            if not ready:
                st.error("코드가 준비되지 않아 PNG를 저장할 수 없습니다.")
            else:
                png_path = paths["png"] / f"{file_name}_{timestamp}_preview.png"
                try:
                    state = call_init_state(namespace["init_state"], params)
                    for frame in range(int(preview_steps)):
                        state = call_update_state(namespace["update_state"], state, params, frame)
                    fig = render_frame(namespace, params, runner, state, int(preview_steps))
                    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
                    plt.close(fig)
                    st.success("PNG 저장 완료")
                    st.write(png_path)
                except Exception:
                    st.error("PNG 저장 오류")
                    st.code(traceback.format_exc(), language="text")

    with col3:
        if st.button("🐍 Save Code", use_container_width=True):
            code_path = paths["code"] / f"{file_name}_{timestamp}_code.py"
            try:
                code_path.write_text(user_code, encoding="utf-8")
                st.success("코드 저장 완료")
                st.write(code_path)
            except Exception:
                st.error("코드 저장 오류")
                st.code(traceback.format_exc(), language="text")

    with col4:
        if st.button("🧾 Save Params", use_container_width=True):
            param_path = paths["params"] / f"{file_name}_{timestamp}_params.json"
            try:
                param_path.write_text(json.dumps(json_safe(params), indent=4, ensure_ascii=False), encoding="utf-8")
                st.success("파라미터 저장 완료")
                st.write(param_path)
            except Exception:
                st.error("파라미터 저장 오류")
                st.code(traceback.format_exc(), language="text")

    st.markdown("---")
    if shutil.which("ffmpeg"):
        if st.button("🎞️ Try Save MP4", use_container_width=True):
            if not ready:
                st.error("코드가 준비되지 않아 MP4를 저장할 수 없습니다.")
            else:
                mp4_path = paths["gif"] / f"{file_name}_{timestamp}.mp4"
                try:
                    with st.spinner("MP4 저장 중..."):
                        ani, fig = create_animation(namespace, params, runner)
                        ani.save(mp4_path, writer=FFMpegWriter(fps=int(fps)))
                        plt.close(fig)
                    st.success("MP4 저장 완료")
                    st.write(mp4_path)
                except Exception:
                    st.error("MP4 저장 오류")
                    st.code(traceback.format_exc(), language="text")
    else:
        st.info("MP4 저장은 FFmpeg가 설치되어 있을 때만 활성화됩니다. 지금은 GIF 저장을 사용하면 됩니다.")

# =========================================================
# 13. Help tab
# =========================================================
with tab_help:
    st.subheader("⑥ 사용 설명")

    st.markdown(
        """
### 이 웹사이트가 하는 일

이 웹사이트는 특정 주제를 미리 알고 있지 않습니다.  
리소그래피, PN junction, 오비탈, 확산, 열전달, 결정격자 등 어떤 주제가 오더라도,
GPT가 주제에 맞게 작성한 코드를 실행하는 범용 플랫폼입니다.

### GPT가 작성해야 하는 코드 구조
"""
    )

    st.code(
        """
PARAM_CONFIG = {
    "parameter_name": {
        "type": "slider",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "step": 0.01,
        "description": "Meaning of this parameter"
    }
}

FIGURE_CONFIG = {
    "projection": "2d",
    "figsize": (8, 5.2),
    "dpi": 110
}

def init_state(params):
    state = {}
    return state

def update_state(state, params, frame):
    return state

def draw_frame(ax, state, params, frame):
    ax.plot([0, 1], [0, 1])
    ax.set_title("Example")
""",
        language="python",
    )

    st.markdown(
        """
### PARAM_CONFIG type 목록

- `slider`: 슬라이더
- `number`: 직접 숫자 입력
- `int`: 정수 입력
- `checkbox`: True/False 체크박스
- `color`: 색상 선택기
- `text`: 글자 입력
- `select`: 선택 박스, `options` 필요
- `log_slider`: 로그 스케일 슬라이더

### 3D가 필요할 때

GPT 코드에 아래 설정을 포함하면 됩니다.
"""
    )

    st.code(
        """
FIGURE_CONFIG = {
    "projection": "3d",
    "figsize": (8, 6),
    "dpi": 110
}

# draw_frame 안에서
# ax.scatter(x, y, z)
# ax.set_zlabel("Z")
""",
        language="python",
    )

    st.markdown(
        """
### 오류가 날 때 확인할 것

1. `PARAM_CONFIG`가 dictionary인지 확인합니다.
2. `init_state`, `update_state`, `draw_frame` 함수 이름이 정확한지 확인합니다.
3. GPT가 `plt.figure()`나 `plt.show()`를 쓰면 안 됩니다. 반드시 `ax`에 그려야 합니다.
4. 외부 파일을 읽거나 쓰는 코드는 넣지 않는 것이 좋습니다.
5. 3D 그래프인데 그림이 이상하면 왼쪽 `Figure projection`을 `3d`로 바꿔보세요.

### 안전 주의

붙여넣은 코드는 현재 컴퓨터에서 실행됩니다. 직접 이해할 수 있는 코드나 신뢰할 수 있는 코드만 실행하세요.
"""
    )
